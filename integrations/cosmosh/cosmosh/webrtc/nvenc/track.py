"""Video track for the NVENC encode path.

Drop-in replacement for :class:`cosmosh.webrtc.media.CosmoshVideoTrack`
when ``video.encoder=nvenc``. Differences from the CPU track:

  - The track does **not** carry pixel data. NAL units flow through a
    separate :class:`queue.Queue` owned by the aiortc adapter
    (:mod:`cosmosh.webrtc.nvenc.aiortc_compat`). This track only
    dispenses *frame markers* that drive aiortc's per-frame
    ``recv()`` cadence.

  - ``recv()`` returns a fixed-size placeholder :class:`av.VideoFrame`
    (16×16 zeros). aiortc's sender uses it as the "input" to
    :meth:`NvencH264Encoder.encode`, which discards the pixel buffer
    and pulls a real NAL list from the encoder queue. The placeholder
    only needs to be a valid VideoFrame so the type checks in aiortc's
    sender path don't trip.

  - Pacing is identical to the CPU track: 1/fps wall-clock deadline,
    re-anchor on stall, single-frame buffer pull per call. Browser
    jitter-buffer behaviour stays unchanged.

Lockstep contract with :mod:`cosmosh.webrtc.nvenc.aiortc_compat`:
  The runtime fills the encoder's NAL queue first (inside
  :meth:`CosmoshNvencH264.encode_chunk`), then calls
  :meth:`enqueue_chunk` on this track to publish N markers. aiortc
  consumes markers and NalFrames 1:1, in order. Marker count must
  never exceed pushed NalFrame count.
"""
from __future__ import annotations

import asyncio
import logging
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack  # type: ignore[import-untyped]
from aiortc.mediastreams import MediaStreamError  # type: ignore[import-untyped]
from av import VideoFrame  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)


# Same thresholds as :mod:`cosmosh.webrtc.media`; keeping them in sync
# means the two tracks emit identical pacing-stall log lines, which
# makes A/B traces directly comparable.
_STALL_THRESHOLD_MS: float = 1.0
_PACING_LAG_LOG_MS: float = 5.0


# A single placeholder frame buffer is shared across all ``recv()``
# calls in a session. aiortc never inspects its bytes (the
# NvencH264Encoder discards the frame's pixel data), so 16×16 zeros
# are sufficient — and the small footprint sidesteps any chance of
# accidentally consuming bandwidth on a wrong-encoder fallback path.
_PLACEHOLDER_BUF: np.ndarray = np.zeros((16, 16, 3), dtype=np.uint8)


class CosmoshNvencVideoTrack(MediaStreamTrack):
    """Marker-only video track for the NVENC encode path.

    The constructor signature matches :class:`CosmoshVideoTrack` so
    session.py can swap implementations based on the resolved encoder
    without further plumbing.
    """

    kind = "video"

    def __init__(self, fps: int = 10) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._fps: int = fps
        self._time_base: Fraction = Fraction(1, fps)
        self._frame_interval_s: float = 1.0 / fps
        self._next_deadline_s: float | None = None
        self._pts: int = 0
        # Asyncio queue — both producer (session.py render loop) and
        # consumer (aiortc's sender via ``recv``) run on the asyncio
        # thread.  Sentinel ``None`` signals shutdown.
        self._markers: asyncio.Queue[int | None] = asyncio.Queue()
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Producer side — called by the runtime / session.py render loop.
    # ------------------------------------------------------------------

    async def enqueue_markers(self, num_frames: int) -> int:
        """Publish ``num_frames`` markers; aiortc will pull that many
        NalFrames from the encoder queue in lockstep.

        Returns the number of markers actually enqueued (0 if the
        track is closed or ``num_frames`` is non-positive). Distinct
        name from :meth:`CosmoshVideoTrack.enqueue_chunk` so the
        session manager can dispatch on track type without an
        isinstance check.
        """
        if self._closed or num_frames <= 0:
            return 0
        for _ in range(num_frames):
            await self._markers.put(1)
        return num_frames

    def qsize(self) -> int:
        """Number of markers buffered but not yet consumed by ``recv``."""
        return self._markers.qsize()

    def drain_recv_stats(self) -> dict[str, float]:
        """Return per-chunk recv-wait / pacing stats and reset accumulators.

        Interface-compat stub matching :meth:`CosmoshVideoTrack.drain_recv_stats`
        so the shared latency logger can consume either backend uniformly.
        Zero-valued today; wire in real accumulators if per-frame NVENC-path
        pacing telemetry becomes useful.
        """
        return {"recv_wait_ms": 0.0, "pacing_ms": 0.0}

    def drain_pending(self) -> int:
        """Discard any pending markers.

        Called on session reset and scene-switch so the browser does
        not see stale motion from the pre-reset rollout. The encoder
        queue is drained separately by the runtime; the lockstep
        contract holds because both queues are drained from the same
        lock-protected region in session.py.
        """
        dropped = 0
        while True:
            try:
                self._markers.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
        return dropped

    # ------------------------------------------------------------------
    # Consumer side — called by aiortc's RTCRtpSender (async).
    # ------------------------------------------------------------------

    async def recv(self) -> VideoFrame:
        if self._closed:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        t_get_start = loop.time()
        marker = await self._markers.get()
        if marker is None:
            # Shutdown sentinel — propagate the stream-end signal aiortc expects.
            raise MediaStreamError
        get_wait_ms = (loop.time() - t_get_start) * 1000.0

        # Pacing logic is bit-identical to ``CosmoshVideoTrack.recv``
        # so A/B traces produce comparable stall / lag warnings.
        first_frame = self._next_deadline_s is None
        just_stalled = (not first_frame) and get_wait_ms > _STALL_THRESHOLD_MS
        if just_stalled:
            LOGGER.warning(
                "Playback stall: pts=%d waited %.1fms for next marker; "
                "queue depth now %d.",
                self._pts, get_wait_ms, self._markers.qsize(),
            )

        now_s = loop.time()
        if first_frame or just_stalled:
            self._next_deadline_s = now_s
        else:
            proposed = self._next_deadline_s + self._frame_interval_s
            wait_s = proposed - now_s
            if wait_s > 0:
                await asyncio.sleep(wait_s)
                self._next_deadline_s = proposed
            else:
                if -wait_s * 1000.0 > _PACING_LAG_LOG_MS:
                    LOGGER.warning(
                        "Pacing lag: pts=%d deadline %.1fms behind "
                        "walltime; re-anchoring (queue depth %d).",
                        self._pts, -wait_s * 1000.0, self._markers.qsize(),
                    )
                self._next_deadline_s = now_s

        frame = VideoFrame.from_ndarray(_PLACEHOLDER_BUF, format="rgb24")
        # frame.pts here is informational — the NvencH264Encoder shim
        # returns its own pts (the NVENC-assigned 90 kHz RTP
        # timestamp from the NalFrame), so this value is not what
        # appears on the wire. We still set a monotonic value so
        # debug logs that inspect the placeholder are well-ordered.
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Idempotent shutdown. Wakes any pending ``recv`` so aiortc
        sees the stream-end signal promptly."""
        if self._closed:
            return
        self._closed = True
        await self._markers.put(None)
        self.stop()


__all__ = ["CosmoshNvencVideoTrack"]
