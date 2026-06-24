from __future__ import annotations

import asyncio
import logging
import time
from fractions import Fraction

import numpy as np
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

LOGGER = logging.getLogger(__name__)
"""Module logger; warns on playback stalls so we can correlate with
session-level chunk timing in :mod:`cosmosh.webrtc.session`."""

_STALL_THRESHOLD_MS = 1.0
"""Minimum ``await get()`` wait in milliseconds that we treat as a stall.
In steady state the queue is non-empty when ``recv`` arrives so ``get``
returns instantly; anything above ~1ms means generation did not keep
ahead of playback for this frame."""

_PACING_LAG_LOG_MS = 5.0
"""Below this lag we re-anchor pacing silently. Above it the lag is
worth a one-line warning so bursts (which the browser jitter buffer
turns into visible playback speed-ups) are correlatable in the log."""


def tensor_chunk_to_rgb_frames(video_chunk: torch.Tensor) -> list[np.ndarray]:
    """Convert Cosmosh per-block output ``[B, C, T, H, W]`` in ``[-1, 1]`` to RGB uint8 frames.

    The runtime returns the 12 generated frames (not including the conditional
    anchor) in the same ``[B, C, T, H, W]`` layout used by run_cosmosh.py's
    final video buffer. Mapping mirrors run_cosmosh.py:506-513.
    """
    if video_chunk.ndim != 5:
        raise ValueError(
            f"Expected video chunk with 5 dimensions [B, C, T, H, W], got {video_chunk.shape}"
        )
    if video_chunk.shape[0] < 1:
        raise ValueError("Video chunk must contain at least one batch element.")

    # [B, C, T, H, W] → [T, H, W, C] for batch 0.
    frames = video_chunk[0].float().permute(1, 2, 3, 0).numpy()
    frames = ((frames + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return [np.ascontiguousarray(frame) for frame in frames]


def _timed_tensor_chunk_to_rgb_frames(video_chunk: torch.Tensor) -> tuple[list[np.ndarray], float]:
    """Wrap ``tensor_chunk_to_rgb_frames`` with wall-clock timing.

    Returns ``(frames, elapsed_ms)`` where ``elapsed_ms`` is the time spent
    in the float→uint8 cast (the part that runs in a worker thread).
    """
    t0 = time.perf_counter()
    frames = tensor_chunk_to_rgb_frames(video_chunk)
    return frames, (time.perf_counter() - t0) * 1000.0


class CosmoshVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, fps: int = 10) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._fps = fps
        self._time_base = Fraction(1, fps)
        self._frame_interval_s = 1.0 / fps
        self._next_deadline_s: float | None = None
        self._pts = 0
        self._frames: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._closed = False
        self._recv_wait_ms: list[float] = []
        self._pacing_ms: list[float] = []

    async def enqueue_chunk(self, video_chunk: torch.Tensor) -> tuple[int, float]:
        # Offload the float->uint8 cast to a worker thread so it doesn't
        # stall the asyncio loop and starve ``recv``'s 1/fps pacing.
        # When this ran inline it was the single biggest source of the
        # empty-queue stalls that ``recv`` then has to re-anchor around.
        frames, cast_ms = await asyncio.to_thread(_timed_tensor_chunk_to_rgb_frames, video_chunk)
        for frame in frames:
            await self._frames.put(frame)
        return len(frames), cast_ms

    def qsize(self) -> int:
        """Number of frames buffered but not yet sent over the wire."""
        return self._frames.qsize()

    def drain_pending(self) -> int:
        """Discard any enqueued frames not yet consumed by ``recv``."""
        dropped = 0
        while True:
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
        return dropped

    async def recv(self) -> VideoFrame:
        if self._closed:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        t_get_start = loop.time()
        frame_array = await self._frames.get()
        if frame_array is None:
            raise MediaStreamError
        get_wait_ms = (loop.time() - t_get_start) * 1000.0
        self._recv_wait_ms.append(get_wait_ms)
        # ``_next_deadline_s is None`` is the single source of truth for
        # "we haven't emitted any frame yet". The pre-first-frame wait
        # is the time aiortc spends calling ``recv`` before the producer
        # has generated anything; it is expected, not a stall.
        first_frame = self._next_deadline_s is None
        just_stalled = (not first_frame) and get_wait_ms > _STALL_THRESHOLD_MS
        if just_stalled:
            LOGGER.warning(
                "Playback stall: pts=%d waited %.1fms for next frame; queue depth now %d.",
                self._pts,
                get_wait_ms,
                self._frames.qsize(),
            )

        now_s = loop.time()
        if first_frame or just_stalled:
            # First frame, or recovering from a queue stall: anchor pacing
            # at ``now`` instead of adding ``frame_interval_s`` to a stale
            # absolute deadline. The catch-up behaviour (``wait_s`` deeply
            # negative for several consecutive recvs) burst-drains the
            # queue in microseconds, which collapses the smooth RTP cadence
            # the browser jitter buffer expects and makes the *next* chunk
            # look like another empty-queue stall, even when generation
            # outpaces playback — the sawtooth pattern visible in the logs.
            self._next_deadline_s = now_s
            self._pacing_ms.append(0.0)
        else:
            proposed = self._next_deadline_s + self._frame_interval_s
            wait_s = proposed - now_s
            if wait_s > 0:
                _t_sleep_start = loop.time()
                await asyncio.sleep(wait_s)
                self._pacing_ms.append((loop.time() - _t_sleep_start) * 1000.0)
                self._next_deadline_s = proposed
            else:
                # Queue had a frame ready (no stall) but our deadline is
                # already in the past — typical causes are aiortc's send
                # loop lagging, ``asyncio.sleep`` over-sleeping, or another
                # task hogging the loop. Without re-anchoring, subsequent
                # recv()s also see ``wait_s < 0`` and burst the queue at
                # aiortc's pull rate, which the browser jitter buffer
                # turns into a visible playback speed-up. Anchor at
                # ``now_s`` so the next frame resumes 1/fps cadence.
                if -wait_s * 1000.0 > _PACING_LAG_LOG_MS:
                    LOGGER.warning(
                        "Pacing lag: pts=%d deadline %.1fms behind walltime; "
                        "re-anchoring to avoid burst (queue depth %d).",
                        self._pts,
                        -wait_s * 1000.0,
                        self._frames.qsize(),
                    )
                self._next_deadline_s = now_s
                self._pacing_ms.append(0.0)

        frame = VideoFrame.from_ndarray(frame_array, format="rgb24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame

    def drain_recv_stats(self) -> dict[str, float]:
        """Return per-chunk average recv wait and pacing, then reset accumulators."""
        def _avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0
        stats = {"recv_wait_ms": _avg(self._recv_wait_ms), "pacing_ms": _avg(self._pacing_ms)}
        self._recv_wait_ms.clear()
        self._pacing_ms.clear()
        return stats

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._frames.put(None)
        self.stop()
