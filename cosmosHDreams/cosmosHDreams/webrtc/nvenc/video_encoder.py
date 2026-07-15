"""VideoEncoder abstraction for the WebRTC video pipeline.

The :class:`VideoEncoder` Protocol pairs a concrete encoder
implementation with the :class:`MediaStreamTrack` type it produces
frames for. Two implementations live in this module:

  * :class:`PyNvHardwareEncoder` — GPU-accelerated NVENC H.264 via
    :class:`~cosmosHDreams.webrtc.nvenc.encoder.CosmoshNvencH264`. Owns the
    NAL queue drained by the aiortc adapter
    (:mod:`cosmosHDreams.webrtc.nvenc.aiortc_compat`), installs / uninstalls
    the encoder shim, and emits ``CosmoshNvencVideoTrack`` markers so
    aiortc's sender pulls one encoded frame per marker.
  * :class:`DefaultRTCVideoEncoder` — aiortc's built-in software
    encoder path. Wraps a :class:`~cosmosHDreams.webrtc.media.CosmoshVideoTrack`
    that carries raw RGB frames; aiortc's own sender loop drives
    per-frame encoding via ``H264Encoder.encode`` / ``VpxEncoder.encode``,
    and the concrete codec is chosen internally by SDP negotiation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from aiortc import MediaStreamTrack  # type: ignore[import-untyped]

from cosmosHDreams.webrtc.nvenc.aiortc_compat import (
    install_nvenc_encoder,
    restrict_codecs_to_h264,
    uninstall_nvenc_encoder,
)
from cosmosHDreams.webrtc.nvenc.encoder import CosmoshNvencH264, NalFrame, NvencConfig
from cosmosHDreams.webrtc.nvenc.resolver import probe_nvenc_available

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ChunkDeliveryResult:
    """Uniform shape returned by :meth:`VideoEncoder.deliver_chunk`.

    Downstream telemetry / logging code uses this rather than the
    encoder-internal ``NalFrame`` / ``EncodedVideoPacket`` types so it
    doesn't need to know which backend produced the chunk.

    Fields:
        backend: short backend identifier (``"pynvvideocodec"`` /
            ``"aiortc"``). Not user-configurable — pins the wire log
            to whichever encoder is actually running.
        num_frames: number of frames actually enqueued on the track.
        num_keyframes: number of IDR frames emitted. Only meaningful
            on the NVENC path; ``0`` on the SW path (aiortc's SW
            encoder decides keyframe cadence internally).
        encode_ms: wall-clock encode time in milliseconds. ``0.0`` on
            the SW path (encode happens later, inside aiortc's sender).
    """

    backend: str
    num_frames: int
    num_keyframes: int
    encode_ms: float


@runtime_checkable
class VideoEncoder(Protocol):
    """Encoder backend paired with a compatible :class:`MediaStreamTrack`.

    Each backend owns two responsibilities:

    - Creating a fresh media track sized for one session
      (:meth:`create_track`).
    - Encoding + enqueueing one chunk of frames onto that track
      (:meth:`deliver_chunk`).

    The concrete pairing — raw RGB frames + aiortc SW encoder vs.
    marker frames + NVENC + shim-driven H.264 packetisation — is
    entirely encapsulated inside the implementation. Callers pick one
    backend at startup and branch nowhere else.
    """

    fps: int
    backend: str
    prefers_codec: Optional[str]
    """SDP codec preference hint. ``"h264"`` for NVENC (must be
    honoured — the shim substitutes only the H.264 encoder, so a
    negotiated VP8 payload type would trip the ``get_encoder``
    dispatch); ``None`` to let aiortc pick its default codec."""

    def create_track(self) -> MediaStreamTrack:
        """Create a fresh session track compatible with this encoder."""
        ...

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        """Encode ``chunk`` and enqueue the result on ``track``."""
        ...

    def reset_session(self) -> None:
        """Restart internal encoder state on rollout reset / scene-switch.

        Guarantees the next :meth:`deliver_chunk` call starts from a
        clean slate (fresh NVENC session for the HW path; a no-op for
        the SW path since aiortc's encoder is re-instantiated per
        :class:`RTCPeerConnection` from our POV).
        """
        ...

    def close(self) -> None:
        """Release encoder-owned resources. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# PyNvVideoCodec (NVENC) hardware encoder implementation.
# ---------------------------------------------------------------------------


class PyNvHardwareEncoder:
    """VideoEncoder backed by :class:`CosmoshNvencH264` + the aiortc shim.

    Owns one NVENC session for the lifetime of the containing rollout
    (recreated on :meth:`reset_session`) plus the NAL queue drained by
    the aiortc adapter. Installs the shim at construction and
    uninstalls it on :meth:`close`, so the caller has a single object
    to hand off ownership of both the encoder and the aiortc
    monkey-patch.

    The paired track type is
    :class:`~cosmosHDreams.webrtc.nvenc.track.CosmoshNvencVideoTrack`
    (marker-only). :meth:`deliver_chunk` pushes one
    :class:`NalFrame` per encoded input frame to the NAL queue and
    then enqueues that same count of markers so the two queues stay
    1:1.
    """

    backend: str = "pynvvideocodec"
    prefers_codec: Optional[str] = "h264"

    @classmethod
    def is_supported(
        cls,
        *,
        gpu_id: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> tuple[bool, str]:
        """Check whether NVENC H.264 is usable on this machine.

        Thin classmethod wrapper around
        :func:`~cosmosHDreams.webrtc.nvenc.resolver.probe_nvenc_available`
        so callers that already hold a reference to the encoder class
        (session managers, factory helpers) can query capability
        without importing the resolver module separately.

        Args:
            gpu_id: GPU device index to probe (matches the value
                passed to :class:`PyNvHardwareEncoder`).
            width, height: Optional target resolution. If both > 0,
                the deep probe (``GetEncoderCaps``) is used to verify
                the target resolution falls within driver-reported
                bounds. Otherwise only the shallow import + CUDA
                availability check runs.

        Returns:
            ``(True, reason)`` if the environment can create an NVENC
            H.264 session at the requested resolution;
            ``(False, reason)`` otherwise. ``reason`` is a
            human-readable diagnostic suitable for logging.
        """
        return probe_nvenc_available(
            gpu_id=gpu_id,
            width=width,
            height=height,
        )

    def __init__(self, config: NvencConfig) -> None:
        self._config = config
        self.fps: int = config.fps
        # Session encoder — constructed eagerly so any NVENC driver /
        # capability problem surfaces here, not at the first render.
        self._encoder: CosmoshNvencH264 = CosmoshNvencH264(config)
        # NAL queue drained by the aiortc shim on its sender thread.
        # One-per-encoded-frame lockstep with track marker enqueue.
        self._nal_queue: "queue.Queue[NalFrame]" = queue.Queue()
        # Install the shim now so the first RTCPeerConnection sees
        # ``NvencH264Encoder`` as its H.264 encoder class. Idempotent
        # across encoders in the same process.
        install_nvenc_encoder(self._nal_queue)
        restrict_codecs_to_h264()
        self._closed: bool = False

    @property
    def config(self) -> NvencConfig:
        return self._config

    @property
    def nal_queue(self) -> "queue.Queue[NalFrame]":
        """The NAL queue drained by the aiortc shim. Runtime uses this
        for the anchor-frame push in :meth:`initial_frame_chunk`."""
        return self._nal_queue

    def raw_encoder(self) -> CosmoshNvencH264:
        """Underlying :class:`CosmoshNvencH264` for the anchor-encode path.

        The session runtime's :meth:`initial_frame_chunk` needs direct
        access to :meth:`CosmoshNvencH264.encode_chunk` so it can run
        the submit-until-SPS loop that discriminates the anchor IDR
        from pipeline-buffered leftovers. Exposed as a method (not
        attribute) to make the coupling explicit.
        """
        return self._encoder

    def create_track(self) -> MediaStreamTrack:
        # Lazy import — track pulls ``av``, and we want the module
        # importable on hosts without PyAV for pure-Python tests.
        from cosmosHDreams.webrtc.nvenc.track import CosmoshNvencVideoTrack

        return CosmoshNvencVideoTrack(fps=self.fps)

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        from cosmosHDreams.webrtc.nvenc.pack import denormalize_and_pack_argb
        from cosmosHDreams.webrtc.nvenc.track import CosmoshNvencVideoTrack

        if not isinstance(track, CosmoshNvencVideoTrack):
            raise TypeError(
                "PyNvHardwareEncoder requires a CosmoshNvencVideoTrack; "
                f"got {type(track).__name__}. Create the track via "
                "encoder.create_track()."
            )

        def _encode_sync() -> tuple[list[NalFrame], float]:
            t0 = time.perf_counter()
            argb = denormalize_and_pack_argb(chunk)
            frames = self._encoder.encode_chunk(argb, force_idr=force_keyframe)
            return frames, (time.perf_counter() - t0) * 1000.0

        nal_frames, encode_ms = await asyncio.to_thread(_encode_sync)

        for nf in nal_frames:
            self._nal_queue.put(nf)
        enqueued = await track.enqueue_markers(len(nal_frames))

        num_keyframes = 1 if force_keyframe and nal_frames else 0
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=num_keyframes,
            encode_ms=encode_ms,
        )

    def reset_session(self) -> None:
        self._encoder.reset_session()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._encoder.close()
        with contextlib.suppress(Exception):
            uninstall_nvenc_encoder()


# ---------------------------------------------------------------------------
# aiortc-default (CPU / libavcodec) encoder implementation.
# ---------------------------------------------------------------------------


class DefaultRTCVideoEncoder:
    """VideoEncoder path via aiortc's built-in PyAV/FFmpeg encoder.

    This class does not encode itself — it wraps a
    :class:`~cosmosHDreams.webrtc.media.CosmoshVideoTrack` that carries raw
    RGB frames and lets aiortc's own RTP sender loop drive encoding
    frame-by-frame. The concrete codec is picked by aiortc's SDP
    negotiation (H.264 or VP8 depending on the browser's offer);
    ``deliver_chunk`` therefore reduces to enqueueing the chunk's
    frames onto the track.
    """

    backend: str = "aiortc"
    prefers_codec: Optional[str] = None

    def __init__(self, *, fps: int) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0; got {fps}")
        self.fps: int = fps

    def create_track(self) -> MediaStreamTrack:
        from cosmosHDreams.webrtc.media import CosmoshVideoTrack

        return CosmoshVideoTrack(fps=self.fps)

    async def deliver_chunk(
        self,
        chunk: torch.Tensor,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        # ``force_keyframe`` is a no-op here: aiortc's SW encoder
        # decides keyframe cadence internally and responds to receiver
        # PLI / FIR feedback, so an out-of-band request from us is
        # neither possible nor needed.
        del force_keyframe
        from cosmosHDreams.webrtc.media import CosmoshVideoTrack

        if not isinstance(track, CosmoshVideoTrack):
            raise TypeError(
                "DefaultRTCVideoEncoder requires a CosmoshVideoTrack; "
                f"got {type(track).__name__}. Create the track via "
                "encoder.create_track()."
            )
        enqueued, _cast_ms = await track.enqueue_chunk(chunk)
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=0,
            encode_ms=0.0,
        )

    def reset_session(self) -> None:
        # No encoder-owned resources; aiortc's encoder is
        # re-instantiated per RTCPeerConnection.
        return

    def close(self) -> None:
        return


__all__ = [
    "ChunkDeliveryResult",
    "DefaultRTCVideoEncoder",
    "PyNvHardwareEncoder",
    "VideoEncoder",
]
