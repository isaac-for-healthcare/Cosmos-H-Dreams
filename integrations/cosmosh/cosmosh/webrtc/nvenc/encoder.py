"""Thin wrapper around PyNvVideoCodec NVENC for the CosmosH WebRTC server.

Owns one NVENC session for the lifetime of a streaming session. Construction
triggers the deferred ``import PyNvVideoCodec``; on non-NVENC hosts callers
should rely on :func:`cosmosh.webrtc.nvenc.resolver.resolve_encoder` to avoid
ever instantiating this class.

The wrapper accepts ARGB GPU tensors as produced by
:func:`cosmosh.webrtc.nvenc.pack.denormalize_and_pack_argb` and yields one
:class:`NalFrame` per generated pixel frame, each carrying the Annex-B
NAL units for that frame.

PyNvVideoCodec 2.1+ exposes a ``CreateEncoder`` factory and an ``Encode``
method that accepts a CUDA-backed surface (torch tensor on a CUDA device).
The exact entry-point names and accepted kwargs have varied between minor
releases — :meth:`CosmoshNvencH264._encode_one` is the single place that
touches that variance.

Keyframe (IDR) policy
---------------------
- The encoder is configured with ``idr_period_s`` seconds between IDR
  frames (``idr_period_frames`` at the encoder's fps). Default 4 s.
- :meth:`encode_chunk(..., force_idr=True)` forces the first frame of the
  passed chunk to be an IDR. The runtime sets this on session start, reset,
  and scene-switch. The natural IDR period bounds recovery latency to
  ``idr_period_s`` if the browser drops a frame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


# NVENC NV_ENC_PIC_FLAGS values (from NVIDIA Video Codec SDK).
# PyNvVideoCodec's ``Encode(surface, picFlags)`` accepts the bitwise-OR
# of these values as the second positional argument.
_NVENC_PIC_FLAG_FORCEIDR: int = 0x01
_NVENC_PIC_FLAG_OUTPUT_SPSPPS: int = 0x04


# Combined picFlags for the "force a keyframe with SPS+PPS" path used on
# session start, reset, scene-switch, and aiortc-driven PLI. SPS+PPS is
# included alongside the IDR so a browser that connects mid-stream can
# initialise the decoder immediately rather than waiting for the next
# natural IDR (every ``idr_period_frames`` frames).
_NVENC_PIC_FLAGS_KEYFRAME: int = (
    _NVENC_PIC_FLAG_FORCEIDR | _NVENC_PIC_FLAG_OUTPUT_SPSPPS
)


@dataclass(frozen=True)
class NvencConfig:
    """NVENC profile passed to :class:`CosmoshNvencH264`.

    Defaults match the unified_tabletop tuning recommended in the
    optimization design doc (low-latency H.264, CBR, IDR every 4 s).
    """

    width: int
    height: int
    fps: int
    bitrate: int = 3_000_000
    preset: str = "P3"
    tuning: str = "ultra_low_latency"
    idr_period_s: float = 4.0
    gpu_id: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"NvencConfig invalid resolution: {self.width}x{self.height}"
            )
        if self.width % 2 or self.height % 2:
            # NVENC H.264 encoders require even dimensions for YUV420
            # chroma subsampling. CosmosH ships at 288x512 which is fine,
            # but catch typos at construction.
            raise ValueError(
                f"NvencConfig resolution must be even: {self.width}x{self.height}"
            )
        if self.fps <= 0:
            raise ValueError(f"NvencConfig invalid fps: {self.fps}")
        if self.bitrate <= 0:
            raise ValueError(f"NvencConfig invalid bitrate: {self.bitrate}")
        if self.idr_period_s <= 0:
            raise ValueError(
                f"NvencConfig invalid idr_period_s: {self.idr_period_s}"
            )

    @property
    def idr_period_frames(self) -> int:
        """IDR cadence expressed in frames (NVENC's native unit)."""
        return max(1, int(round(self.fps * self.idr_period_s)))


@dataclass(frozen=True)
class NalFrame:
    """One encoded video frame ready for the aiortc shim.

    ``nal_units`` are raw Annex-B NAL bytes (no start codes); the
    adapter's :class:`NvencH264Encoder` packetizes them via aiortc's
    RFC 6184 helper (``H264Encoder._packetize``) before they hit the
    RTP sender. The RTP timestamp is *not* carried in this struct —
    the adapter derives it from the placeholder ``av.VideoFrame``'s
    ``pts`` via ``convert_timebase``, exactly as aiortc's stock CPU
    H264 encoder does.
    """

    nal_units: list[bytes]


def _split_annexb(blob: bytes) -> list[bytes]:
    """Split an Annex-B byte stream into NAL units (no start codes).

    Recognises both 3-byte (``00 00 01``) and 4-byte (``00 00 00 01``)
    start codes per H.264 Annex-B. Returns an empty list for empty
    input — useful when NVENC buffers frames internally and returns
    nothing on a given submit.

    Slicing is exact: each NAL extends from the byte after its own
    start code up to the byte before the next start code (or to EOF
    for the last NAL). The H.264 emulation-prevention rule
    (``00 00 [01|02|03]`` cannot appear inside a NAL payload — the
    encoder inserts an ``0x03`` byte to escape) guarantees the
    in-stream lookahead never false-positives inside a NAL.
    """
    if not blob:
        return []
    n = len(blob)
    # Record (start_code_offset, start_code_length) per NAL boundary,
    # not the data-start offset. Using the *next* start-code offset as
    # the slice endpoint avoids leaking trailing zero+01 bytes of the
    # following start code into the preceding NAL.
    boundaries: list[tuple[int, int]] = []
    i = 0
    while i < n - 2:
        if blob[i] == 0 and blob[i + 1] == 0:
            if blob[i + 2] == 1:
                boundaries.append((i, 3))
                i += 3
                continue
            if i + 3 < n and blob[i + 2] == 0 and blob[i + 3] == 1:
                boundaries.append((i, 4))
                i += 4
                continue
        i += 1
    if not boundaries:
        return []
    out: list[bytes] = []
    for k, (sc_pos, sc_len) in enumerate(boundaries):
        data_start = sc_pos + sc_len
        data_end = boundaries[k + 1][0] if k + 1 < len(boundaries) else n
        unit = bytes(blob[data_start:data_end])
        if unit:
            out.append(unit)
    return out


class CosmoshNvencH264:
    """Owns one NVENC session for the lifetime of a streaming session.

    Lifecycle:
      1. ``__init__``: import PyNvVideoCodec, create the NVENC session.
      2. ``encode_chunk(argb_chunk, force_idr=False)``: submit one chunk.
         Returns a :class:`NalFrame` per input frame (NVENC's input/output
         cadence is 1:1 with low-latency tuning and B-frames disabled).
      3. ``flush()``: drain any internally buffered output.
      4. ``close()``: idempotent; releases NVENC resources.

    The class is *not* thread-safe; callers must serialise
    ``encode_chunk`` with their own lock (the CosmosH render loop already
    holds the step lock when calling into the runtime, which transitively
    serialises encoding).
    """

    def __init__(self, config: NvencConfig) -> None:
        # Deferred import — keep nvenc/encoder.py importable on hosts
        # without the PyNvVideoCodec wheel.
        try:
            import PyNvVideoCodec as nvc  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "PyNvVideoCodec is required for video.encoder=nvenc."
            ) from exc

        self._nvc = nvc
        self.config = config
        self._closed: bool = False

        # PyNvVideoCodec ``CreateEncoder`` takes 4 positional args
        # (width, height, fmt, usecpuinputbuffer) plus a ``**kwargs``
        # tail for encoder configuration. The positional ``fmt`` is the
        # *input pixel format* (ARGB); the output codec is selected
        # via the ``codec`` kwarg.
        #
        # Kwarg names below are taken verbatim from the PyNvVideoCodec
        # 2.1 Programming Guide ("Optional Parameters for CreateEncoder").
        # If a future release renames or removes any of them, this is
        # the single place to update.
        #
        # NOTE on profile: PyNvVideoCodec 2.x does *not* expose an
        # H.264 profile-selection kwarg. NVENC emits H.264 High by
        # default. We instead make the SDP match what NVENC emits —
        # see ``aiortc_compat._add_h264_high_codec_capability``.
        self._kwargs: dict[str, Any] = {
            "codec": "h264",
            "preset": config.preset,
            "tuning_info": config.tuning,
            "rc": "cbr",
            "bitrate": str(config.bitrate),
            "fps": str(config.fps),
            "gop": str(config.idr_period_frames),
            "idrperiod": str(config.idr_period_frames),
            # Browsers default to BT.709 for modern WebRTC video.
            # NVENC needs ``colorspace`` set explicitly for ARGB/ABGR
            # inputs (per the Programming Guide) so the ARGB→YUV
            # conversion uses the matching matrix.
            "colorspace": "bt709",
            # Repeat SPS/PPS at every IDR. Without this, a browser
            # that connects mid-stream (or after a network blip) has
            # no decoder state and shows a black canvas until the
            # next natural IDR — which is up to ``idr_period_s`` away.
            "repeatspspps": 1,
        }
        self._enc = self._open_nvenc_session()

    def _open_nvenc_session(self) -> Any:
        """Construct a fresh NVENC session from ``self.config`` / ``self._kwargs``.

        Called from ``__init__`` and from :meth:`reset_session` so the
        ``CreateEncoder`` invocation lives in exactly one place.
        """
        enc = self._nvc.CreateEncoder(
            self.config.width,
            self.config.height,
            "ARGB",
            False,        # usecpuinputbuffer — we feed CUDA tensors
            **self._kwargs,
        )
        LOGGER.info(
            "H.264 encoder created on GPU %d: %dx%d @ %d fps · "
            "bitrate %d bps · preset %s · tuning %s · IDR period %d frames",
            self.config.gpu_id, self.config.width, self.config.height,
            self.config.fps, self.config.bitrate, self.config.preset,
            self.config.tuning, self.config.idr_period_frames,
        )
        return enc

    def reset_session(self) -> None:
        """Tear down and re-create the NVENC session.

        Discards any frames buffered inside NVENC's pipeline so a
        subsequent encode starts from a known-empty state. Called on
        rollout reset / scene-switch so stale P-frames from the prior
        rollout cannot leak into the new anchor's bitstream. The caller
        is responsible for draining the runtime-owned NAL queue
        separately (the encoder doesn't own it).
        """
        if self._closed:
            raise RuntimeError("NVENC encoder is closed")
        try:
            self._enc.EndEncode()
        except Exception:  # noqa: BLE001
            LOGGER.exception(
                "EndEncode raised during reset_session; continuing with recreate"
            )
        try:
            del self._enc
        except Exception:
            LOGGER.exception("encoder reference cleanup failed in reset_session")
        self._enc = self._open_nvenc_session()

    @property
    def closed(self) -> bool:
        return self._closed

    def encode_chunk(
        self,
        argb_chunk: torch.Tensor,
        *,
        force_idr: bool = False,
    ) -> list[NalFrame]:
        """Encode a chunk of ARGB frames; return one NalFrame per input frame.

        Args:
            argb_chunk: ``[T, H, W, 4]`` ``torch.uint8`` ARGB on the same
                GPU as the encoder. Channel 0 is alpha (NVENC ignores it
                but it must be present; the pack kernel sets it to 255).
            force_idr: if True, the first frame of the chunk is encoded
                as an IDR. Used by the runtime on session start, reset,
                and scene-switch.

        Returns:
            A list of NalFrames, one per input frame. With the
            ultra-low-latency tuning + B-frames disabled, NVENC's
            input/output cadence is 1:1, so the list length equals
            ``argb_chunk.shape[0]``.
        """
        if self._closed:
            raise RuntimeError("NVENC encoder is closed")
        if argb_chunk.ndim != 4 or argb_chunk.shape[-1] != 4:
            raise ValueError(
                "expected ARGB chunk [T, H, W, 4]; got shape "
                f"{tuple(argb_chunk.shape)}"
            )
        if argb_chunk.dtype != torch.uint8:
            raise ValueError(
                f"expected uint8 ARGB; got dtype {argb_chunk.dtype}"
            )

        T = int(argb_chunk.shape[0])
        out: list[NalFrame] = []
        for f in range(T):
            surface = argb_chunk[f].contiguous()
            blob = self._encode_one(
                surface, force_idr=(force_idr and f == 0)
            )
            nalus = _split_annexb(blob)
            if not nalus:
                # PyNvVideoCodec's ``Encode`` buffers the first submit
                # internally and returns 0 bytes; the encoded output
                # appears on the next submit. Skip empty slots so the
                # caller can keep the marker queue and the NAL queue
                # 1:1 by enqueuing one fewer marker for that chunk.
                continue
            out.append(NalFrame(nal_units=nalus))
        return out

    def _encode_one(
        self, surface: torch.Tensor, *, force_idr: bool
    ) -> bytes:
        """Submit one ARGB surface to NVENC; return Annex-B bytes.

        Output may be empty if NVENC buffered the frame internally; the
        bytes will appear on a later call (PyNvVideoCodec 2.x with the
        ultra-low-latency tuning typically returns a non-empty buffer
        per submit, but we handle the empty case defensively).

        ``force_idr=True`` triggers the 2-arg ``Encode(surface, picFlags)``
        overload with ``picFlags = FORCEIDR | OUTPUT_SPSPPS`` so the IDR
        is accompanied by SPS+PPS — required for a browser that
        connects mid-stream to be able to initialise its decoder.
        """
        if force_idr:
            blob = self._enc.Encode(surface, _NVENC_PIC_FLAGS_KEYFRAME)
        else:
            blob = self._enc.Encode(surface)
        return bytes(blob) if blob else b""

    def flush(self) -> list[NalFrame]:
        """Flush any buffered output; return it as trailing NalFrames.

        Called on shutdown so trailing P-frames make it to the wire
        before the encoder is released. Multi-frame flush output is
        emitted as a single trailing pseudo-frame because we cannot
        delimit individual frames from the byte stream without parsing
        ``slice_type`` — only matters on the very last chunk before
        session close.
        """
        if self._closed:
            return []
        try:
            blob = self._enc.EndEncode()
        except Exception:  # noqa: BLE001 — EndEncode quirks vary by version
            LOGGER.exception(
                "EndEncode raised; treating as a no-op flush"
            )
            return []
        nalus = _split_annexb(bytes(blob) if blob else b"")
        if not nalus:
            return []
        return [NalFrame(nal_units=nalus)]

    def close(self) -> None:
        """Release the NVENC session. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        except Exception:
            LOGGER.exception("flush failed during close")
        try:
            del self._enc
        except Exception:
            LOGGER.exception("encoder reference cleanup failed")


__all__ = [
    "CosmoshNvencH264",
    "NalFrame",
    "NvencConfig",
]
