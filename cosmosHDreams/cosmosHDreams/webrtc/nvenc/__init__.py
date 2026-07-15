"""NVENC-accelerated video encode path for the CosmosH WebRTC server.

This subpackage adds an opt-in path that keeps the per-block pixel
chunk on the NVIDIA GPU and encodes it to H.264 via NVENC (through
PyNvVideoCodec), instead of copying to host memory and encoding with
libavcodec.

The CPU path (the existing ``cosmosHDreams.webrtc.media`` module) is preserved
as the default and as the fallback when NVENC is unavailable. Selection
is controlled by ``runtime.video.encoder`` in YAML, the ``--encoder``
CLI flag, or the ``COSMOSH_VIDEO_ENCODER`` environment variable; see
:mod:`cosmosHDreams.webrtc.nvenc.resolver` for the precedence and the auto
probe.

Public surface (consumed by the rest of CosmosH):

- :func:`resolver.resolve_encoder` — three-value selector (``auto`` /
  ``nvenc`` / ``cpu_libav``) returning the encoder that will actually
  run plus a one-line explanation.
- :func:`pack.denormalize_and_pack_argb` — GPU kernel that converts the
  runtime's bf16 [-1, 1] chunk into a uint8 ARGB surface ready for
  NVENC.

The subpackage ships the resolver, the pack kernel, the
PyNvVideoCodec encoder wrapper, the aiortc adapter, and the NVENC
video track — all the pieces needed to feed NVENC-produced H.264
bytes into aiortc's WebRTC sender in place of aiortc's stock libav
encoder.
"""
from __future__ import annotations

from cosmosHDreams.webrtc.nvenc.aiortc_compat import (
    AiortcCompatError,
    NvencH264Encoder,
    assert_aiortc_symbols_present,
    install_nvenc_encoder,
    restrict_codecs_to_h264,
    uninstall_nvenc_encoder,
)
from cosmosHDreams.webrtc.nvenc.encoder import (
    CosmoshNvencH264,
    NalFrame,
    NvencConfig,
)
from cosmosHDreams.webrtc.nvenc.pack import denormalize_and_pack_argb
from cosmosHDreams.webrtc.nvenc.resolver import (
    ENCODER_AUTO,
    ENCODER_CPU_LIBAV,
    ENCODER_NVENC,
    NvencUnavailableError,
    VALID_ENCODERS,
    probe_nvenc_available,
    resolve_encoder,
)
from cosmosHDreams.webrtc.nvenc.track import CosmoshNvencVideoTrack
from cosmosHDreams.webrtc.nvenc.video_encoder import (
    ChunkDeliveryResult,
    DefaultRTCVideoEncoder,
    PyNvHardwareEncoder,
    VideoEncoder,
)

__all__ = [
    "AiortcCompatError",
    "ChunkDeliveryResult",
    "CosmoshNvencH264",
    "CosmoshNvencVideoTrack",
    "DefaultRTCVideoEncoder",
    "ENCODER_AUTO",
    "ENCODER_CPU_LIBAV",
    "ENCODER_NVENC",
    "NalFrame",
    "NvencConfig",
    "NvencH264Encoder",
    "NvencUnavailableError",
    "PyNvHardwareEncoder",
    "VALID_ENCODERS",
    "VideoEncoder",
    "assert_aiortc_symbols_present",
    "denormalize_and_pack_argb",
    "install_nvenc_encoder",
    "probe_nvenc_available",
    "resolve_encoder",
    "restrict_codecs_to_h264",
    "uninstall_nvenc_encoder",
]
