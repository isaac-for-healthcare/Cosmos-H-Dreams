"""Unit tests for the :class:`VideoEncoder` Protocol and its two
implementations.

Pure-Python + fake-aiortc: verifies Protocol conformance,
capability-probe delegation, backend/prefers_codec labelling, and
:meth:`DefaultRTCVideoEncoder.deliver_chunk` routing to the CPU
track's ``enqueue_chunk``.

Hardware-only cases (NVENC session construction, real encode) live in
``test_nvenc_integration.py`` behind the ``ci_gpu`` mark.
"""
from __future__ import annotations

from unittest import mock

import pytest
import torch

from cosmosHDreams.webrtc.nvenc import (
    ChunkDeliveryResult,
    DefaultRTCVideoEncoder,
    PyNvHardwareEncoder,
    VideoEncoder,
)

pytestmark = pytest.mark.ci_cpu


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_default_rtc_encoder_conforms_to_protocol() -> None:
    enc = DefaultRTCVideoEncoder(fps=30)
    assert isinstance(enc, VideoEncoder)


def test_pynv_hardware_encoder_class_conforms_to_protocol_shape() -> None:
    """The class exposes every Protocol attribute at the class level.

    We cannot instantiate ``PyNvHardwareEncoder`` on a CPU-only host
    (PyNvVideoCodec / CUDA aren't available), so this test asserts
    class-level shape only: the attributes / methods a
    :class:`VideoEncoder` requires are all defined.
    """
    for attr in (
        "backend",
        "prefers_codec",
        "create_track",
        "deliver_chunk",
        "reset_session",
        "close",
        "is_supported",
    ):
        assert hasattr(PyNvHardwareEncoder, attr), (
            f"PyNvHardwareEncoder missing Protocol member {attr!r}"
        )


# ---------------------------------------------------------------------------
# Backend / prefers_codec labels
# ---------------------------------------------------------------------------


def test_default_rtc_encoder_labels() -> None:
    enc = DefaultRTCVideoEncoder(fps=10)
    assert enc.backend == "aiortc"
    assert enc.prefers_codec is None
    assert enc.fps == 10


def test_pynv_hardware_encoder_labels() -> None:
    """Backend + codec labels are class attributes; no instance needed."""
    assert PyNvHardwareEncoder.backend == "pynvvideocodec"
    assert PyNvHardwareEncoder.prefers_codec == "h264"


# ---------------------------------------------------------------------------
# DefaultRTCVideoEncoder — construction validation
# ---------------------------------------------------------------------------


def test_default_rtc_encoder_rejects_bad_fps() -> None:
    with pytest.raises(ValueError, match="fps must be > 0"):
        DefaultRTCVideoEncoder(fps=0)
    with pytest.raises(ValueError, match="fps must be > 0"):
        DefaultRTCVideoEncoder(fps=-5)


# ---------------------------------------------------------------------------
# DefaultRTCVideoEncoder.create_track — returns a CosmoshVideoTrack
# ---------------------------------------------------------------------------


def test_default_rtc_encoder_create_track_returns_cpu_track() -> None:
    from cosmosHDreams.webrtc.media import CosmoshVideoTrack
    enc = DefaultRTCVideoEncoder(fps=30)
    track = enc.create_track()
    assert isinstance(track, CosmoshVideoTrack)


# ---------------------------------------------------------------------------
# DefaultRTCVideoEncoder.deliver_chunk — routes to track.enqueue_chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_rtc_encoder_deliver_chunk_routes_to_track() -> None:
    from cosmosHDreams.webrtc.media import CosmoshVideoTrack
    enc = DefaultRTCVideoEncoder(fps=30)
    track = enc.create_track()
    assert isinstance(track, CosmoshVideoTrack)
    # 3 rgb frames on CPU. CosmoshVideoTrack.enqueue_chunk expects the
    # runtime chunk layout ``[1, 3, T, H, W]``.
    chunk = torch.zeros((1, 3, 3, 16, 16), dtype=torch.uint8)
    result = await enc.deliver_chunk(chunk, track, force_keyframe=False)
    assert isinstance(result, ChunkDeliveryResult)
    assert result.backend == "aiortc"
    assert result.num_frames == 3
    assert result.num_keyframes == 0
    assert result.encode_ms == 0.0


@pytest.mark.asyncio
async def test_default_rtc_encoder_rejects_wrong_track_type() -> None:
    enc = DefaultRTCVideoEncoder(fps=30)
    with pytest.raises(TypeError, match="requires a CosmoshVideoTrack"):
        await enc.deliver_chunk(
            torch.zeros((1, 3, 1, 4, 4)), object(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# DefaultRTCVideoEncoder — lifecycle no-ops
# ---------------------------------------------------------------------------


def test_default_rtc_encoder_reset_session_is_noop() -> None:
    enc = DefaultRTCVideoEncoder(fps=30)
    enc.reset_session()
    enc.reset_session()  # idempotent


def test_default_rtc_encoder_close_is_noop() -> None:
    enc = DefaultRTCVideoEncoder(fps=30)
    enc.close()
    enc.close()  # idempotent


# ---------------------------------------------------------------------------
# PyNvHardwareEncoder.is_supported — delegates to probe_nvenc_available
# ---------------------------------------------------------------------------


def test_is_supported_delegates_to_probe(monkeypatch) -> None:
    """``PyNvHardwareEncoder.is_supported`` must return whatever
    ``probe_nvenc_available`` returns and forward its kwargs verbatim."""
    from cosmosHDreams.webrtc.nvenc import video_encoder as ve_mod

    seen: dict = {}

    def fake_probe(**kw):
        seen.update(kw)
        return True, "test-fake OK"

    monkeypatch.setattr(ve_mod, "probe_nvenc_available", fake_probe)
    ok, reason = PyNvHardwareEncoder.is_supported(
        gpu_id=3, width=1920, height=1080,
    )
    assert ok is True
    assert reason == "test-fake OK"
    assert seen == {"gpu_id": 3, "width": 1920, "height": 1080}


def test_is_supported_defaults(monkeypatch) -> None:
    """No kwargs -> shallow probe (width=0, height=0, gpu_id=0)."""
    from cosmosHDreams.webrtc.nvenc import video_encoder as ve_mod

    seen: dict = {}

    def fake_probe(**kw):
        seen.update(kw)
        return False, "no CUDA"

    monkeypatch.setattr(ve_mod, "probe_nvenc_available", fake_probe)
    ok, reason = PyNvHardwareEncoder.is_supported()
    assert ok is False
    assert "no CUDA" in reason
    assert seen == {"gpu_id": 0, "width": 0, "height": 0}


# ---------------------------------------------------------------------------
# ChunkDeliveryResult — dataclass shape
# ---------------------------------------------------------------------------


def test_chunk_delivery_result_is_frozen() -> None:
    r = ChunkDeliveryResult(
        backend="aiortc", num_frames=5, num_keyframes=0, encode_ms=0.0,
    )
    with pytest.raises(Exception):
        r.num_frames = 10  # type: ignore[misc]


def test_chunk_delivery_result_carries_all_fields() -> None:
    r = ChunkDeliveryResult(
        backend="pynvvideocodec",
        num_frames=12,
        num_keyframes=1,
        encode_ms=3.5,
    )
    assert r.backend == "pynvvideocodec"
    assert r.num_frames == 12
    assert r.num_keyframes == 1
    assert r.encode_ms == 3.5
