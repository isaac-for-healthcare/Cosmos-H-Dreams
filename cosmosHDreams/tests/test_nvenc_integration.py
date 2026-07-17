# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware integration tests for the NVENC encode path.

These exercise ``CosmoshNvencH264`` against a live PyNvVideoCodec +
CUDA device — the layer the CPU-safe ``test_nvenc_*`` unit tests
cannot cover. Gated on :data:`pytestmark` = ``ci_gpu`` **and** a
runtime probe that skips gracefully when the encoder isn't usable
on the host, so CPU-only dev boxes running the full suite still
see this file as ``skipped`` rather than ``failed``.

Scope of coverage that would otherwise be missed:

  - PyNvVideoCodec's ``CreateEncoder`` accepts our kwargs.
  - ``encode_chunk`` produces H.264 NAL units with the expected
    header bytes and an SPS on the first IDR (proves NVENC honours
    ``FORCEIDR + OUTPUT_SPSPPS``).
  - Steady-state encode of multiple chunks doesn't leak frames or
    drop wildly beneath the input count.
  - ``flush()`` and ``close()`` don't hang or leak sessions.
  - ``reset_session()`` gives a clean encoder afterwards.
  - The runtime → NAL queue → aiortc shim pipeline delivers real
    NVENC bytes end-to-end when the aiortc adapter is installed.

Assertions are deliberately loose on counts (NVENC pipeline depth
varies by SDK / driver — we hit exactly this during Quest mode
bring-up) but strict on structural properties (NAL type bits, SPS
presence, non-empty bitstream). Bit-exact output comparisons are
avoided — they would be brittle across driver upgrades.
"""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Hardware probe — same predicate the resolver uses, so this file skips
# exactly when the resolver would refuse to pick nvenc.
# ---------------------------------------------------------------------------


def _hardware_available() -> tuple[bool, str]:
    try:
        import PyNvVideoCodec  # noqa: F401
    except ImportError as exc:
        return False, f"PyNvVideoCodec not importable ({exc})"
    try:
        import torch
    except ImportError as exc:
        return False, f"torch not importable ({exc})"
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        if torch.cuda.device_count() < 1:
            return False, "no CUDA device visible"
    except Exception as exc:  # noqa: BLE001
        return False, f"CUDA probe failed ({exc})"
    return True, ""


_HW_OK, _HW_REASON = _hardware_available()

# ``ci_gpu`` selects this file under the project's marker policy;
# the ``skipif`` guarantees graceful skipping when hardware is
# actually missing, regardless of runner labels (so a full
# ``pytest`` run on a CPU-only dev box shows these as ``skipped``
# rather than erroring).
pytestmark = [
    pytest.mark.ci_gpu,
    pytest.mark.skipif(not _HW_OK, reason=f"NVENC unavailable: {_HW_REASON}"),
]


# ---------------------------------------------------------------------------
# Test-side helpers
# ---------------------------------------------------------------------------


_T_DEFAULT = 12
_H_DEFAULT = 288
_W_DEFAULT = 512
_FPS = 30


def _make_argb_chunk(T: int = _T_DEFAULT, H: int = _H_DEFAULT, W: int = _W_DEFAULT):
    """A ``[T, H, W, 4]`` uint8 BGRA-in-memory tensor on CUDA.

    Uses the production ``denormalize_and_pack_argb`` kernel so the
    input path matches what the runtime feeds NVENC. Content is a
    deterministic random signal — enough entropy to force P-frame
    coding without being adversarial.
    """
    import torch
    from cosmosHDreams.webrtc.nvenc.pack import denormalize_and_pack_argb

    g = torch.Generator(device="cpu").manual_seed(0xC05DE)
    raw = (torch.rand((1, 3, T, H, W), generator=g) * 2.0 - 1.0).to(torch.bfloat16)
    return denormalize_and_pack_argb(raw.to("cuda"))


def _nal_type(unit: bytes) -> int:
    """H.264 NAL unit type (low 5 bits of the first byte)."""
    return unit[0] & 0x1F if unit else -1


def _has_sps(nal_frames) -> bool:
    return any(_nal_type(u) == 7 for nf in nal_frames for u in nf.nal_units)


def _stub_frame(pts: int, fps: int = _FPS) -> SimpleNamespace:
    """An ``av.VideoFrame``-shaped stub the shim's ``convert_timebase`` accepts."""
    return SimpleNamespace(pts=pts, time_base=Fraction(1, fps))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nvenc_encoder() -> Iterator:
    """A fresh ``CosmoshNvencH264`` closed after the test."""
    from cosmosHDreams.webrtc.nvenc.encoder import CosmoshNvencH264, NvencConfig

    enc = CosmoshNvencH264(
        NvencConfig(width=_W_DEFAULT, height=_H_DEFAULT, fps=_FPS)
    )
    try:
        yield enc
    finally:
        if not enc.closed:
            enc.close()


@pytest.fixture
def isolated_adapter_state() -> Iterator[None]:
    """Snapshot / restore aiortc state around adapter-integration tests.

    Mirrors the fixture in ``test_nvenc_aiortc_compat.py`` — a leaked
    install would swap ``H264Encoder`` for every later test.
    """
    import aiortc.codecs as _aiortc_codecs  # type: ignore[import-untyped]
    import aiortc.codecs.h264 as _aiortc_h264  # type: ignore[import-untyped]
    from cosmosHDreams.webrtc.nvenc.aiortc_compat import uninstall_nvenc_encoder

    if hasattr(_aiortc_codecs, "init_codecs"):
        if not _aiortc_codecs.CODECS.get("video"):
            _aiortc_codecs.init_codecs()
    saved_video = list(_aiortc_codecs.CODECS.get("video", []))
    saved_cls = _aiortc_h264.H264Encoder

    yield

    uninstall_nvenc_encoder()
    _aiortc_h264.H264Encoder = saved_cls
    _aiortc_codecs.CODECS["video"] = saved_video


# ---------------------------------------------------------------------------
# Construction / lifecycle
# ---------------------------------------------------------------------------


def test_encoder_constructs_on_real_hardware(nvenc_encoder):
    """A live NVENC session opens with our production kwargs.

    Regression guard against PyNvVideoCodec kwarg renames — the
    kwargs dict is our only interface to ``CreateEncoder``, and the
    library has renamed keys between minor versions in the past.
    """
    assert not nvenc_encoder.closed
    assert nvenc_encoder.config.width == _W_DEFAULT
    assert nvenc_encoder.config.height == _H_DEFAULT
    assert nvenc_encoder.config.fps == _FPS


def test_close_is_idempotent(nvenc_encoder):
    """A double close does not raise (real NVENC session lifecycle)."""
    nvenc_encoder.close()
    assert nvenc_encoder.closed
    nvenc_encoder.close()  # must not raise


# ---------------------------------------------------------------------------
# encode_chunk — the hot path
# ---------------------------------------------------------------------------


def test_encode_chunk_produces_valid_h264_with_sps_on_idr(nvenc_encoder):
    """A single 12-frame chunk with force_idr=True yields valid H.264.

    Assertions are structural, not bit-exact:
      * At least one NalFrame comes back (encoders with deeper
        pipeline may buffer some — we ``flush()`` to be sure).
      * At least one emitted NAL is an SPS (type 7), proving
        ``OUTPUT_SPSPPS`` was honoured.
      * Every emitted NAL has a valid H.264 NAL header type
        (1..12 — slice / IDR / SPS / PPS / SEI / AUD / etc.).
    """
    argb = _make_argb_chunk(T=_T_DEFAULT)
    nal_frames = list(nvenc_encoder.encode_chunk(argb, force_idr=True))
    nal_frames.extend(nvenc_encoder.flush())

    assert nal_frames, "NVENC returned no NAL frames for a full chunk"
    assert _has_sps(nal_frames), (
        "expected an SPS NAL (type 7) somewhere in the bitstream "
        "for a force_idr=True chunk; NVENC's OUTPUT_SPSPPS flag "
        "may not be firing"
    )
    for nf in nal_frames:
        assert nf.nal_units, "NalFrame carries no NAL units"
        for u in nf.nal_units:
            t = _nal_type(u)
            assert 1 <= t <= 12, (
                f"NAL type {t} out of H.264 range 1..12 "
                f"(first byte {u[0]:#04x})"
            )


def test_encode_chunk_steady_state_matches_input_count(nvenc_encoder):
    """Three consecutive chunks encode without loss beyond pipeline depth.

    NVENC's pipeline may absorb a few frames on cold start; over
    multiple chunks the input/output cadence should be effectively
    1:1 in steady state. We use a loose lower bound (75 % of
    submitted frames) so a small driver-side buffer doesn't flake
    the test, while still catching order-of-magnitude regressions.
    """
    submitted = 0
    emitted = 0
    for chunk_idx in range(3):
        argb = _make_argb_chunk(T=_T_DEFAULT)
        submitted += _T_DEFAULT
        nal_frames = nvenc_encoder.encode_chunk(
            argb, force_idr=(chunk_idx == 0)
        )
        emitted += len(nal_frames)

    # Include flush output — the final buffered frames should surface.
    emitted += len(nvenc_encoder.flush())

    assert emitted >= int(0.75 * submitted), (
        f"NVENC emitted {emitted} NAL frames for {submitted} submitted; "
        f"pipeline depth or drop rate is far above expectations"
    )


def test_encode_chunk_rejects_bad_shape(nvenc_encoder):
    """Shape guardrails still fire on live hardware (not just mocks)."""
    import torch

    bad = torch.zeros((5, 128, 128), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="expected ARGB chunk"):
        nvenc_encoder.encode_chunk(bad, force_idr=False)


def test_encode_chunk_rejects_bad_dtype(nvenc_encoder):
    """Non-uint8 input is rejected before hitting NVENC."""
    import torch

    bad = torch.zeros((_T_DEFAULT, _H_DEFAULT, _W_DEFAULT, 4),
                      dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="expected uint8"):
        nvenc_encoder.encode_chunk(bad, force_idr=False)


# ---------------------------------------------------------------------------
# reset_session — the tear-down + recreate path
# ---------------------------------------------------------------------------


def test_reset_session_leaves_encoder_usable(nvenc_encoder):
    """After ``reset_session``, another encode round works normally."""
    argb = _make_argb_chunk(T=_T_DEFAULT)
    nvenc_encoder.encode_chunk(argb, force_idr=True)

    nvenc_encoder.reset_session()  # must not raise; must not leak

    argb2 = _make_argb_chunk(T=_T_DEFAULT)
    nal_frames = list(nvenc_encoder.encode_chunk(argb2, force_idr=True))
    nal_frames.extend(nvenc_encoder.flush())
    assert nal_frames, "encoder produced nothing after reset_session"
    assert _has_sps(nal_frames), (
        "post-reset force_idr chunk did not surface an SPS — the "
        "fresh session's OUTPUT_SPSPPS may not be firing"
    )


def test_reset_session_on_closed_encoder_raises(nvenc_encoder):
    """Explicit guard: reset after close should fail loudly, not corrupt state."""
    nvenc_encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        nvenc_encoder.reset_session()


# ---------------------------------------------------------------------------
# End-to-end: runtime → NAL queue → aiortc shim
# ---------------------------------------------------------------------------


def test_shim_delivers_real_nvenc_bytes(nvenc_encoder, isolated_adapter_state):
    """Full path: real NVENC bytes → NAL queue → aiortc shim.encode()."""
    import queue as _queue

    from cosmosHDreams.webrtc.nvenc.aiortc_compat import (
        NvencH264Encoder,
        install_nvenc_encoder,
    )

    nq: "_queue.Queue" = _queue.Queue()
    install_nvenc_encoder(nq)

    # Encode a chunk into the queue via the real hardware path.
    argb = _make_argb_chunk(T=_T_DEFAULT)
    nal_frames = list(nvenc_encoder.encode_chunk(argb, force_idr=True))
    nal_frames.extend(nvenc_encoder.flush())
    assert nal_frames, "hardware encode returned no NAL frames"
    for nf in nal_frames:
        nq.put(nf)

    # Drain each queued NalFrame through the shim exactly as aiortc's
    # sender would (one .encode() call per placeholder frame).
    import aiortc.codecs.h264 as _aiortc_h264  # type: ignore[import-untyped]

    shim_cls = _aiortc_h264.H264Encoder  # already swapped in by install
    assert shim_cls is NvencH264Encoder, (
        "install_nvenc_encoder did not swap the H264Encoder class "
        "in aiortc.codecs.h264 — dispatcher would bypass the shim"
    )
    enc = shim_cls()

    total_payloads = 0
    for i in range(len(nal_frames)):
        payloads, ts = enc.encode(frame=_stub_frame(pts=i), force_keyframe=False)
        assert isinstance(payloads, list) and payloads, (
            f"shim returned no payloads for NalFrame index {i}"
        )
        assert isinstance(ts, int) and ts >= 0
        total_payloads += len(payloads)

    assert total_payloads >= len(nal_frames), (
        "shim produced fewer payload lists than input NalFrames — "
        "packetiser dropped frames"
    )
    assert nq.empty(), "NAL queue not drained after all shim.encode() calls"
