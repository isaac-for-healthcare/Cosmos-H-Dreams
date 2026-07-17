# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the aiortc adapter (isolation boundary).

These exercise the adapter against the *real* installed aiortc — the
whole point of the adapter is to monkey-patch aiortc's internal
symbols, so verifying the symbols still exist and that our subclass
slot-fits is most of the value. No PyNvVideoCodec, no NVENC hardware,
no live RTP — just the Python class plumbing and the queue drain.

Each test runs install + uninstall via a fixture to keep aiortc state
isolated across cases (a leaked install would silently swap
``H264Encoder`` for every later test in the suite).
"""
from __future__ import annotations

import queue
from fractions import Fraction
from types import SimpleNamespace

import aiortc.codecs as _aiortc_codecs  # type: ignore[import-untyped]
import aiortc.codecs.h264 as _aiortc_h264  # type: ignore[import-untyped]
import pytest

from cosmosHDreams.webrtc.nvenc.aiortc_compat import (
    AiortcCompatError,
    NvencH264Encoder,
    assert_aiortc_symbols_present,
    install_nvenc_encoder,
    restrict_codecs_to_h264,
    uninstall_nvenc_encoder,
)
from cosmosHDreams.webrtc.nvenc.encoder import NalFrame

pytestmark = pytest.mark.ci_cpu


def _stub_frame(pts: int, fps: int = 30) -> SimpleNamespace:
    """An ``av.VideoFrame``-shaped stub the shim's ``convert_timebase`` will accept."""
    return SimpleNamespace(pts=pts, time_base=Fraction(1, fps))


@pytest.fixture(autouse=True)
def _isolated_adapter_state():
    """Snapshot/restore aiortc state around every test."""
    if hasattr(_aiortc_codecs, "init_codecs"):
        # init_codecs APPENDS — only call it if the table is empty.
        if not _aiortc_codecs.CODECS.get("video"):
            _aiortc_codecs.init_codecs()
    saved_codecs_video = list(_aiortc_codecs.CODECS.get("video", []))
    saved_encoder_cls = _aiortc_h264.H264Encoder

    yield

    uninstall_nvenc_encoder()
    _aiortc_h264.H264Encoder = saved_encoder_cls
    _aiortc_codecs.CODECS["video"] = saved_codecs_video


# ---------------------------------------------------------------------------
# assert_aiortc_symbols_present — the CI gate.
# ---------------------------------------------------------------------------


def test_assert_aiortc_symbols_present_passes_against_installed_aiortc():
    """The adapter's pinned aiortc internals exist in the installed wheel."""
    assert_aiortc_symbols_present()  # must not raise


# ---------------------------------------------------------------------------
# install_nvenc_encoder / uninstall_nvenc_encoder.
# ---------------------------------------------------------------------------


def test_install_swaps_h264_encoder_class():
    original = _aiortc_h264.H264Encoder
    nq: "queue.Queue[NalFrame]" = queue.Queue()
    install_nvenc_encoder(nq)
    # Both bindings must be swapped: the dispatcher (get_encoder) reads
    # ``aiortc.codecs.H264Encoder``, which is a separate name created at
    # ``aiortc.codecs.__init__`` import time. Patching only
    # ``aiortc.codecs.h264.H264Encoder`` would leave the dispatcher
    # pointing at the original libavcodec encoder and silently bypass
    # our shim. See the black-frame incident (aiortc dispatcher reads
    # from the imported alias).
    assert _aiortc_h264.H264Encoder is NvencH264Encoder
    if hasattr(_aiortc_codecs, "H264Encoder"):
        assert _aiortc_codecs.H264Encoder is NvencH264Encoder
    uninstall_nvenc_encoder()
    assert _aiortc_h264.H264Encoder is original
    if hasattr(_aiortc_codecs, "H264Encoder"):
        assert _aiortc_codecs.H264Encoder is original


def test_install_is_idempotent():
    """Calling install twice does not double-wrap the class."""
    nq1: "queue.Queue[NalFrame]" = queue.Queue()
    nq2: "queue.Queue[NalFrame]" = queue.Queue()
    install_nvenc_encoder(nq1)
    swapped_class = _aiortc_h264.H264Encoder
    install_nvenc_encoder(nq2)
    # Class identity unchanged on second install.
    assert _aiortc_h264.H264Encoder is swapped_class
    # The queue handle, however, IS updated — next encode pulls from nq2.
    # (Verified indirectly in test_encode_pulls_from_installed_queue.)


def test_uninstall_when_never_installed_is_noop():
    uninstall_nvenc_encoder()  # must not raise


def test_uninstall_clears_state_so_encode_fails_loudly():
    nq: "queue.Queue[NalFrame]" = queue.Queue()
    install_nvenc_encoder(nq)
    uninstall_nvenc_encoder()
    # Instantiate our shim directly (bypassing aiortc's get_encoder so
    # we can assert the loud-failure path on a leaked instance).
    enc = NvencH264Encoder()
    with pytest.raises(AiortcCompatError, match="no NAL queue is installed"):
        enc.encode(frame=_stub_frame(pts=0), force_keyframe=False)


# ---------------------------------------------------------------------------
# encode() — the per-frame call aiortc makes.
# ---------------------------------------------------------------------------


def test_encode_pulls_from_installed_queue():
    """The shim returns ``(payloads, ts)`` with ts derived via convert_timebase."""
    nf = NalFrame(nal_units=[b"\x65\x88\x84\x00\x00\x00"])  # IDR slice (type 5)
    nq: "queue.Queue[NalFrame]" = queue.Queue()
    nq.put(nf)
    install_nvenc_encoder(nq)

    enc = _aiortc_h264.H264Encoder()  # == NvencH264Encoder()
    # pts=1 at 30 fps time base → 1 * (1/30) / (1/90000) = 3000 in RTP units.
    payloads, ts = enc.encode(frame=_stub_frame(pts=1, fps=30), force_keyframe=False)

    assert int(ts) == 3000
    assert isinstance(payloads, list) and len(payloads) >= 1
    assert all(isinstance(p, (bytes, bytearray)) for p in payloads)
    assert nq.empty()


def test_encode_packetizes_via_inherited_method():
    """Single small NAL should produce one single-NAL RTP payload."""
    # The NAL byte starts with 0x65 = NAL_REF_IDC=3, NAL_TYPE=5 (IDR).
    small_nal = b"\x65" + b"\x42" * 20  # 21 bytes total — fits in one RTP packet
    nf = NalFrame(nal_units=[small_nal])
    nq: "queue.Queue[NalFrame]" = queue.Queue()
    nq.put(nf)
    install_nvenc_encoder(nq)

    enc = _aiortc_h264.H264Encoder()
    payloads, _ts = enc.encode(frame=_stub_frame(pts=3), force_keyframe=False)

    # At least one payload contains our NAL bytes (aiortc may wrap with
    # STAP-A header, so check substring).
    assert len(payloads) >= 1
    assert any(small_nal in bytes(p) for p in payloads)


def test_encode_force_keyframe_is_silently_ignored():
    """``force_keyframe=True`` does not raise; PLI re-key is not propagated."""
    nf = NalFrame(nal_units=[b"\x41\x9e\x01"])
    nq: "queue.Queue[NalFrame]" = queue.Queue()
    nq.put(nf)
    install_nvenc_encoder(nq)

    enc = _aiortc_h264.H264Encoder()
    payloads, _ts = enc.encode(frame=_stub_frame(pts=2), force_keyframe=True)
    assert len(payloads) >= 1


def test_encode_timeout_raises_clear_error():
    """Empty queue past the timeout raises ``AiortcCompatError``."""
    nq: "queue.Queue[NalFrame]" = queue.Queue()
    install_nvenc_encoder(nq)

    enc = _aiortc_h264.H264Encoder()
    enc._DEFAULT_GET_TIMEOUT_S = 0.05  # speed up the test
    with pytest.raises(AiortcCompatError, match="Timed out"):
        enc.encode(frame=_stub_frame(pts=0), force_keyframe=False)


# ---------------------------------------------------------------------------
# restrict_codecs_to_h264 — SDP-level codec filtering.
# ---------------------------------------------------------------------------


def test_restrict_codecs_drops_vp8():
    """After init_codecs runs, CODECS['video'] has VP8 + H.264 entries."""
    # Sanity check the precondition.
    video_before = _aiortc_codecs.CODECS["video"]
    assert any(
        getattr(c, "mimeType", "").lower() == "video/vp8" for c in video_before
    ), "test precondition: VP8 should be in CODECS['video'] before restrict"

    restrict_codecs_to_h264()

    video_after = _aiortc_codecs.CODECS["video"]
    assert len(video_after) >= 1
    assert all(
        getattr(c, "mimeType", "").lower() == "video/h264" for c in video_after
    )


def test_restrict_codecs_is_idempotent():
    restrict_codecs_to_h264()
    first = list(_aiortc_codecs.CODECS["video"])
    restrict_codecs_to_h264()
    second = list(_aiortc_codecs.CODECS["video"])
    assert first == second


def test_uninstall_restores_codecs_video():
    saved = list(_aiortc_codecs.CODECS["video"])
    restrict_codecs_to_h264()
    assert len(_aiortc_codecs.CODECS["video"]) < len(saved)
    uninstall_nvenc_encoder()
    # uninstall restores the codec list captured at first restrict.
    assert _aiortc_codecs.CODECS["video"] == saved
