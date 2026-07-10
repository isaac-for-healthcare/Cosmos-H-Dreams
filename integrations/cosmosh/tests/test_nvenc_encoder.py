"""Unit tests for ``cosmosh.webrtc.nvenc.encoder``.

Covers the pure-Python helpers (``_split_annexb``, ``NvencConfig``
validation, ``NalFrame`` value semantics). Runs without PyNvVideoCodec
or any GPU.
"""
from __future__ import annotations

import pytest

from cosmosh.webrtc.nvenc.encoder import (
    NalFrame,
    NvencConfig,
    _split_annexb,
)

pytestmark = pytest.mark.ci_cpu


# ---------------------------------------------------------------------------
# NvencConfig validation
# ---------------------------------------------------------------------------


def test_nvenc_config_defaults():
    cfg = NvencConfig(width=288, height=512, fps=30)
    assert cfg.bitrate == 3_000_000
    assert cfg.preset == "P3"
    assert cfg.tuning == "ultra_low_latency"
    assert cfg.idr_period_s == 4.0
    # IDR period in frames = round(fps * idr_period_s)
    assert cfg.idr_period_frames == 120


def test_nvenc_config_idr_period_frames_minimum():
    """A sub-1-frame IDR period rounds up to at least 1."""
    cfg = NvencConfig(width=288, height=512, fps=30, idr_period_s=0.001)
    assert cfg.idr_period_frames == 1


@pytest.mark.parametrize("w,h", [(0, 100), (100, 0), (-2, 100), (288, -1)])
def test_nvenc_config_rejects_non_positive_resolution(w, h):
    with pytest.raises(ValueError, match="invalid resolution"):
        NvencConfig(width=w, height=h, fps=30)


@pytest.mark.parametrize("w,h", [(287, 512), (288, 511), (1, 1)])
def test_nvenc_config_rejects_odd_resolution(w, h):
    with pytest.raises(ValueError, match="must be even"):
        NvencConfig(width=w, height=h, fps=30)


def test_nvenc_config_rejects_non_positive_fps():
    with pytest.raises(ValueError, match="invalid fps"):
        NvencConfig(width=288, height=512, fps=0)


def test_nvenc_config_rejects_non_positive_bitrate():
    with pytest.raises(ValueError, match="invalid bitrate"):
        NvencConfig(width=288, height=512, fps=30, bitrate=0)


def test_nvenc_config_rejects_non_positive_idr_period():
    with pytest.raises(ValueError, match="invalid idr_period_s"):
        NvencConfig(width=288, height=512, fps=30, idr_period_s=0)


# ---------------------------------------------------------------------------
# _split_annexb — Annex-B start code parsing
# ---------------------------------------------------------------------------


def test_split_annexb_empty():
    assert _split_annexb(b"") == []


def test_split_annexb_single_3byte_start_code():
    # 0x000001 then NAL payload "ABC"
    blob = b"\x00\x00\x01ABC"
    assert _split_annexb(blob) == [b"ABC"]


def test_split_annexb_single_4byte_start_code():
    blob = b"\x00\x00\x00\x01ABC"
    assert _split_annexb(blob) == [b"ABC"]


def test_split_annexb_multiple_nals_mixed_start_codes():
    blob = (
        b"\x00\x00\x00\x01AAA"   # 4-byte SC + "AAA"
        b"\x00\x00\x01BB"        # 3-byte SC + "BB"
        b"\x00\x00\x00\x01CCCC"  # 4-byte SC + "CCCC"
    )
    assert _split_annexb(blob) == [b"AAA", b"BB", b"CCCC"]


def test_split_annexb_no_start_code_leak_between_nals():
    # When a 3-byte SC is followed by a 4-byte SC, the trailing 00 and
    # the 01 of the 4-byte SC must not leak into the preceding NAL.
    blob = b"\x00\x00\x01AAA\x00\x00\x00\x01BB"
    assert _split_annexb(blob) == [b"AAA", b"BB"]


def test_split_annexb_no_start_code_returns_empty():
    """Input without any start code yields no NALs."""
    assert _split_annexb(b"ABCDEFG") == []


# ---------------------------------------------------------------------------
# NalFrame value semantics
# ---------------------------------------------------------------------------


def test_nal_frame_is_immutable():
    """``NalFrame`` is a frozen dataclass — attempts to mutate raise."""
    nf = NalFrame(nal_units=[b"\x65"])
    with pytest.raises((AttributeError, Exception)):
        nf.nal_units = [b"\x42"]  # type: ignore[misc]
