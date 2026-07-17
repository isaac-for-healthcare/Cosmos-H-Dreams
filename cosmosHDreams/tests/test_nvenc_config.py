# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the encoder / NVENC YAML fields in ``config_loader``.

Pure-Python — does not import torch, PyNvVideoCodec, aiortc internals,
or any model code.
"""
from __future__ import annotations

import pytest

from cosmosHDreams.webrtc.config_loader import (
    ConfigError,
    _NVENC_DEFAULTS,
    _validate_nvenc_block,
    get_encoder_settings,
)
from cosmosHDreams.webrtc.nvenc.resolver import (
    ENCODER_AUTO,
    ENCODER_CPU_LIBAV,
    ENCODER_NVENC,
)

pytestmark = pytest.mark.ci_cpu


# ---------------------------------------------------------------------------
# get_encoder_settings — defaults and field validation
# ---------------------------------------------------------------------------


def test_get_encoder_defaults_to_cpu_libav_when_video_absent():
    out = get_encoder_settings({})
    assert out["encoder"] == ENCODER_CPU_LIBAV
    assert out["nvenc"] == _NVENC_DEFAULTS


def test_get_encoder_defaults_to_cpu_libav_when_encoder_field_absent():
    out = get_encoder_settings({"video": {"jpeg_quality": 85}})
    assert out["encoder"] == ENCODER_CPU_LIBAV


def test_get_encoder_explicit_nvenc():
    out = get_encoder_settings({"video": {"encoder": "nvenc"}})
    assert out["encoder"] == ENCODER_NVENC


def test_get_encoder_explicit_auto():
    out = get_encoder_settings({"video": {"encoder": "AUTO"}})  # case-insensitive
    assert out["encoder"] == ENCODER_AUTO


def test_get_encoder_explicit_cpu_libav():
    out = get_encoder_settings({"video": {"encoder": "cpu_libav"}})
    assert out["encoder"] == ENCODER_CPU_LIBAV


def test_get_encoder_rejects_unknown_value():
    with pytest.raises(ConfigError, match="video.encoder must be one of"):
        get_encoder_settings({"video": {"encoder": "vp8_hw"}})


def test_get_encoder_rejects_non_dict_video():
    with pytest.raises(ConfigError, match="video section must be a mapping"):
        get_encoder_settings({"video": "not a dict"})


# ---------------------------------------------------------------------------
# _validate_nvenc_block — defaults, overrides, validation
# ---------------------------------------------------------------------------


def test_nvenc_block_defaults_when_empty():
    out = _validate_nvenc_block({})
    assert out == _NVENC_DEFAULTS
    # Returned dict is independent of the module-level defaults.
    assert out is not _NVENC_DEFAULTS


def test_nvenc_block_overrides_preset_and_tuning():
    out = _validate_nvenc_block({"preset": "P1", "tuning": "low_latency_hq"})
    assert out["preset"] == "P1"
    assert out["tuning"] == "low_latency_hq"
    # Untouched fields still default.
    assert out["bitrate"] == _NVENC_DEFAULTS["bitrate"]


def test_nvenc_block_overrides_bitrate():
    out = _validate_nvenc_block({"bitrate": 5_000_000})
    assert out["bitrate"] == 5_000_000


def test_nvenc_block_overrides_idr_period():
    out = _validate_nvenc_block({"idr_period_s": 2.0})
    assert out["idr_period_s"] == 2.0


def test_nvenc_block_rejects_non_positive_bitrate():
    with pytest.raises(ConfigError, match="bitrate must be > 0"):
        _validate_nvenc_block({"bitrate": 0})
    with pytest.raises(ConfigError, match="bitrate must be > 0"):
        _validate_nvenc_block({"bitrate": -1000})


def test_nvenc_block_rejects_non_positive_idr_period():
    with pytest.raises(ConfigError, match="idr_period_s must be > 0"):
        _validate_nvenc_block({"idr_period_s": 0})


def test_nvenc_block_rejects_non_numeric_bitrate():
    with pytest.raises(ConfigError, match="bitrate must be an integer"):
        _validate_nvenc_block({"bitrate": "fast"})


def test_nvenc_block_rejects_non_dict():
    with pytest.raises(ConfigError, match="video.nvenc must be a mapping"):
        _validate_nvenc_block(["P1"])  # type: ignore[arg-type]


def test_nvenc_block_warns_on_unknown_key(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="cosmosHDreams.webrtc.config_loader"):
        out = _validate_nvenc_block({"qpmin": 18})  # unknown
    assert out == _NVENC_DEFAULTS
    assert any("Unknown video.nvenc key 'qpmin'" in rec.message
               for rec in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end: full video section with all the new fields
# ---------------------------------------------------------------------------


def test_get_encoder_full_video_section():
    cfg = {
        "video": {
            "jpeg_quality": 90,
            "encoder": "nvenc",
            "nvenc": {
                "preset": "P3",
                "tuning": "ultra_low_latency",
                "bitrate": 4_000_000,
                "idr_period_s": 2.0,
            },
        }
    }
    out = get_encoder_settings(cfg)
    assert out["encoder"] == ENCODER_NVENC
    assert out["nvenc"] == {
        "preset": "P3",
        "tuning": "ultra_low_latency",
        "bitrate": 4_000_000,
        "idr_period_s": 2.0,
    }
