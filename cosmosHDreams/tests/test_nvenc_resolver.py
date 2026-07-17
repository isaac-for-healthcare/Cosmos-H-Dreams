# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``cosmosHDreams.webrtc.nvenc.resolver``.

Pure-Python — no CUDA, no PyNvVideoCodec, no aiortc internals. Runs on
any host with the project's standard test extras.

The probe itself (:func:`probe_nvenc_available`) is tested only for
its contract (returns a ``(bool, str)`` tuple); we cannot meaningfully
test its return *value* on a CPU-only dev host. End-to-end NVENC
verification lives in :mod:`tests.test_nvenc_integration`, which is
marked ``ci_gpu`` and skipped where the probe would say the encoder
is unusable.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

from cosmosHDreams.webrtc.nvenc import resolver
from cosmosHDreams.webrtc.nvenc.resolver import (
    ENCODER_AUTO,
    ENCODER_CPU_LIBAV,
    ENCODER_NVENC,
    NvencUnavailableError,
    VALID_ENCODERS,
    normalize_encoder_value,
    probe_nvenc_available,
    read_encoder_env,
    resolve_encoder,
    select_requested_encoder,
)

pytestmark = pytest.mark.ci_cpu


# ---------------------------------------------------------------------------
# normalize_encoder_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("nvenc", "nvenc"),
        ("NVENC", "nvenc"),
        ("  Auto  ", "auto"),
        ("cpu_libav", "cpu_libav"),
    ],
)
def test_normalize_encoder_value(raw, expected):
    assert normalize_encoder_value(raw) == expected


# ---------------------------------------------------------------------------
# select_requested_encoder — precedence
# ---------------------------------------------------------------------------


def test_select_default_when_no_layer_specified():
    assert select_requested_encoder() == ENCODER_CPU_LIBAV


def test_select_yaml_overrides_code_default():
    assert (
        select_requested_encoder(yaml_value="auto")
        == ENCODER_AUTO
    )


def test_select_env_overrides_yaml():
    assert (
        select_requested_encoder(yaml_value="cpu_libav", env_value="auto")
        == ENCODER_AUTO
    )


def test_select_cli_overrides_env():
    assert (
        select_requested_encoder(
            cli_value="nvenc",
            env_value="auto",
            yaml_value="cpu_libav",
        )
        == ENCODER_NVENC
    )


def test_select_empty_layers_fall_through():
    # An empty CLI value should not block lower-priority layers.
    assert (
        select_requested_encoder(
            cli_value="",
            env_value=None,
            yaml_value="nvenc",
        )
        == ENCODER_NVENC
    )


def test_select_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown video encoder"):
        select_requested_encoder(cli_value="vp8_hw")


# ---------------------------------------------------------------------------
# probe_nvenc_available — contract
# ---------------------------------------------------------------------------


def test_probe_returns_bool_and_string():
    ok, reason = probe_nvenc_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    assert reason  # non-empty


def test_probe_handles_missing_pynvvideocodec(monkeypatch):
    """Force the PyNvVideoCodec import to fail and check the probe degrades."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PyNvVideoCodec":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, reason = probe_nvenc_available()
    assert ok is False
    assert "PyNvVideoCodec" in reason


# ---------------------------------------------------------------------------
# resolve_encoder — explicit values
# ---------------------------------------------------------------------------


def test_resolve_explicit_cpu():
    active, reason = resolve_encoder(ENCODER_CPU_LIBAV)
    assert active == ENCODER_CPU_LIBAV
    assert "cpu_libav" in reason


def test_resolve_rejects_unknown():
    with pytest.raises(ValueError, match="unknown video encoder"):
        resolve_encoder("vp8_hw")


def test_resolve_nvenc_fails_loud_when_probe_negative(monkeypatch):
    """nvenc requested explicitly + probe negative -> hard error."""
    monkeypatch.setattr(
        resolver, "probe_nvenc_available",
        lambda **_kw: (False, "simulated: no NVENC"),
    )
    with pytest.raises(NvencUnavailableError, match="simulated: no NVENC"):
        resolve_encoder(ENCODER_NVENC)


def test_resolve_nvenc_succeeds_when_probe_positive(monkeypatch):
    monkeypatch.setattr(
        resolver, "probe_nvenc_available",
        lambda **_kw: (True, "simulated: NVENC available"),
    )
    active, reason = resolve_encoder(ENCODER_NVENC)
    assert active == ENCODER_NVENC
    assert "nvenc" in reason


# ---------------------------------------------------------------------------
# resolve_encoder — auto path
# ---------------------------------------------------------------------------


def test_resolve_auto_falls_back_when_probe_negative(monkeypatch):
    monkeypatch.setattr(
        resolver, "probe_nvenc_available",
        lambda **_kw: (False, "simulated: no NVENC"),
    )
    active, reason = resolve_encoder(ENCODER_AUTO)
    assert active == ENCODER_CPU_LIBAV
    assert "auto -> cpu_libav" in reason
    assert "simulated: no NVENC" in reason


def test_resolve_auto_picks_nvenc_when_probe_positive(monkeypatch):
    monkeypatch.setattr(
        resolver, "probe_nvenc_available",
        lambda **_kw: (True, "simulated: NVENC available"),
    )
    active, reason = resolve_encoder(ENCODER_AUTO)
    assert active == ENCODER_NVENC
    assert "auto -> nvenc" in reason


def test_resolve_forwards_resolution_to_probe(monkeypatch):
    """When resolution kwargs are provided they reach ``probe_nvenc_available``."""
    seen: dict = {}

    def spy_probe(**kw):
        seen.update(kw)
        return True, "spy"

    monkeypatch.setattr(resolver, "probe_nvenc_available", spy_probe)
    resolve_encoder(ENCODER_AUTO, gpu_id=7, width=1920, height=1080)
    assert seen == {"gpu_id": 7, "width": 1920, "height": 1080}


# ---------------------------------------------------------------------------
# probe_nvenc_available — deep probe (GetEncoderCaps)
# ---------------------------------------------------------------------------


def test_probe_deep_rejects_resolution_above_max(monkeypatch):
    """Deep probe fails when width/height exceed the caps-reported max."""
    fake_caps = {"width_max": 4096, "height_max": 2160, "width_min": 32, "height_min": 32}
    _install_fake_nvc(monkeypatch, caps=fake_caps)
    ok, reason = probe_nvenc_available(gpu_id=0, width=8192, height=4320)
    assert ok is False
    assert "exceeds NVENC H.264 max" in reason


def test_probe_deep_rejects_resolution_below_min(monkeypatch):
    """Deep probe fails when width/height are below the caps-reported min."""
    fake_caps = {"width_max": 4096, "height_max": 2160, "width_min": 128, "height_min": 128}
    _install_fake_nvc(monkeypatch, caps=fake_caps)
    ok, reason = probe_nvenc_available(gpu_id=0, width=32, height=32)
    assert ok is False
    assert "below NVENC H.264 min" in reason


def test_probe_deep_accepts_valid_resolution(monkeypatch):
    """Deep probe passes when caps are OK and resolution is within bounds."""
    fake_caps = {"width_max": 4096, "height_max": 2160, "width_min": 32, "height_min": 32}
    _install_fake_nvc(monkeypatch, caps=fake_caps)
    ok, reason = probe_nvenc_available(gpu_id=0, width=512, height=288)
    assert ok is True
    assert "NVENC H.264 caps OK" in reason


def test_probe_deep_handles_getencodercaps_raise(monkeypatch):
    """A raising ``GetEncoderCaps`` degrades the probe cleanly."""
    _install_fake_nvc(monkeypatch, caps_raises=RuntimeError("no NVENC on this GPU"))
    ok, reason = probe_nvenc_available(gpu_id=0, width=512, height=288)
    assert ok is False
    assert "GetEncoderCaps" in reason
    assert "no NVENC on this GPU" in reason


def test_probe_deep_handles_empty_caps(monkeypatch):
    """A ``GetEncoderCaps`` that returns nothing degrades the probe."""
    _install_fake_nvc(monkeypatch, caps={})
    ok, reason = probe_nvenc_available(gpu_id=0, width=512, height=288)
    assert ok is False
    assert "returned no capabilities" in reason


def _install_fake_nvc(monkeypatch, *, caps=None, caps_raises=None):
    """Install a fake PyNvVideoCodec module with a stub ``GetEncoderCaps``.

    Also patches ``torch.cuda.is_available`` / ``device_count`` so the
    shallow-probe portion succeeds.
    """
    import sys
    import types
    fake = types.SimpleNamespace()
    if caps_raises is not None:
        def _get(**_kw):
            raise caps_raises
        fake.GetEncoderCaps = _get
    else:
        fake.GetEncoderCaps = lambda **_kw: caps
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


# ---------------------------------------------------------------------------
# Environment variable
# ---------------------------------------------------------------------------


def test_read_encoder_env_unset(monkeypatch):
    monkeypatch.delenv("COSMOSH_VIDEO_ENCODER", raising=False)
    assert read_encoder_env() is None


def test_read_encoder_env_set(monkeypatch):
    monkeypatch.setenv("COSMOSH_VIDEO_ENCODER", "NVENC")
    assert read_encoder_env() == "nvenc"


def test_read_encoder_env_empty_is_none(monkeypatch):
    monkeypatch.setenv("COSMOSH_VIDEO_ENCODER", "")
    assert read_encoder_env() is None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_valid_encoders_set():
    assert VALID_ENCODERS == {ENCODER_AUTO, ENCODER_NVENC, ENCODER_CPU_LIBAV}
