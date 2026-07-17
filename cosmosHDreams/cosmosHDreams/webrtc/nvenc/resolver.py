# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder selection — auto / nvenc / cpu_libav.

Three values are accepted from YAML, CLI, or environment:

- ``auto`` — at startup, probe PyNvVideoCodec import and CUDA
  availability; pick ``nvenc`` if both are present, else fall back to
  ``cpu_libav``. Safe everywhere. Eventual default.
- ``nvenc`` — force the NVENC path. If unavailable, raise
  :class:`NvencUnavailableError` at startup so a misconfigured test rig
  fails fast instead of silently running on CPU.
- ``cpu_libav`` — force the existing libavcodec path. The current
  default in this PR.

Precedence (highest wins) is enforced by the caller:

1. ``--encoder`` CLI flag
2. ``COSMOSH_VIDEO_ENCODER`` environment variable
3. ``runtime.video.encoder`` YAML field
4. code default (``cpu_libav`` — the safe fallback when no other
   layer specifies)

The probe is intentionally permissive: it only checks that the Python
imports succeed and that a CUDA device is visible. The deeper NVENC
session check happens later when the encoder is actually constructed.
This keeps startup latency low and lets the resolver answer quickly
during ``preload_runtime``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

LOGGER = logging.getLogger(__name__)

ENCODER_AUTO = "auto"
ENCODER_NVENC = "nvenc"
ENCODER_CPU_LIBAV = "cpu_libav"

VALID_ENCODERS: frozenset[str] = frozenset(
    {ENCODER_AUTO, ENCODER_NVENC, ENCODER_CPU_LIBAV}
)

ENV_VAR_NAME = "COSMOSH_VIDEO_ENCODER"


class NvencUnavailableError(RuntimeError):
    """Raised when ``video.encoder='nvenc'`` is requested but NVENC is unusable."""


def normalize_encoder_value(value: Optional[str]) -> Optional[str]:
    """Lowercase + strip; return ``None`` for empty input.

    Useful for treating empty YAML fields, empty env vars, and missing
    CLI flags uniformly as "not specified, fall through to next layer".
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def select_requested_encoder(
    *,
    cli_value: Optional[str] = None,
    env_value: Optional[str] = None,
    yaml_value: Optional[str] = None,
    code_default: str = ENCODER_CPU_LIBAV,
) -> str:
    """Apply the four-layer precedence and return one of :data:`VALID_ENCODERS`.

    Each layer is consulted in order; the first non-empty value wins.
    The returned value is validated against :data:`VALID_ENCODERS` and
    is the "requested" string — not yet resolved against the probe.
    Call :func:`resolve_encoder` next to resolve ``auto`` and validate
    that ``nvenc`` is actually usable.
    """
    for value in (
        normalize_encoder_value(cli_value),
        normalize_encoder_value(env_value),
        normalize_encoder_value(yaml_value),
        normalize_encoder_value(code_default),
    ):
        if value is not None:
            if value not in VALID_ENCODERS:
                raise ValueError(
                    f"unknown video encoder {value!r}; valid values: "
                    + ", ".join(sorted(VALID_ENCODERS))
                )
            return value
    return ENCODER_CPU_LIBAV


def probe_nvenc_available(
    *,
    gpu_id: int = 0,
    width: int = 0,
    height: int = 0,
) -> tuple[bool, str]:
    """Probe whether the NVENC encode path can run in this process.

    Returns ``(ok, reason)``. ``reason`` is a short, human-readable
    string fragment intended for the startup log line.

    Two-phase check:

      1. **Shallow** — ``PyNvVideoCodec`` importable, a CUDA device
         visible via ``torch.cuda``. Cheap; the only check when
         ``width`` / ``height`` are unset.
      2. **Deep** — if ``width > 0 and height > 0``, additionally
         call ``PyNvVideoCodec.GetEncoderCaps(gpuid=gpu_id,
         codec='h264')`` and verify the target resolution falls
         within the driver-reported bounds. Cheap (does not
         allocate an encoder session) and catches the
         "silicon present but H.264 disabled" and
         "resolution out of range" cases that the shallow probe
         would let through, so a misconfigured rig fails at
         probe-time with a clear message instead of dying later
         at ``CreateEncoder`` with a generic error.
    """
    try:
        import PyNvVideoCodec as nvc  # type: ignore[import-untyped]
    except ImportError as exc:
        return False, f"PyNvVideoCodec not importable ({exc})"
    try:
        import torch
    except ImportError as exc:
        return False, f"torch not importable ({exc})"
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        device_count = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001 - torch.cuda probe can throw various
        return False, f"CUDA probe failed ({exc})"

    # Shallow probe done. If the caller passed a target resolution,
    # deepen the check with GetEncoderCaps.
    if width > 0 and height > 0:
        try:
            # PyNvVideoCodec 2.1+ signature: GetEncoderCaps(gpuid, codec).
            # ``gpuid`` (no underscore) is the keyword name in the C
            # binding — mistyping to ``gpu_id`` raises TypeError.
            caps = nvc.GetEncoderCaps(gpuid=gpu_id, codec="h264")
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"GetEncoderCaps(gpuid={gpu_id}, codec='h264') raised "
                f"{type(exc).__name__}: {exc}"
            )
        if not caps:
            return False, (
                f"GetEncoderCaps(gpuid={gpu_id}, codec='h264') "
                "returned no capabilities"
            )
        max_w = int(caps.get("width_max", 0) or 0)
        max_h = int(caps.get("height_max", 0) or 0)
        min_w = int(caps.get("width_min", 0) or 0)
        min_h = int(caps.get("height_min", 0) or 0)
        if max_w > 0 and max_h > 0 and (width > max_w or height > max_h):
            return False, (
                f"resolution {width}x{height} exceeds NVENC H.264 max "
                f"{max_w}x{max_h} on GPU {gpu_id}"
            )
        if (min_w > 0 or min_h > 0) and (width < min_w or height < min_h):
            return False, (
                f"resolution {width}x{height} is below NVENC H.264 min "
                f"{min_w}x{min_h} on GPU {gpu_id}"
            )
        return True, (
            f"PyNvVideoCodec + {device_count} CUDA device(s) + "
            f"NVENC H.264 caps OK (max {max_w}x{max_h})"
        )

    return True, f"PyNvVideoCodec import + {device_count} CUDA device(s)"


def resolve_encoder(
    requested: str,
    *,
    gpu_id: int = 0,
    width: int = 0,
    height: int = 0,
) -> tuple[str, str]:
    """Resolve a requested encoder value to the encoder that will actually run.

    Args:
        requested: one of :data:`VALID_ENCODERS`. Typically the return
            value of :func:`select_requested_encoder`.
        gpu_id / width / height: forwarded to
            :func:`probe_nvenc_available`. When ``width`` and ``height``
            are both > 0, the probe queries ``GetEncoderCaps`` and
            verifies the target resolution — recommended so that
            "auto → nvenc" doesn't succeed on a GPU that can't
            actually encode the requested resolution.

    Returns:
        ``(active, reason)`` where ``active`` is one of
        ``ENCODER_NVENC`` or ``ENCODER_CPU_LIBAV`` (never ``auto``),
        and ``reason`` is a one-line human-readable explanation
        suitable for a startup log line.

    Raises:
        ValueError: ``requested`` is not in :data:`VALID_ENCODERS`.
        NvencUnavailableError: ``requested == 'nvenc'`` but the probe
            says NVENC is not usable.
    """
    if requested not in VALID_ENCODERS:
        raise ValueError(
            f"unknown video encoder {requested!r}; valid values: "
            + ", ".join(sorted(VALID_ENCODERS))
        )

    if requested == ENCODER_CPU_LIBAV:
        return ENCODER_CPU_LIBAV, "explicit (cpu_libav)"

    probe_kwargs = {"gpu_id": gpu_id, "width": width, "height": height}

    if requested == ENCODER_NVENC:
        ok, why = probe_nvenc_available(**probe_kwargs)
        if not ok:
            raise NvencUnavailableError(
                "video.encoder='nvenc' was requested explicitly but NVENC "
                f"is not usable: {why}. Set video.encoder='auto' to fall "
                "back gracefully, or set video.encoder='cpu_libav' to "
                "force the existing path."
            )
        return ENCODER_NVENC, f"explicit (nvenc; {why})"

    # auto
    ok, why = probe_nvenc_available(**probe_kwargs)
    if ok:
        return ENCODER_NVENC, f"auto -> nvenc ({why})"
    return ENCODER_CPU_LIBAV, f"auto -> cpu_libav (nvenc unavailable: {why})"


def read_encoder_env() -> Optional[str]:
    """Return the value of :data:`ENV_VAR_NAME`, or ``None`` if unset/empty."""
    return normalize_encoder_value(os.environ.get(ENV_VAR_NAME))
