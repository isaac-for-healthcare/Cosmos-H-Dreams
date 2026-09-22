# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quest 3 controller state + action-chunk computation.

Counterpart to :mod:`cosmosHDreams.webrtc.controls` (keyboard). Wire schema::

    {"type": "vr_input", "t_ms": float,
     "right": {"dpos": [dx, dy, dz], "drot": [rx, ry, rz], "trigger": float},
     "left":  {"dpos": [dx, dy, dz], "drot": [rx, ry, rz], "trigger": float}}

Each arm carries:
- ``dpos``: controller position delta over one browser frame (meters).
- ``drot``: controller orientation delta as an axis-angle 3-vector
  (magnitude = angle in radians).
- ``trigger``: analog trigger value in ``[0.0, 1.0]``
  (``gamepad.buttons[0].value``), lerped into that arm's gripper dim —
  squeeze to close, matching dVRK master-grip semantics.

Right controller drives PSM1 (right arm in camera frame, dims 0..9); left
controller drives PSM2 (left arm in camera frame, dims 10..19). Other
buttons (A/grip/thumbstick) are intentionally not wired.

Both grippers rest at their ``OPEN`` endpoint (right) / PSM2_GRIPPER_OPEN
(left): the wider dynamic range gives the model something visible to push
against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cosmosHDreams.webrtc.controls import (
    PSM1_GRIPPER_CLOSED,
    PSM1_GRIPPER_DIM,
    PSM1_GRIPPER_OPEN,
    PSM2_GRIPPER_CLOSED,
    PSM2_GRIPPER_DIM,
    PSM2_GRIPPER_OPEN,
)
from cosmosHDreams.webrtc.utils import (
    ACTION_DIM_NORMALISED,
    write_rotation_ramp,
    write_translate_ramp,
)

# Slice starts within the 20-dim action layout (see ``utils.ACTION_DIM_NORMALISED``).
_PSM1_TRANSLATE_SLICE_START = 0
_PSM1_ROT6D_SLICE_START = 3
_PSM2_TRANSLATE_SLICE_START = 10
_PSM2_ROT6D_SLICE_START = 13


def _payload_to_vec(value: Any, length: int) -> np.ndarray:
    """Coerce a JSON list to an ``np.float64`` vector of the given length.

    Returns a zero vector on any shape / type / non-finite mismatch — one
    malformed message becomes "no motion this frame" rather than crashing
    the loop or carrying stale state forward.
    """
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return np.zeros(length, dtype=np.float64)
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(length, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return np.zeros(length, dtype=np.float64)
    return arr


def _payload_to_float(value: Any) -> float:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return 0.0


@dataclass(slots=True)
class VRArmInput:
    """Latest sample for one controller (one arm).

    Fields drive a single arm's slice of the action chunk:
    - ``dpos``: per-browser-frame position delta (meters, play-space).
    - ``drot``: per-browser-frame orientation delta (axis-angle, radians).
    - ``trigger``: analog trigger in ``[0.0, 1.0]``.
    """

    dpos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    drot: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    trigger: float = 0.0

    def _apply(self, payload: dict[str, Any] | None) -> None:
        """Overwrite from a per-arm sub-payload (right or left section)."""
        if not isinstance(payload, dict):
            payload = {}
        self.dpos = _payload_to_vec(payload.get("dpos"), 3)
        self.drot = _payload_to_vec(payload.get("drot"), 3)
        self.trigger = float(
            np.clip(_payload_to_float(payload.get("trigger")), 0.0, 1.0)
        )


@dataclass(slots=True)
class VRControllerState:
    """Latest two-arm sample. Right → PSM1, left → PSM2.

    The integrator reads this at chunk start; the WebSocket handler
    overwrites it on every ``vr_input`` message. ``t_ms`` is debug-only.
    Missing arm sections default to all-zero — the corresponding chunk
    slice writes resting values (no motion, gripper OPEN).
    """

    right: VRArmInput = field(default_factory=VRArmInput)
    left: VRArmInput = field(default_factory=VRArmInput)
    t_ms: float = 0.0  # time.perf_counter() * 1000.0 at server receive time

    def apply_vr_input(self, payload: dict[str, Any], recv_t_ms: float = 0.0) -> bool:
        """Replace both arms' state from a ``vr_input`` payload.

        Returns ``True`` if the payload had the right shape; ``False`` for
        non-``vr_input`` messages or non-dict payloads. Missing per-arm
        sections fall through to a zero arm input — one bad / dropped frame
        becomes "no motion this frame" rather than stale carry-over.

        ``recv_t_ms`` must be ``time.perf_counter() * 1000.0`` stamped by the
        caller immediately on receive — the same clock used in
        ``_generate_one_chunk_vr_sync`` so the subtraction is meaningful.
        """
        if not isinstance(payload, dict) or payload.get("type") != "vr_input":
            return False
        self.right._apply(payload.get("right"))
        self.left._apply(payload.get("left"))
        self.t_ms = recv_t_ms
        return True


def _as_translate_scale_vec(value: Any) -> np.ndarray:
    """Coerce a scalar or 3-element sequence into a length-3 ``float64`` vec.

    Scalars broadcast to ``[s, s, s]`` so the historical scalar API still
    works for callers (and tests) that don't care about per-axis tuning.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    if arr.shape == (3,):
        return arr
    raise ValueError(
        f"translate_scale must be a scalar or 3-element sequence; got shape {arr.shape}"
    )


def _resolve_per_arm_translate(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(right_vec3, left_vec3)`` from any accepted input shape.

    Accepts:
    - scalar or 3-element sequence → broadcast to both arms.
    - ``{"right": ..., "left": ...}`` dict → per-arm (both keys required;
      each value is itself a scalar or 3-vec).
    """
    if isinstance(value, dict):
        for k in ("right", "left"):
            if k not in value:
                raise ValueError(
                    f"translate_scale dict must include both 'right' and "
                    f"'left' keys; got {sorted(value.keys())}"
                )
        return (
            _as_translate_scale_vec(value["right"]),
            _as_translate_scale_vec(value["left"]),
        )
    vec = _as_translate_scale_vec(value)
    return vec, vec.copy()


def _resolve_per_arm_rotate(value: Any) -> tuple[float, float]:
    """Return ``(right_scalar, left_scalar)`` from any accepted input shape."""
    if isinstance(value, dict):
        for k in ("right", "left"):
            if k not in value:
                raise ValueError(
                    f"rotate_scale dict must include both 'right' and "
                    f"'left' keys; got {sorted(value.keys())}"
                )
        return float(value["right"]), float(value["left"])
    s = float(value)
    return s, s


def _quest_to_camera_axes(v: np.ndarray) -> np.ndarray:
    """Empirical remap from Quest play-space to camera-frame xyz.

    Found by user testing on the right controller for ``dpos``; reused
    *tentatively* for ``drot`` (axis-angle vectors compose differently
    from polar vectors under sign flips — expect headset sweeps to flip
    individual components) and for the left controller (which is the
    play-space mirror image — signs may not all match the right side).

    Layout:
        chunk dim 0 (x) ← -v[0]
        chunk dim 1 (y) ← -v[2]
        chunk dim 2 (z) ← -v[1]
    """
    return np.array([-v[0], -v[2], -v[1]], dtype=np.float64)


def _write_arm(
    chunk: np.ndarray,
    *,
    arm: VRArmInput,
    num_frames: int,
    start_pos: np.ndarray | None = None,
    translate_scale: np.ndarray,
    rotate_scale: float,
    translate_slice_start: int,
    rot6d_slice_start: int,
    gripper_dim: int,
    gripper_open: float,
    gripper_closed: float,
    rot6d_mean: np.ndarray | None,
    rot6d_std: np.ndarray | None,
    rot6d_identity_norm: np.ndarray | None,
) -> None:
    """Write one arm's (translate + rotation + gripper) slice into ``chunk``.

    ``start_pos`` is the arm's tracked position in normalised action space
    (shape ``[3]``). When provided it is mutated in place to reflect the arm's
    position at the end of this chunk, so the caller can pass it again on the
    next call to maintain continuity. Defaults to zeros (arm at rest) when
    ``None``.

    ``translate_scale`` is a length-3 float64 vec applied **element-wise to
    play-space dpos** *before* the camera-frame remap — so caller-facing
    [x, y, z] correspond to the user's physical hand axes, not the
    camera-frame action axes.
    """
    if start_pos is None:
        start_pos = np.zeros(3, dtype=np.float64)
    scaled_dpos = arm.dpos * translate_scale
    v_xyz = _quest_to_camera_axes(scaled_dpos).astype(np.float64)
    write_translate_ramp(
        chunk,
        num_frames=num_frames,
        v_xyz=v_xyz,
        slice_start=translate_slice_start,
        start_pos=start_pos.copy(),
    )
    start_pos[:] += num_frames * v_xyz

    if (
        rotate_scale > 0.0
        and rot6d_mean is not None
        and rot6d_std is not None
        and rot6d_identity_norm is not None
    ):
        omega = _quest_to_camera_axes(arm.drot) * rotate_scale
        write_rotation_ramp(
            chunk,
            num_frames=num_frames,
            omega=omega,
            mean=rot6d_mean,
            std=rot6d_std,
            identity_norm=rot6d_identity_norm,
            slice_start=rot6d_slice_start,
        )

    gripper = gripper_open + (gripper_closed - gripper_open) * arm.trigger
    chunk[:, gripper_dim] = np.float32(gripper)


def compute_action_chunk(
    state: VRControllerState,
    *,
    num_frames: int = 12,
    psm1_pos: np.ndarray | None = None,
    psm2_pos: np.ndarray | None = None,
    translate_scale: Any,
    rotate_scale: Any = 0.0,
    psm1_rot6d_mean: np.ndarray | None = None,
    psm1_rot6d_std: np.ndarray | None = None,
    psm1_rot6d_identity_norm: np.ndarray | None = None,
    psm2_rot6d_mean: np.ndarray | None = None,
    psm2_rot6d_std: np.ndarray | None = None,
    psm2_rot6d_identity_norm: np.ndarray | None = None,
    psm1_gripper_open: float = PSM1_GRIPPER_OPEN,
    psm1_gripper_closed: float = PSM1_GRIPPER_CLOSED,
    psm2_gripper_open: float = PSM2_GRIPPER_OPEN,
    psm2_gripper_closed: float = PSM2_GRIPPER_CLOSED,
) -> np.ndarray:
    """Build a ``(num_frames, 20)`` action chunk from the latest Quest sample.

    Right controller → PSM1 (translate dims 0..2, rot6d 3..8, gripper 9);
    left controller → PSM2 (translate 10..12, rot6d 13..18, gripper 19).
    Same per-arm pipeline on both sides: scale play-space dpos per axis,
    apply the camera-frame axis remap, then ramp into the slice.

    ``translate_scale`` accepts:
    - scalar or 3-element sequence ``[x, y, z]`` (play-space axes) →
      broadcast to both arms.
    - ``{"right": <scalar|3-vec>, "left": <scalar|3-vec>}`` → asymmetric.

    ``rotate_scale`` accepts:
    - scalar → broadcast to both arms.
    - ``{"right": <scalar>, "left": <scalar>}`` → asymmetric.

    Rotation is opt-in per arm: only written when that arm's resolved
    ``rotate_scale > 0`` and the arm's three rot6d stats are supplied.
    Keeps the function callable from tests / code paths that only care
    about translate / gripper.

    Gripper mapping (squeeze-to-close, dVRK master-grip semantics):
    ``trigger=0`` → ``OPEN`` endpoint, ``trigger=1`` → ``CLOSED`` endpoint.
    Per-arm endpoints default to the ``cosmosHDreams.webrtc.controls`` module
    constants but are normally supplied by the runtime from the loaded stats
    (q01/q99 → normalised) so the range matches the active model. Written
    constant across the chunk because ``gripper`` is absolute every frame
    (``actions.md`` "Per-frame semantics"), not a delta.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    chunk = np.zeros((num_frames, ACTION_DIM_NORMALISED), dtype=np.float32)
    right_translate, left_translate = _resolve_per_arm_translate(translate_scale)
    right_rotate, left_rotate = _resolve_per_arm_rotate(rotate_scale)

    _write_arm(
        chunk,
        arm=state.right,
        num_frames=num_frames,
        start_pos=psm1_pos,
        translate_scale=right_translate,
        rotate_scale=right_rotate,
        translate_slice_start=_PSM1_TRANSLATE_SLICE_START,
        rot6d_slice_start=_PSM1_ROT6D_SLICE_START,
        gripper_dim=PSM1_GRIPPER_DIM,
        gripper_open=psm1_gripper_open,
        gripper_closed=psm1_gripper_closed,
        rot6d_mean=psm1_rot6d_mean,
        rot6d_std=psm1_rot6d_std,
        rot6d_identity_norm=psm1_rot6d_identity_norm,
    )
    _write_arm(
        chunk,
        arm=state.left,
        num_frames=num_frames,
        start_pos=psm2_pos,
        translate_scale=left_translate,
        rotate_scale=left_rotate,
        translate_slice_start=_PSM2_TRANSLATE_SLICE_START,
        rot6d_slice_start=_PSM2_ROT6D_SLICE_START,
        gripper_dim=PSM2_GRIPPER_DIM,
        gripper_open=psm2_gripper_open,
        gripper_closed=psm2_gripper_closed,
        rot6d_mean=psm2_rot6d_mean,
        rot6d_std=psm2_rot6d_std,
        rot6d_identity_norm=psm2_rot6d_identity_norm,
    )
    return chunk
