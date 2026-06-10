from __future__ import annotations

import numpy as np
import pytest
from cosmosh.webrtc.controls import (
    PSM1_GRIPPER_CLOSED,
    PSM1_GRIPPER_DIM,
    PSM1_GRIPPER_OPEN,
    PSM2_GRIPPER_CLOSED,
    PSM2_GRIPPER_DIM,
    PSM2_GRIPPER_OPEN,
)
from cosmosh.webrtc.controls_quest import (
    VRArmInput,
    VRControllerState,
    compute_action_chunk,
)
from cosmosh.webrtc.utils import (
    ACTION_DIM_NORMALISED,
    IDENTITY_ROT6D,
    matrix_to_rot6d,
    rotvec_to_matrix,
)

pytestmark = pytest.mark.ci_cpu


# Default rot6d stats used by the rotation tests below. Identity mean +
# unit std + zero identity baseline → the rot6d ramp formula collapses to
# ``rot6d(exp((f+1)·ω)) - identity_rot6d``. Makes hand-computed expectations
# easy without dragging stats_cosmos.json into the test fixture.
_UNIT_MEAN = np.zeros(6, dtype=np.float64)
_UNIT_STD = np.ones(6, dtype=np.float64)
_IDENTITY_ROT6D_NORM_UNIT = IDENTITY_ROT6D.copy()  # (IDENTITY - 0) / 1 = IDENTITY

# PSM1 covers chunk dims 0..9 (xyz, rot6d, gripper); PSM2 covers 10..19.
_PSM1_TRANSLATE_SLICE = slice(0, 3)
_PSM1_ROT6D_SLICE = slice(3, 9)
_PSM2_TRANSLATE_SLICE = slice(10, 13)
_PSM2_ROT6D_SLICE = slice(13, 19)


# ----- Payload + state construction helpers -----------------------------


def _arm_payload(*, dpos=(0.0, 0.0, 0.0), drot=(0.0, 0.0, 0.0), trigger=0.0):
    return {"dpos": list(dpos), "drot": list(drot), "trigger": trigger}


def _vr_payload(*, t_ms=0.0, right=None, left=None):
    """Build a wire-format ``vr_input`` payload.

    Arm sections default to ``None`` (omitted); pass the result of
    :func:`_arm_payload` to populate one. Mirrors the nested schema the
    browser actually sends.
    """
    payload = {"type": "vr_input", "t_ms": t_ms}
    if right is not None:
        payload["right"] = right
    if left is not None:
        payload["left"] = left
    return payload


def _arm(*, dpos=(0.0, 0.0, 0.0), drot=(0.0, 0.0, 0.0), trigger=0.0) -> VRArmInput:
    """Construct a ``VRArmInput`` directly. Used by behavior tests so they
    can skip the parsing layer."""
    return VRArmInput(
        dpos=np.array(dpos, dtype=np.float64),
        drot=np.array(drot, dtype=np.float64),
        trigger=trigger,
    )


def _mask_all_other_dims(*indices: int) -> np.ndarray:
    mask = np.ones(ACTION_DIM_NORMALISED, dtype=bool)
    for idx in indices:
        mask[idx] = False
    return mask


# ----- VRControllerState: parsing + state updates -----------------------


def test_vr_controller_state_default_is_zero() -> None:
    state = VRControllerState()
    for arm in (state.right, state.left):
        assert np.all(arm.dpos == 0.0)
        assert np.all(arm.drot == 0.0)
        assert arm.trigger == 0.0
    assert state.t_ms == 0.0


def test_vr_controller_state_apply_updates_both_arms() -> None:
    state = VRControllerState()
    assert state.apply_vr_input(
        _vr_payload(
            t_ms=1234.5,
            right=_arm_payload(
                dpos=(0.01, -0.02, 0.03),
                drot=(0.04, -0.05, 0.06),
                trigger=0.7,
            ),
            left=_arm_payload(
                dpos=(-0.01, 0.02, -0.03),
                drot=(-0.04, 0.05, -0.06),
                trigger=0.3,
            ),
        )
    )
    assert np.allclose(state.right.dpos, [0.01, -0.02, 0.03])
    assert np.allclose(state.right.drot, [0.04, -0.05, 0.06])
    assert state.right.trigger == 0.7
    assert np.allclose(state.left.dpos, [-0.01, 0.02, -0.03])
    assert np.allclose(state.left.drot, [-0.04, 0.05, -0.06])
    assert state.left.trigger == 0.3
    assert state.t_ms == 1234.5


def test_vr_controller_state_missing_arm_section_zeroes_that_arm() -> None:
    """A payload without a ``left`` section zeroes the left arm.

    Contract: each apply_vr_input call is the latest-sample-wins snapshot;
    a single bad / dropped arm becomes "no motion this frame", never
    stale carry-over from a prior frame.
    """
    state = VRControllerState()
    # Seed both arms with non-zero state.
    state.apply_vr_input(
        _vr_payload(
            right=_arm_payload(dpos=(0.1, 0.0, 0.0), trigger=0.5),
            left=_arm_payload(dpos=(0.2, 0.0, 0.0), trigger=0.5),
        )
    )
    assert np.allclose(state.right.dpos, [0.1, 0.0, 0.0])
    assert np.allclose(state.left.dpos, [0.2, 0.0, 0.0])
    # Now send a payload with only the right section.
    state.apply_vr_input(_vr_payload(right=_arm_payload(dpos=(0.3, 0.0, 0.0))))
    assert np.allclose(state.right.dpos, [0.3, 0.0, 0.0])
    assert np.all(state.left.dpos == 0.0)
    assert state.left.trigger == 0.0


def test_vr_controller_state_malformed_arm_fields_zero_out() -> None:
    state = VRControllerState()
    state.apply_vr_input(
        _vr_payload(right=_arm_payload(dpos=(0.5, 0.5, 0.5), drot=(0.1, 0.2, 0.3)))
    )
    assert np.allclose(state.right.dpos, [0.5, 0.5, 0.5])
    assert np.allclose(state.right.drot, [0.1, 0.2, 0.3])
    # Malformed dpos / drot → zero vectors. One bad frame can't carry
    # stale motion forward.
    state.apply_vr_input(
        _vr_payload(right={"dpos": "not a list", "drot": 7, "trigger": "bad"})
    )
    assert np.all(state.right.dpos == 0.0)
    assert np.all(state.right.drot == 0.0)
    assert state.right.trigger == 0.0


def test_vr_controller_state_clips_trigger_to_unit_interval() -> None:
    state = VRControllerState()
    state.apply_vr_input(_vr_payload(right=_arm_payload(trigger=1.5)))
    assert state.right.trigger == 1.0
    state.apply_vr_input(_vr_payload(left=_arm_payload(trigger=-0.2)))
    assert state.left.trigger == 0.0


def test_vr_controller_state_rejects_non_vr_input() -> None:
    state = VRControllerState()
    assert not state.apply_vr_input({"type": "action", "key": "w"})
    assert not state.apply_vr_input("not a dict")  # type: ignore[arg-type]


def test_vr_controller_state_rejects_non_finite_dpos() -> None:
    state = VRControllerState()
    state.apply_vr_input(
        _vr_payload(right=_arm_payload(dpos=(float("nan"), 0.0, 0.0)))
    )
    assert np.all(state.right.dpos == 0.0)


# ----- compute_action_chunk: zero / gripper rest ------------------------


def test_compute_action_chunk_zero_state_rests_both_grippers_open() -> None:
    """At-rest input: both grippers at their OPEN endpoint, all else zero.

    Per ``actions.md``, gripper is absolute every frame (not a delta), so
    the chunk must carry an explicit value every iteration. ``trigger=0``
    (controller trigger not pulled) anchors each arm at its OPEN endpoint;
    squeezing lerps toward CLOSED.
    """
    chunk = compute_action_chunk(
        VRControllerState(), num_frames=12, translate_scale=500.0
    )
    assert chunk.shape == (12, ACTION_DIM_NORMALISED)
    assert chunk.dtype == np.float32
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], PSM1_GRIPPER_OPEN)
    assert np.allclose(chunk[:, PSM2_GRIPPER_DIM], PSM2_GRIPPER_OPEN)
    mask = _mask_all_other_dims(PSM1_GRIPPER_DIM, PSM2_GRIPPER_DIM)
    assert np.all(chunk[:, mask] == 0.0)


def test_compute_action_chunk_right_trigger_closes_psm1_gripper() -> None:
    state = VRControllerState(right=_arm(trigger=1.0))
    chunk = compute_action_chunk(state, num_frames=4, translate_scale=500.0)
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], PSM1_GRIPPER_CLOSED)
    # PSM2 stays at rest (trigger=0 → OPEN).
    assert np.allclose(chunk[:, PSM2_GRIPPER_DIM], PSM2_GRIPPER_OPEN)


def test_compute_action_chunk_left_trigger_closes_psm2_gripper() -> None:
    state = VRControllerState(left=_arm(trigger=1.0))
    chunk = compute_action_chunk(state, num_frames=4, translate_scale=500.0)
    assert np.allclose(chunk[:, PSM2_GRIPPER_DIM], PSM2_GRIPPER_CLOSED)
    # PSM1 stays at rest.
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], PSM1_GRIPPER_OPEN)


def test_compute_action_chunk_trigger_midpoints_lerp_per_arm() -> None:
    state = VRControllerState(
        right=_arm(trigger=0.5),
        left=_arm(trigger=0.5),
    )
    chunk = compute_action_chunk(state, num_frames=3, translate_scale=500.0)
    psm1_mid = (PSM1_GRIPPER_OPEN + PSM1_GRIPPER_CLOSED) / 2.0
    psm2_mid = (PSM2_GRIPPER_OPEN + PSM2_GRIPPER_CLOSED) / 2.0
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], psm1_mid)
    assert np.allclose(chunk[:, PSM2_GRIPPER_DIM], psm2_mid)


# ----- compute_action_chunk: translate ramps ----------------------------


def test_compute_action_chunk_right_translate_lands_on_psm1_slice() -> None:
    """Browser dpos = +x on right → chunk[:, 0] negative ramp; PSM2 untouched."""
    state = VRControllerState(right=_arm(dpos=(0.01, 0.0, 0.0)))
    chunk = compute_action_chunk(state, num_frames=12, translate_scale=500.0)
    expected_x = np.array([-(i + 1) * 5.0 for i in range(12)], dtype=np.float32)
    assert np.allclose(chunk[:, 0], expected_x)
    assert np.all(chunk[:, 1:3] == 0.0)
    # PSM2 translate / rot6d / gripper-delta all stay zero.
    assert np.all(chunk[:, _PSM2_TRANSLATE_SLICE] == 0.0)
    assert np.all(chunk[:, _PSM2_ROT6D_SLICE] == 0.0)


def test_compute_action_chunk_left_translate_lands_on_psm2_slice() -> None:
    """Browser dpos = +x on left → chunk[:, 10] negative ramp; PSM1 untouched."""
    state = VRControllerState(left=_arm(dpos=(0.01, 0.0, 0.0)))
    chunk = compute_action_chunk(state, num_frames=12, translate_scale=500.0)
    expected_x = np.array([-(i + 1) * 5.0 for i in range(12)], dtype=np.float32)
    assert np.allclose(chunk[:, 10], expected_x)
    assert np.all(chunk[:, 11:13] == 0.0)
    # PSM1 translate / rot6d stay zero.
    assert np.all(chunk[:, _PSM1_TRANSLATE_SLICE] == 0.0)
    assert np.all(chunk[:, _PSM1_ROT6D_SLICE] == 0.0)


def test_compute_action_chunk_right_browser_y_lands_on_chunk_z() -> None:
    """Browser y → chunk z (dim 2) per empirical remap: chunk[:, 2] = -dpos[1]·scale."""
    state = VRControllerState(right=_arm(dpos=(0.0, 0.05, 0.0)))
    chunk = compute_action_chunk(state, num_frames=2, translate_scale=100.0)
    assert np.allclose(chunk[:, 0], 0.0)
    assert np.allclose(chunk[:, 1], 0.0)
    assert np.allclose(chunk[:, 2], np.array([-5.0, -10.0], dtype=np.float32))


def test_compute_action_chunk_right_browser_z_lands_on_chunk_y() -> None:
    """Browser z → chunk y (dim 1) per empirical remap: chunk[:, 1] = -dpos[2]·scale."""
    state = VRControllerState(right=_arm(dpos=(0.0, 0.0, 0.05)))
    chunk = compute_action_chunk(state, num_frames=2, translate_scale=100.0)
    assert np.allclose(chunk[:, 0], 0.0)
    assert np.allclose(chunk[:, 1], np.array([-5.0, -10.0], dtype=np.float32))
    assert np.allclose(chunk[:, 2], 0.0)


def test_compute_action_chunk_both_arms_translate_independently() -> None:
    # Right dpos = (+0.01, -0.02, +0.03), Left dpos = (-0.01, +0.02, -0.03).
    # Same remap on both: (-dpos[0], -dpos[2], -dpos[1]) * scale.
    state = VRControllerState(
        right=_arm(dpos=(0.01, -0.02, 0.03)),
        left=_arm(dpos=(-0.01, 0.02, -0.03)),
    )
    chunk = compute_action_chunk(state, num_frames=2, translate_scale=100.0)
    assert np.allclose(chunk[0, 0:3], [-1.0, -3.0, 2.0])
    assert np.allclose(chunk[1, 0:3], [-2.0, -6.0, 4.0])
    assert np.allclose(chunk[0, 10:13], [1.0, 3.0, -2.0])
    assert np.allclose(chunk[1, 10:13], [2.0, 6.0, -4.0])


def test_compute_action_chunk_rejects_zero_num_frames() -> None:
    state = VRControllerState()
    try:
        compute_action_chunk(state, num_frames=0, translate_scale=500.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for num_frames=0")


def test_compute_action_chunk_per_axis_translate_scale() -> None:
    """Per-axis ``translate_scale`` multiplies dpos element-wise in play-space.

    Use-case: amplify vertical hand motion (play-space y → camera-frame
    depth) without affecting horizontal axes. dpos = (+0.01, +0.02, +0.03);
    scale = [100, 1000, 100] → scaled_dpos in play-space = (1, 20, 3).
    After the ``[-x, -z, -y]`` remap that's v_xyz = (-1, -3, -20). Ramp
    multiplies by (f + 1).
    """
    state = VRControllerState(right=_arm(dpos=(0.01, 0.02, 0.03)))
    chunk = compute_action_chunk(
        state,
        num_frames=2,
        translate_scale=[100.0, 1000.0, 100.0],
    )
    assert np.allclose(chunk[0, 0:3], [-1.0, -3.0, -20.0])
    assert np.allclose(chunk[1, 0:3], [-2.0, -6.0, -40.0])
    # Sanity: scalar form gives the same result as broadcasting.
    chunk_scalar = compute_action_chunk(
        VRControllerState(right=_arm(dpos=(0.01, 0.02, 0.03))),
        num_frames=2,
        translate_scale=100.0,
    )
    # The y-scaled variant only differs on dim 2 (where play-space y lands).
    assert np.allclose(chunk[:, 0], chunk_scalar[:, 0])
    assert np.allclose(chunk[:, 1], chunk_scalar[:, 1])
    assert not np.allclose(chunk[:, 2], chunk_scalar[:, 2])


def test_compute_action_chunk_rejects_bad_translate_scale_shape() -> None:
    state = VRControllerState()
    for bad in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], "not a number"):
        try:
            compute_action_chunk(state, num_frames=2, translate_scale=bad)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"expected ValueError/TypeError for translate_scale={bad!r}")


# ----- compute_action_chunk: per-arm scales -----------------------------


def test_compute_action_chunk_per_arm_translate_scale_dict() -> None:
    """Asymmetric translate_scale.right vs translate_scale.left lands per-arm.

    Same dpos on both arms; right gets the bigger scale, left gets the
    smaller — the resulting chunk slices should differ proportionally.
    """
    state = VRControllerState(
        right=_arm(dpos=(0.01, 0.0, 0.0)),
        left=_arm(dpos=(0.01, 0.0, 0.0)),
    )
    chunk = compute_action_chunk(
        state,
        num_frames=2,
        translate_scale={"right": 1000.0, "left": 100.0},
    )
    # Remap: chunk[:, 0] = -dpos[0] * scale, ramped by (f+1).
    assert np.allclose(chunk[:, 0], [-10.0, -20.0])      # right
    assert np.allclose(chunk[:, 10], [-1.0, -2.0])       # left
    # Other axes stay zero on both arms.
    assert np.all(chunk[:, 1:3] == 0.0)
    assert np.all(chunk[:, 11:13] == 0.0)


def test_compute_action_chunk_per_arm_rotate_scale_dict() -> None:
    """Asymmetric rotate_scale.right vs left controls each arm's rot6d ramp.

    With right=0 → PSM1 rot6d stays zero even though right.drot is non-zero.
    With left>0 + stats → PSM2 rot6d follows the usual ramp.
    """
    drot = np.array([0.05, 0.0, 0.0], dtype=np.float64)
    state = VRControllerState(
        right=_arm(drot=tuple(drot)),
        left=_arm(drot=tuple(drot)),
    )
    chunk = compute_action_chunk(
        state,
        num_frames=3,
        translate_scale=500.0,
        rotate_scale={"right": 0.0, "left": 2.0},
        psm1_rot6d_mean=_UNIT_MEAN,
        psm1_rot6d_std=_UNIT_STD,
        psm1_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
        psm2_rot6d_mean=_UNIT_MEAN,
        psm2_rot6d_std=_UNIT_STD,
        psm2_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
    )
    # right rotate_scale=0 → PSM1 rot6d slice stays exactly zero.
    assert np.all(chunk[:, _PSM1_ROT6D_SLICE] == 0.0)
    # left rotate_scale=2 → PSM2 rot6d matches the expected ramp.
    expected = _expected_rot6d_ramp(drot=drot, rotate_scale=2.0, num_frames=3)
    assert np.allclose(chunk[:, _PSM2_ROT6D_SLICE], expected, atol=1e-5)


def test_compute_action_chunk_per_arm_dict_requires_both_keys() -> None:
    state = VRControllerState()
    try:
        compute_action_chunk(
            state, num_frames=2, translate_scale={"right": 100.0}
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError when 'left' key is missing")


# ----- compute_action_chunk: rotation path ------------------------------


def test_compute_action_chunk_skips_rotation_when_rotate_scale_zero() -> None:
    """rotate_scale=0 (default) → both rot6d slices stay zero regardless of drot."""
    state = VRControllerState(
        right=_arm(drot=(0.5, 0.5, 0.5)),
        left=_arm(drot=(0.5, 0.5, 0.5)),
    )
    chunk = compute_action_chunk(state, num_frames=4, translate_scale=500.0)
    assert np.all(chunk[:, _PSM1_ROT6D_SLICE] == 0.0)
    assert np.all(chunk[:, _PSM2_ROT6D_SLICE] == 0.0)


def test_compute_action_chunk_skips_arm_rotation_when_its_stats_missing() -> None:
    """Per-arm gating: passing psm1 stats writes PSM1 rot6d, no psm2 stats → PSM2 stays zero."""
    state = VRControllerState(
        right=_arm(drot=(0.1, 0.0, 0.0)),
        left=_arm(drot=(0.1, 0.0, 0.0)),
    )
    chunk = compute_action_chunk(
        state,
        num_frames=4,
        translate_scale=500.0,
        rotate_scale=1.0,
        psm1_rot6d_mean=_UNIT_MEAN,
        psm1_rot6d_std=_UNIT_STD,
        psm1_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
        # psm2_* deliberately omitted.
    )
    assert not np.all(chunk[:, _PSM1_ROT6D_SLICE] == 0.0)
    assert np.all(chunk[:, _PSM2_ROT6D_SLICE] == 0.0)


def test_compute_action_chunk_zero_drot_leaves_rot6d_zero() -> None:
    """drot=0 + rotation stats → both rot6d slices stay exactly zero.

    The identity-baseline subtraction inside ``write_rotation_ramp`` is the
    feature being asserted: ``rot6d(exp(0·f)) = IDENTITY_ROT6D``, and
    subtracting ``identity_norm`` (= IDENTITY here) gives zero.
    """
    chunk = compute_action_chunk(
        VRControllerState(),
        num_frames=4,
        translate_scale=500.0,
        rotate_scale=1.0,
        psm1_rot6d_mean=_UNIT_MEAN,
        psm1_rot6d_std=_UNIT_STD,
        psm1_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
        psm2_rot6d_mean=_UNIT_MEAN,
        psm2_rot6d_std=_UNIT_STD,
        psm2_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
    )
    assert np.all(chunk[:, _PSM1_ROT6D_SLICE] == 0.0)
    assert np.all(chunk[:, _PSM2_ROT6D_SLICE] == 0.0)


def _expected_rot6d_ramp(*, drot: np.ndarray, rotate_scale: float, num_frames: int) -> np.ndarray:
    """Recompute the per-frame rot6d ramp the way ``compute_action_chunk`` should."""
    omega = np.array(
        [-drot[0], -drot[2], -drot[1]], dtype=np.float64
    ) * rotate_scale
    out = np.zeros((num_frames, 6), dtype=np.float32)
    for f in range(num_frames):
        R_f = rotvec_to_matrix(omega * float(f + 1))
        out[f] = (matrix_to_rot6d(R_f) - IDENTITY_ROT6D).astype(np.float32)
    return out


def test_compute_action_chunk_writes_psm1_rot6d_ramp_for_right_drot() -> None:
    drot = np.array([0.05, 0.0, 0.0], dtype=np.float64)
    rotate_scale = 2.0
    num_frames = 3
    chunk = compute_action_chunk(
        VRControllerState(right=_arm(drot=tuple(drot))),
        num_frames=num_frames,
        translate_scale=500.0,
        rotate_scale=rotate_scale,
        psm1_rot6d_mean=_UNIT_MEAN,
        psm1_rot6d_std=_UNIT_STD,
        psm1_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
        psm2_rot6d_mean=_UNIT_MEAN,
        psm2_rot6d_std=_UNIT_STD,
        psm2_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
    )
    expected = _expected_rot6d_ramp(
        drot=drot, rotate_scale=rotate_scale, num_frames=num_frames
    )
    assert np.allclose(chunk[:, _PSM1_ROT6D_SLICE], expected, atol=1e-5)
    # PSM2 rot6d untouched (left.drot = 0).
    assert np.all(chunk[:, _PSM2_ROT6D_SLICE] == 0.0)


def test_compute_action_chunk_writes_psm2_rot6d_ramp_for_left_drot() -> None:
    drot = np.array([0.0, 0.04, 0.0], dtype=np.float64)
    rotate_scale = 1.5
    num_frames = 5
    chunk = compute_action_chunk(
        VRControllerState(left=_arm(drot=tuple(drot))),
        num_frames=num_frames,
        translate_scale=500.0,
        rotate_scale=rotate_scale,
        psm1_rot6d_mean=_UNIT_MEAN,
        psm1_rot6d_std=_UNIT_STD,
        psm1_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
        psm2_rot6d_mean=_UNIT_MEAN,
        psm2_rot6d_std=_UNIT_STD,
        psm2_rot6d_identity_norm=_IDENTITY_ROT6D_NORM_UNIT,
    )
    expected = _expected_rot6d_ramp(
        drot=drot, rotate_scale=rotate_scale, num_frames=num_frames
    )
    assert np.allclose(chunk[:, _PSM2_ROT6D_SLICE], expected, atol=1e-5)
    # PSM1 rot6d untouched.
    assert np.all(chunk[:, _PSM1_ROT6D_SLICE] == 0.0)
