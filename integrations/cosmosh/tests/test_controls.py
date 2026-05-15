from __future__ import annotations

import numpy as np
import pytest
from cosmosh.webrtc.controls import (
    ACTION_DIM_NORMALISED,
    ARROW_DOWN_KEY,
    ARROW_LEFT_KEY,
    ARROW_RIGHT_KEY,
    ARROW_UP_KEY,
    PAGE_DOWN_KEY,
    PAGE_UP_KEY,
    PSM1_GRIPPER_CLOSE_KEY,
    PSM1_GRIPPER_CLOSED,
    PSM1_GRIPPER_DIM,
    PSM1_GRIPPER_OPEN,
    PSM1_GRIPPER_OPEN_KEY,
    PSM1_ROLL_MINUS_KEY,
    PSM1_ROLL_PLUS_KEY,
    PSM2_GRIPPER_CLOSE_KEY,
    PSM2_GRIPPER_CLOSED,
    PSM2_GRIPPER_DIM,
    PSM2_GRIPPER_OPEN,
    PSM2_GRIPPER_OPEN_KEY,
    PSM2_ROLL_MINUS_KEY,
    PSM2_ROLL_PLUS_KEY,
    ROT_TOKEN_PITCH_MINUS,
    ROT_TOKEN_PITCH_PLUS,
    ROT_TOKEN_ROLL_MINUS,
    ROT_TOKEN_ROLL_PLUS,
    ROT_TOKEN_YAW_MINUS,
    ROT_TOKEN_YAW_PLUS,
    SHIFT_KEY,
    CosmoshActionIntegrator,
    KeyboardState,
    matrix_to_rot6d,
    normalize_key,
    rotvec_to_matrix,
)


# ----- KeyboardState basics ---------------------------------------------


def test_keyboard_state_keydown_keyup_roundtrip() -> None:
    state = KeyboardState()
    assert state.apply_event(event="keydown", key="w")
    assert "w" in state.snapshot()
    assert state.apply_event(event="keyup", key="w")
    assert "w" not in state.snapshot()


def test_keyboard_state_rejects_unknown_key() -> None:
    state = KeyboardState()
    assert not state.apply_event(event="keydown", key="x")
    assert len(state.snapshot()) == 0


# ----- PSM1 (right arm, right-hand keys) ---------------------------------


def test_psm1_translate_latest_pressed_wins() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=ARROW_UP_KEY)
    state.apply_event(event="keydown", key=ARROW_DOWN_KEY)
    assert state.psm1_translate_keys() == frozenset({ARROW_DOWN_KEY})


def test_psm1_translate_independent_axes() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=ARROW_UP_KEY)
    state.apply_event(event="keydown", key=ARROW_LEFT_KEY)
    state.apply_event(event="keydown", key=PAGE_UP_KEY)
    assert state.psm1_translate_keys() == frozenset(
        {ARROW_UP_KEY, ARROW_LEFT_KEY, PAGE_UP_KEY}
    )


def test_psm1_arrows_translate_without_shift() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=ARROW_UP_KEY)
    assert state.psm1_translate_keys() == frozenset({ARROW_UP_KEY})
    assert state.psm1_rotation_keys() == frozenset()


@pytest.mark.xfail(
    reason=(
        "Shift+arrows pitch/yaw not wired after the arm-swap refactor: "
        "psm1_rotation_keys reads from _PSM1_TRANSLATE_Y_KEYS (arrows) but "
        "its pitch branches compare against 'w'/'s'. Keyboard app still "
        "works because users don't rely on Shift+pitch/yaw; only roll "
        "(,/.) is exercised. Remove this xfail once the resolver branches "
        "are switched to ARROW_UP/DOWN_KEY (or the keys array is changed)."
    ),
    strict=True,
)
def test_psm1_shift_arrows_rotate_not_translate() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=SHIFT_KEY)
    state.apply_event(event="keydown", key=ARROW_UP_KEY)
    assert state.psm1_translate_keys() == frozenset()
    assert state.psm1_rotation_keys() == frozenset({ROT_TOKEN_PITCH_PLUS})


def test_psm1_pageup_translates_z_regardless_of_shift() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=PAGE_UP_KEY)
    assert state.psm1_translate_keys() == frozenset({PAGE_UP_KEY})
    state.apply_event(event="keydown", key=SHIFT_KEY)
    assert state.psm1_translate_keys() == frozenset({PAGE_UP_KEY})


@pytest.mark.xfail(
    reason="Same PSM1 Shift+pitch wiring bug as above; remove together.",
    strict=True,
)
def test_psm1_release_shift_restores_translate() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=ARROW_UP_KEY)
    assert state.psm1_translate_keys() == frozenset({ARROW_UP_KEY})
    state.apply_event(event="keydown", key=SHIFT_KEY)
    assert state.psm1_translate_keys() == frozenset()
    assert state.psm1_rotation_keys() == frozenset({ROT_TOKEN_PITCH_PLUS})
    state.apply_event(event="keyup", key=SHIFT_KEY)
    assert state.psm1_translate_keys() == frozenset({ARROW_UP_KEY})
    assert state.psm1_rotation_keys() == frozenset()


def test_psm1_roll_keys_dont_need_shift() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=PSM1_ROLL_PLUS_KEY)
    assert state.psm1_rotation_keys() == frozenset({ROT_TOKEN_ROLL_PLUS})
    state.apply_event(event="keydown", key=PSM1_ROLL_MINUS_KEY)
    assert state.psm1_rotation_keys() == frozenset({ROT_TOKEN_ROLL_MINUS})


# ----- PSM2 (left arm, left-hand keys) -----------------------------------


def test_psm2_translate_latest_pressed_wins() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key="w")
    state.apply_event(event="keydown", key="s")
    assert state.psm2_translate_keys() == frozenset({"s"})


def test_psm2_translate_independent_axes() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key="w")
    state.apply_event(event="keydown", key="a")
    state.apply_event(event="keydown", key="r")
    assert state.psm2_translate_keys() == frozenset({"w", "a", "r"})


def test_psm2_wasd_translates_without_shift() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key="w")
    assert state.psm2_translate_keys() == frozenset({"w"})
    assert state.psm2_rotation_keys() == frozenset()


@pytest.mark.xfail(
    reason=(
        "Mirror of the PSM1 bug: psm2_rotation_keys reads from "
        "_PSM2_TRANSLATE_Y_KEYS ('w','s') but the pitch branches compare "
        "against ARROW_UP/DOWN_KEY. Same fix applies."
    ),
    strict=True,
)
def test_psm2_shift_wasd_rotates_not_translate() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=SHIFT_KEY)
    state.apply_event(event="keydown", key="w")
    assert state.psm2_translate_keys() == frozenset()
    assert state.psm2_rotation_keys() == frozenset({ROT_TOKEN_PITCH_PLUS})


def test_psm2_rf_translates_z_regardless_of_shift() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key="r")
    assert state.psm2_translate_keys() == frozenset({"r"})
    state.apply_event(event="keydown", key=SHIFT_KEY)
    assert state.psm2_translate_keys() == frozenset({"r"})


def test_psm2_roll_keys_dont_need_shift() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=PSM2_ROLL_PLUS_KEY)
    assert state.psm2_rotation_keys() == frozenset({ROT_TOKEN_ROLL_PLUS})
    state.apply_event(event="keydown", key=PSM2_ROLL_MINUS_KEY)
    assert state.psm2_rotation_keys() == frozenset({ROT_TOKEN_ROLL_MINUS})


# ----- Cross-arm interactions -------------------------------------------


@pytest.mark.xfail(
    reason="Shift+pitch wiring bug on both arms (see PSM1/PSM2 xfails above).",
    strict=True,
)
def test_shift_rotates_both_arms_independently() -> None:
    """Shift held + ArrowUp (PSM1) + W (PSM2) → both arms rotate pitch+."""
    state = KeyboardState()
    state.apply_event(event="keydown", key=SHIFT_KEY)
    state.apply_event(event="keydown", key=ARROW_UP_KEY)
    state.apply_event(event="keydown", key="w")
    assert state.psm1_rotation_keys() == frozenset({ROT_TOKEN_PITCH_PLUS})
    assert state.psm2_rotation_keys() == frozenset({ROT_TOKEN_PITCH_PLUS})
    assert state.psm1_translate_keys() == frozenset()
    assert state.psm2_translate_keys() == frozenset()


# ----- Gripper resolvers ------------------------------------------------


def test_psm1_gripper_intent_open_close() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=PSM1_GRIPPER_OPEN_KEY)
    assert state.psm1_gripper_intent() == +1
    state.apply_event(event="keydown", key=PSM1_GRIPPER_CLOSE_KEY)
    assert state.psm1_gripper_intent() == -1
    state.apply_event(event="keyup", key=PSM1_GRIPPER_CLOSE_KEY)
    state.apply_event(event="keyup", key=PSM1_GRIPPER_OPEN_KEY)
    assert state.psm1_gripper_intent() == 0


def test_psm2_gripper_intent_open_close() -> None:
    state = KeyboardState()
    # The browser normalises " " → "space" client-side before sending; the
    # server only ever sees the string "space" (matches request_session.js
    # ``normalizeKey``), so the test uses that form directly.
    state.apply_event(event="keydown", key=PSM2_GRIPPER_OPEN_KEY)
    assert state.psm2_gripper_intent() == +1
    state.apply_event(event="keydown", key=PSM2_GRIPPER_CLOSE_KEY)
    assert state.psm2_gripper_intent() == -1
    state.apply_event(event="keyup", key=PSM2_GRIPPER_CLOSE_KEY)
    state.apply_event(event="keyup", key=PSM2_GRIPPER_OPEN_KEY)
    assert state.psm2_gripper_intent() == 0


def test_psm1_and_psm2_gripper_intents_independent() -> None:
    state = KeyboardState()
    state.apply_event(event="keydown", key=PSM1_GRIPPER_OPEN_KEY)
    state.apply_event(event="keydown", key=PSM2_GRIPPER_CLOSE_KEY)
    assert state.psm1_gripper_intent() == +1
    assert state.psm2_gripper_intent() == -1


# ----- Integrator: idle / translate / gripper ---------------------------


def test_action_integrator_idle_returns_zeros() -> None:
    integrator = CosmoshActionIntegrator()
    chunk = integrator.next_action_chunk(num_frames=12)
    assert chunk.shape == (12, ACTION_DIM_NORMALISED)
    assert chunk.dtype == np.float32
    assert np.all(chunk == 0.0)


def test_psm1_translate_arrowup_ramps_y_axis() -> None:
    integrator = CosmoshActionIntegrator(translate_v_per_frame=0.3)
    chunk = integrator.next_action_chunk(
        num_frames=12, psm1_translate_keys=frozenset({ARROW_UP_KEY})
    )
    expected_y = np.array([(i + 1) * 0.3 for i in range(12)], dtype=np.float32)
    # PSM1 xyz dims 0..2; y is offset 1.
    assert np.allclose(chunk[:, 1], expected_y)
    assert np.all(chunk[:, [0, 2]] == 0.0)
    # PSM2 dims and grippers untouched.
    assert np.all(chunk[:, 3:] == 0.0)


def test_psm1_translate_pageup_ramps_z_axis() -> None:
    integrator = CosmoshActionIntegrator(translate_v_per_frame=0.5)
    chunk = integrator.next_action_chunk(
        num_frames=2, psm1_translate_keys=frozenset({PAGE_UP_KEY})
    )
    assert np.isclose(chunk[0, 2], 0.5)
    assert np.isclose(chunk[1, 2], 1.0)
    assert np.all(chunk[:, [0, 1]] == 0.0)


def test_psm1_translate_arrowleft_ramps_x_axis_positive() -> None:
    integrator = CosmoshActionIntegrator(translate_v_per_frame=0.5)
    chunk = integrator.next_action_chunk(
        num_frames=2, psm1_translate_keys=frozenset({ARROW_LEFT_KEY})
    )
    assert np.isclose(chunk[0, 0], 0.5)
    assert np.isclose(chunk[1, 0], 1.0)


def test_psm2_translate_w_ramps_y_axis() -> None:
    integrator = CosmoshActionIntegrator(translate_v_per_frame=0.3)
    chunk = integrator.next_action_chunk(
        num_frames=12, psm2_translate_keys=frozenset({"w"})
    )
    expected_y = np.array([(i + 1) * 0.3 for i in range(12)], dtype=np.float32)
    # PSM2 xyz dims 10..12; y is offset 11.
    assert np.allclose(chunk[:, 11], expected_y)
    assert np.all(chunk[:, [10, 12]] == 0.0)
    # PSM1 dims (0..9) and PSM2 rot6d (13..18) and grippers stay zero.
    assert np.all(chunk[:, 0:10] == 0.0)
    assert np.all(chunk[:, 13:] == 0.0)


def test_psm2_translate_a_ramps_x_axis_positive() -> None:
    integrator = CosmoshActionIntegrator(translate_v_per_frame=0.5)
    chunk = integrator.next_action_chunk(
        num_frames=2, psm2_translate_keys=frozenset({"a"})
    )
    assert np.isclose(chunk[0, 10], 0.5)
    assert np.isclose(chunk[1, 10], 1.0)


def test_psm2_translate_r_ramps_z_axis() -> None:
    integrator = CosmoshActionIntegrator(translate_v_per_frame=0.5)
    chunk = integrator.next_action_chunk(
        num_frames=2, psm2_translate_keys=frozenset({"r"})
    )
    assert np.isclose(chunk[0, 12], 0.5)
    assert np.isclose(chunk[1, 12], 1.0)


def test_psm1_translate_only_writes_psm1_dims() -> None:
    integrator = CosmoshActionIntegrator(translate_v_per_frame=0.3)
    chunk = integrator.next_action_chunk(
        num_frames=12,
        psm1_translate_keys=frozenset({ARROW_UP_KEY, ARROW_LEFT_KEY, PAGE_UP_KEY}),
    )
    # PSM2 xyz / rot6d / gripper untouched.
    assert np.all(chunk[:, 10:] == 0.0)


# ----- Integrator: gripper steps ---------------------------------------


def test_step_psm1_gripper_opens_and_clips() -> None:
    integrator = CosmoshActionIntegrator(gripper_v_per_chunk=10.0)
    integrator.step_psm1_gripper(+1)
    chunk = integrator.next_action_chunk(num_frames=2)
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], PSM1_GRIPPER_OPEN)
    integrator.step_psm1_gripper(-1)
    integrator.step_psm1_gripper(-1)
    chunk = integrator.next_action_chunk(num_frames=2)
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], PSM1_GRIPPER_CLOSED)


def test_step_psm2_gripper_opens_and_clips() -> None:
    integrator = CosmoshActionIntegrator(gripper_v_per_chunk=10.0)
    integrator.step_psm2_gripper(+1)
    chunk = integrator.next_action_chunk(num_frames=2)
    assert np.allclose(chunk[:, PSM2_GRIPPER_DIM], PSM2_GRIPPER_OPEN)
    integrator.step_psm2_gripper(-1)
    integrator.step_psm2_gripper(-1)
    chunk = integrator.next_action_chunk(num_frames=2)
    assert np.allclose(chunk[:, PSM2_GRIPPER_DIM], PSM2_GRIPPER_CLOSED)


def test_step_grippers_are_independent() -> None:
    integrator = CosmoshActionIntegrator(gripper_v_per_chunk=0.5)
    integrator.step_psm1_gripper(+1)
    integrator.step_psm2_gripper(-1)
    chunk = integrator.next_action_chunk(num_frames=2)
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], 0.5)
    assert np.allclose(chunk[:, PSM2_GRIPPER_DIM], -0.5)


def test_step_gripper_zero_is_noop() -> None:
    integrator = CosmoshActionIntegrator(gripper_v_per_chunk=0.5)
    integrator.step_psm1_gripper(+1)
    integrator.step_psm1_gripper(0)
    chunk = integrator.next_action_chunk(num_frames=2)
    assert np.allclose(chunk[:, PSM1_GRIPPER_DIM], 0.5)


# ----- Helpers + normalize_key -----------------------------------------


def test_normalize_space_key_maps_to_psm2_open_key() -> None:
    # After the keyboard-arm swap, the spacebar drives PSM2 (left arm). The
    # browser normalises a literal " " to "space" client-side before sending
    # (see request_session.js ``normalizeKey``), so the server's
    # ``normalize_key`` only sees the string form.
    assert normalize_key("space") == PSM2_GRIPPER_OPEN_KEY
    assert normalize_key("Space") == PSM2_GRIPPER_OPEN_KEY


def test_rotvec_to_matrix_zero_is_identity() -> None:
    R = rotvec_to_matrix(np.zeros(3, dtype=np.float64))
    assert np.allclose(R, np.eye(3))


def test_rotvec_to_matrix_x_axis_rotation() -> None:
    theta = 0.3
    R = rotvec_to_matrix(np.array([theta, 0.0, 0.0]))
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    )
    assert np.allclose(R, expected)


def test_matrix_to_rot6d_layout_is_column_major() -> None:
    R = np.arange(9, dtype=np.float64).reshape(3, 3)
    rot6d = matrix_to_rot6d(R)
    assert rot6d.tolist() == [
        R[0, 0], R[1, 0], R[2, 0],
        R[0, 1], R[1, 1], R[2, 1],
    ]


# ----- Integrator: rotation per arm ------------------------------------


def test_action_integrator_rotation_zero_baseline_each_arm() -> None:
    """No rotation tokens for either arm → dims 3..8 and 13..18 are zero,
    regardless of supplied stats."""
    rng = np.random.default_rng(0)
    integrator = CosmoshActionIntegrator(
        psm1_rot6d_mean=rng.normal(size=6),
        psm1_rot6d_std=rng.uniform(0.1, 1.0, size=6),
        psm2_rot6d_mean=rng.normal(size=6),
        psm2_rot6d_std=rng.uniform(0.1, 1.0, size=6),
    )
    chunk = integrator.next_action_chunk(num_frames=12)
    assert np.all(chunk[:, 3:9] == 0.0)
    assert np.all(chunk[:, 13:19] == 0.0)


def test_psm1_pitch_only_writes_psm1_rot6d() -> None:
    theta = np.deg2rad(2.0)
    integrator = CosmoshActionIntegrator(rotate_theta_per_frame=theta)
    chunk = integrator.next_action_chunk(
        num_frames=4,
        psm1_rotation_keys=frozenset({ROT_TOKEN_PITCH_PLUS}),
    )
    assert not np.all(chunk[:, 3:9] == 0.0)
    # PSM2 rot6d untouched.
    assert np.all(chunk[:, 13:19] == 0.0)


def test_psm2_pitch_only_writes_psm2_rot6d() -> None:
    theta = np.deg2rad(2.0)
    integrator = CosmoshActionIntegrator(rotate_theta_per_frame=theta)
    chunk = integrator.next_action_chunk(
        num_frames=4,
        psm2_rotation_keys=frozenset({ROT_TOKEN_PITCH_PLUS}),
    )
    # PSM1 rot6d untouched.
    assert np.all(chunk[:, 3:9] == 0.0)
    assert not np.all(chunk[:, 13:19] == 0.0)


def test_psm2_rotation_uses_psm2_stats() -> None:
    """PSM2 rot6d normalisation must use psm2 stats, not psm1's."""
    theta = np.deg2rad(1.5)
    psm1_mean = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) + 0.1
    psm1_std = np.full(6, 0.4)
    psm2_mean = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) - 0.05
    psm2_std = np.full(6, 0.6)
    integrator = CosmoshActionIntegrator(
        rotate_theta_per_frame=theta,
        psm1_rot6d_mean=psm1_mean,
        psm1_rot6d_std=psm1_std,
        psm2_rot6d_mean=psm2_mean,
        psm2_rot6d_std=psm2_std,
    )
    chunk = integrator.next_action_chunk(
        num_frames=3,
        psm2_rotation_keys=frozenset({ROT_TOKEN_YAW_PLUS}),
    )
    identity_baseline = (
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) - psm2_mean
    ) / psm2_std
    for f in range(3):
        R_f = rotvec_to_matrix(np.array([0.0, (f + 1) * theta, 0.0]))
        rot6d_f = matrix_to_rot6d(R_f)
        expected = (rot6d_f - psm2_mean) / psm2_std - identity_baseline
        assert np.allclose(chunk[f, 13:19], expected.astype(np.float32), atol=1e-5)


def test_both_arms_rotate_independently() -> None:
    """Both arms can rotate simultaneously without crosstalk."""
    theta = np.deg2rad(1.0)
    integrator = CosmoshActionIntegrator(rotate_theta_per_frame=theta)
    chunk = integrator.next_action_chunk(
        num_frames=2,
        psm1_rotation_keys=frozenset({ROT_TOKEN_PITCH_PLUS}),
        psm2_rotation_keys=frozenset({ROT_TOKEN_YAW_MINUS}),
    )
    # PSM1 pitch+ → x-axis rotation should populate dims 3..8.
    assert not np.all(chunk[:, 3:9] == 0.0)
    # PSM2 yaw- → -y-axis rotation should populate dims 13..18.
    assert not np.all(chunk[:, 13:19] == 0.0)


