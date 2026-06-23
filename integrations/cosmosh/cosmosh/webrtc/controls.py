"""Keyboard input state + action-chunk computation for the keyboard server.

Quest-side state lives in :mod:`cosmosh.webrtc.controls_quest`; truly shared
helpers (action-width constant, translate-ramp writer) are in
:mod:`cosmosh.webrtc.utils`. Don't import Quest stuff from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cosmosh.webrtc.utils import (
    ACTION_DIM_NORMALISED,
    IDENTITY_ROT6D,
    matrix_to_rot6d,
    rotvec_to_matrix,
    write_rotation_ramp,
    write_translate_ramp,
)

PSM1_GRIPPER_DIM = 9
PSM2_GRIPPER_DIM = 19

# Fallback gripper endpoints in normalised space, from actions.md "Gripper
# raw stats and endpoints in normalised space" (the dVRK ``stats_cosmos.json``
# model). These are only used when the runtime has no loaded stats to derive
# from (e.g. a bare integrator in tests, or a stats file without gripper
# percentiles). At runtime the integrator's ``gripper_{open,closed}_psm{1,2}``
# fields are populated per-scene from the loaded stats' q01/q99 — see
# ``session._load_action_stats``. The OPEN values were empirically capped on
# the Quest 3 controller (stats q99 normalise to ≈3.3/3.0); stats-derived
# endpoints use the full range instead.
PSM1_GRIPPER_OPEN = 1.25  # 3.34
PSM1_GRIPPER_CLOSED = -0.45
PSM2_GRIPPER_OPEN = 1.25
PSM2_GRIPPER_CLOSED = -1.36

# ----- PSM1 (RIGHT arm) keys ---------------------------------------------
# Right-hand keyboard region: arrows + PgUp/PgDn translate, Shift+arrows
# pitch/yaw, ,/. roll, ;/' gripper. Right hand drives the arm visible on
# the right side of the camera frame (PSM1 = RIGHT per actions.md).
ARROW_UP_KEY = "arrowup"
ARROW_DOWN_KEY = "arrowdown"
ARROW_LEFT_KEY = "arrowleft"
ARROW_RIGHT_KEY = "arrowright"
PAGE_UP_KEY = "pageup"
PAGE_DOWN_KEY = "pagedown"
PSM1_TRANSLATE_KEYS = frozenset(
    {
        ARROW_UP_KEY,
        ARROW_DOWN_KEY,
        ARROW_LEFT_KEY,
        ARROW_RIGHT_KEY,
        PAGE_UP_KEY,
        PAGE_DOWN_KEY,
    }
)
_PSM1_TRANSLATE_Y_KEYS: tuple[str, ...] = (ARROW_UP_KEY, ARROW_DOWN_KEY)
_PSM1_TRANSLATE_X_KEYS: tuple[str, ...] = (ARROW_LEFT_KEY, ARROW_RIGHT_KEY)
_PSM1_TRANSLATE_Z_KEYS: tuple[str, ...] = (PAGE_UP_KEY, PAGE_DOWN_KEY)

PSM1_GRIPPER_OPEN_KEY = ";"
PSM1_GRIPPER_CLOSE_KEY = "'"
PSM1_GRIPPER_KEYS = frozenset({PSM1_GRIPPER_OPEN_KEY, PSM1_GRIPPER_CLOSE_KEY})

PSM1_ROLL_PLUS_KEY = ","
PSM1_ROLL_MINUS_KEY = "."
PSM1_ROTATION_KEYS = frozenset({PSM1_ROLL_PLUS_KEY, PSM1_ROLL_MINUS_KEY})

# ----- PSM2 (LEFT arm) keys ----------------------------------------------
# Left-hand keyboard region: WASD + RF translate, Shift+WASD pitch/yaw, Q/E
# roll, Space/C gripper. Left hand drives the arm visible on the left side
# of the camera frame (PSM2 = LEFT per actions.md).
PSM2_TRANSLATE_KEYS = frozenset({"w", "a", "s", "d", "r", "f"})
_PSM2_TRANSLATE_Y_KEYS: tuple[str, ...] = ("w", "s")
_PSM2_TRANSLATE_X_KEYS: tuple[str, ...] = ("a", "d")
_PSM2_TRANSLATE_Z_KEYS: tuple[str, ...] = ("r", "f")

# Browser ``KeyboardEvent.key`` for spacebar is a literal " "; we normalise
# to ``"space"`` on both sides.
PSM2_GRIPPER_OPEN_KEY = "space"
PSM2_GRIPPER_CLOSE_KEY = "c"
PSM2_GRIPPER_KEYS = frozenset({PSM2_GRIPPER_OPEN_KEY, PSM2_GRIPPER_CLOSE_KEY})

PSM2_ROLL_PLUS_KEY = "q"
PSM2_ROLL_MINUS_KEY = "e"
PSM2_ROTATION_KEYS = frozenset({PSM2_ROLL_PLUS_KEY, PSM2_ROLL_MINUS_KEY})

# Shift gates WASD ↔ pitch/yaw for PSM1 and arrows ↔ pitch/yaw for PSM2.
# Both arms share the single Shift modifier; that's fine because Shift+WASD
# vs Shift+Arrow are distinguishable by the non-Shift key.
SHIFT_KEY = "shift"

# Resolved tokens emitted by the rotation resolvers (shared across arms;
# the integrator routes each arm's tokens to its own rot6d slice).
ROT_TOKEN_PITCH_PLUS = "pitch+"
ROT_TOKEN_PITCH_MINUS = "pitch-"
ROT_TOKEN_YAW_PLUS = "yaw+"
ROT_TOKEN_YAW_MINUS = "yaw-"
ROT_TOKEN_ROLL_PLUS = "roll+"
ROT_TOKEN_ROLL_MINUS = "roll-"

# Per-axis unit vectors. Tentative — axis labels and signs are unverified
# against the trained model; expect to flip empirically per arm.
_PITCH_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_YAW_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_ROLL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

_ROTATION_TOKEN_TO_AXIS: dict[str, np.ndarray] = {
    ROT_TOKEN_PITCH_PLUS: +_PITCH_AXIS,
    ROT_TOKEN_PITCH_MINUS: -_PITCH_AXIS,
    ROT_TOKEN_YAW_PLUS: +_YAW_AXIS,
    ROT_TOKEN_YAW_MINUS: -_YAW_AXIS,
    ROT_TOKEN_ROLL_PLUS: +_ROLL_AXIS,
    ROT_TOKEN_ROLL_MINUS: -_ROLL_AXIS,
}

# Union accepted by ``KeyboardState.apply_event``.
SUPPORTED_KEYS = (
    PSM1_TRANSLATE_KEYS
    | PSM1_GRIPPER_KEYS
    | PSM1_ROTATION_KEYS
    | PSM2_TRANSLATE_KEYS
    | PSM2_GRIPPER_KEYS
    | PSM2_ROTATION_KEYS
    | frozenset({SHIFT_KEY})
)

# PSM1 / PSM2 translate dim mapping (relative within each arm's xyz block).
# Held over a 12-frame chunk this becomes a linear ramp [1·v, 2·v, …, 12·v].
# PSM1 (right hand) is the starting guess — signs/dims are tentative and
# expected to need an empirical pass.
_PSM1_TRANSLATE_KEY_TO_DIM_AND_SIGN: dict[str, tuple[int, float]] = {
    ARROW_UP_KEY: (1, +1.0),
    ARROW_DOWN_KEY: (1, -1.0),
    ARROW_LEFT_KEY: (0, +1.0),
    ARROW_RIGHT_KEY: (0, -1.0),
    PAGE_UP_KEY: (2, +1.0),
    PAGE_DOWN_KEY: (2, -1.0),
}
# PSM2 (left hand) inherits the empirical PSM1-translate mapping that was
# verified before the keyboard-arm swap.
_PSM2_TRANSLATE_KEY_TO_DIM_AND_SIGN: dict[str, tuple[int, float]] = {
    "w": (1, +1.0),
    "s": (1, -1.0),
    "a": (0, +1.0),
    "d": (0, -1.0),
    "r": (2, +1.0),
    "f": (2, -1.0),
}


def normalize_key(key: str) -> str:
    raw = key.lower()
    # ``KeyboardEvent.key`` for spacebar is " "; map to "space" before
    # strip() collapses it to "".
    if raw == " ":
        return PSM2_GRIPPER_OPEN_KEY
    return raw.strip()


@dataclass(slots=True)
class KeyboardState:
    """Tracks held keys with latest-pressed precedence per component."""

    pressed_keys: set[str] = field(default_factory=set)
    _press_order: dict[str, int] = field(default_factory=dict)
    _press_counter: int = 0

    def apply_event(self, *, event: str, key: str) -> bool:
        normalized_key = normalize_key(key)
        if normalized_key not in SUPPORTED_KEYS:
            return False

        normalized_event = event.strip().lower()
        if normalized_event == "keydown":
            self.pressed_keys.add(normalized_key)
            self._press_counter += 1
            self._press_order[normalized_key] = self._press_counter
            return True
        if normalized_event == "keyup":
            self.pressed_keys.discard(normalized_key)
            self._press_order.pop(normalized_key, None)
            return True
        return False

    def snapshot(self) -> frozenset[str]:
        return frozenset(self.pressed_keys)

    def _latest_pressed(self, keys: tuple[str, ...]) -> str | None:
        latest_key: str | None = None
        latest_idx = -1
        for key in keys:
            if key not in self.pressed_keys:
                continue
            idx = self._press_order.get(key, -1)
            if idx >= latest_idx:
                latest_idx = idx
                latest_key = key
        return latest_key

    # --- PSM1 resolvers -------------------------------------------------

    def psm1_translate_keys(self) -> frozenset[str]:
        """PSM1 translate intent. Arrows gated on Shift not held; PgUp/PgDn always."""
        effective: set[str] = set()
        if SHIFT_KEY not in self.pressed_keys:
            for key in (
                self._latest_pressed(_PSM1_TRANSLATE_Y_KEYS),
                self._latest_pressed(_PSM1_TRANSLATE_X_KEYS),
            ):
                if key is not None:
                    effective.add(key)
        z_key = self._latest_pressed(_PSM1_TRANSLATE_Z_KEYS)
        if z_key is not None:
            effective.add(z_key)
        return frozenset(effective)

    def psm1_rotation_keys(self) -> frozenset[str]:
        """PSM1 rotation tokens. Shift+Arrows = pitch/yaw, ,/. = roll."""
        effective: set[str] = set()
        if SHIFT_KEY in self.pressed_keys:
            pitch = self._latest_pressed(_PSM1_TRANSLATE_Y_KEYS)
            if pitch == ARROW_UP_KEY:
                effective.add(ROT_TOKEN_PITCH_PLUS)
            elif pitch == ARROW_DOWN_KEY:
                effective.add(ROT_TOKEN_PITCH_MINUS)
            yaw = self._latest_pressed(_PSM1_TRANSLATE_X_KEYS)
            if yaw == ARROW_LEFT_KEY:
                effective.add(ROT_TOKEN_YAW_PLUS)
            elif yaw == ARROW_RIGHT_KEY:
                effective.add(ROT_TOKEN_YAW_MINUS)
        roll = self._latest_pressed((PSM1_ROLL_PLUS_KEY, PSM1_ROLL_MINUS_KEY))
        if roll == PSM1_ROLL_PLUS_KEY:
            effective.add(ROT_TOKEN_ROLL_PLUS)
        elif roll == PSM1_ROLL_MINUS_KEY:
            effective.add(ROT_TOKEN_ROLL_MINUS)
        return frozenset(effective)

    def psm1_gripper_intent(self) -> int:
        """+1 for PSM1 open key held, -1 for close, 0 for neither."""
        latest = self._latest_pressed((PSM1_GRIPPER_OPEN_KEY, PSM1_GRIPPER_CLOSE_KEY))
        if latest == PSM1_GRIPPER_OPEN_KEY:
            return +1
        if latest == PSM1_GRIPPER_CLOSE_KEY:
            return -1
        return 0

    # --- PSM2 resolvers -------------------------------------------------

    def psm2_translate_keys(self) -> frozenset[str]:
        """PSM2 translate intent. WASD gated on Shift not held; R/F always."""
        effective: set[str] = set()
        if SHIFT_KEY not in self.pressed_keys:
            for key in (
                self._latest_pressed(_PSM2_TRANSLATE_Y_KEYS),
                self._latest_pressed(_PSM2_TRANSLATE_X_KEYS),
            ):
                if key is not None:
                    effective.add(key)
        z_key = self._latest_pressed(_PSM2_TRANSLATE_Z_KEYS)
        if z_key is not None:
            effective.add(z_key)
        return frozenset(effective)

    def psm2_rotation_keys(self) -> frozenset[str]:
        """PSM2 rotation tokens. Shift+WASD = pitch/yaw, Q/E = roll."""
        effective: set[str] = set()
        if SHIFT_KEY in self.pressed_keys:
            pitch = self._latest_pressed(_PSM2_TRANSLATE_Y_KEYS)
            if pitch == "w":
                effective.add(ROT_TOKEN_PITCH_PLUS)
            elif pitch == "s":
                effective.add(ROT_TOKEN_PITCH_MINUS)
            yaw = self._latest_pressed(_PSM2_TRANSLATE_X_KEYS)
            if yaw == "a":
                effective.add(ROT_TOKEN_YAW_PLUS)
            elif yaw == "d":
                effective.add(ROT_TOKEN_YAW_MINUS)
        roll = self._latest_pressed((PSM2_ROLL_PLUS_KEY, PSM2_ROLL_MINUS_KEY))
        if roll == PSM2_ROLL_PLUS_KEY:
            effective.add(ROT_TOKEN_ROLL_PLUS)
        elif roll == PSM2_ROLL_MINUS_KEY:
            effective.add(ROT_TOKEN_ROLL_MINUS)
        return frozenset(effective)

    def psm2_gripper_intent(self) -> int:
        """+1 for PSM2 open key held, -1 for close, 0 for neither."""
        latest = self._latest_pressed((PSM2_GRIPPER_OPEN_KEY, PSM2_GRIPPER_CLOSE_KEY))
        if latest == PSM2_GRIPPER_OPEN_KEY:
            return +1
        if latest == PSM2_GRIPPER_CLOSE_KEY:
            return -1
        return 0


_DEFAULT_ROTATE_THETA_PER_FRAME: float = float(np.deg2rad(1.0))


@dataclass(slots=True)
class CosmoshActionIntegrator:
    """Maps current pressed keys to a per-chunk action tensor in normalised space.

    Output shape: ``(num_frames, 20)`` float32. The 20-dim layout matches
    ``stats_cosmos.json["action"]`` (PSM1 xyz/rot6d/gripper + PSM2 xyz/rot6d/gripper).
    Writes per-frame translate ramps + rot6d (axis-angle ω · f → R_f) for both
    arms + latched grippers.
    """

    # Per-frame translate velocity in normalised (stddev) units. 0.3 ≈ 7 mm/s
    # at fps=10 for PSM1 — surgical-pace per actions.md. PSM2 uses the same
    # normalised v; physical mm/s differs because PSM2 stds are smaller.
    translate_v_per_frame: float = 0.3
    # Per-chunk gripper increment in normalised stddev units. With the default
    # 1.0 it takes ~4 chunks to traverse the full PSM1 [CLOSED, OPEN] range.
    gripper_v_per_chunk: float = 1.0
    # Per-frame rotation magnitude (radians). 1° ≈ 1.7e-2 rad — small enough
    # that the non-commutative-multi-axis approximation (sum of axis-angle
    # vectors) is reasonable.
    rotate_theta_per_frame: float = _DEFAULT_ROTATE_THETA_PER_FRAME
    # rot6d normalisation stats. Defaults to identity-mean / unit-std so a
    # Phase-1 instance without rotation stats still produces zero-rot6d
    # output for empty rotation_keys (see __post_init__).
    psm1_rot6d_mean: np.ndarray = field(default_factory=lambda: IDENTITY_ROT6D.copy())
    psm1_rot6d_std: np.ndarray = field(
        default_factory=lambda: np.ones(6, dtype=np.float64)
    )
    psm2_rot6d_mean: np.ndarray = field(default_factory=lambda: IDENTITY_ROT6D.copy())
    psm2_rot6d_std: np.ndarray = field(
        default_factory=lambda: np.ones(6, dtype=np.float64)
    )
    # Latched gripper values in normalised space. Both arms start at the
    # dataset mean (≈ 0); held-key steps move toward the OPEN/CLOSED endpoints.
    latched_gripper_psm1: float = 0.0
    latched_gripper_psm2: float = 0.0
    # Per-arm gripper clip endpoints in normalised space. Default to the dVRK
    # module constants; the runtime overrides them per-scene with values
    # derived from the loaded stats (q01/q99 → normalised). ``closed < open``
    # is enforced in ``__post_init__``.
    gripper_open_psm1: float = PSM1_GRIPPER_OPEN
    gripper_closed_psm1: float = PSM1_GRIPPER_CLOSED
    gripper_open_psm2: float = PSM2_GRIPPER_OPEN
    gripper_closed_psm2: float = PSM2_GRIPPER_CLOSED

    # Pre-computed normalised baselines for identity rotation (one per arm).
    # Subtracting these makes "no rotation key held" yield exact zero in the
    # rot6d slices regardless of the supplied stats.
    _psm1_identity_rot6d_norm: np.ndarray = field(init=False)
    _psm2_identity_rot6d_norm: np.ndarray = field(init=False)
    # Tracked arm positions in normalised action space. Updated each chunk so
    # key-release writes the last position instead of zeros (which the model
    # interprets as "arm at dataset-mean position" and generates a visual snap
    # back to the initial frame).
    _psm1_pos: np.ndarray = field(init=False)
    _psm2_pos: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        for label, arr in (
            ("psm1_rot6d_mean", self.psm1_rot6d_mean),
            ("psm1_rot6d_std", self.psm1_rot6d_std),
            ("psm2_rot6d_mean", self.psm2_rot6d_mean),
            ("psm2_rot6d_std", self.psm2_rot6d_std),
        ):
            if arr.shape != (6,):
                raise ValueError(f"{label} must have shape (6,); got {arr.shape}.")
        for arm, closed, open_ in (
            ("psm1", self.gripper_closed_psm1, self.gripper_open_psm1),
            ("psm2", self.gripper_closed_psm2, self.gripper_open_psm2),
        ):
            if not closed < open_:
                raise ValueError(
                    f"{arm} gripper endpoints must satisfy closed < open; "
                    f"got closed={closed}, open={open_}."
                )
        self._psm1_identity_rot6d_norm = (
            IDENTITY_ROT6D - self.psm1_rot6d_mean
        ) / self.psm1_rot6d_std
        self._psm2_identity_rot6d_norm = (
            IDENTITY_ROT6D - self.psm2_rot6d_mean
        ) / self.psm2_rot6d_std
        self._psm1_pos = np.zeros(3, dtype=np.float64)
        self._psm2_pos = np.zeros(3, dtype=np.float64)

    def step_psm1_gripper(self, direction: int) -> None:
        """Apply one chunk's worth of PSM1 gripper motion in ``direction``."""
        if direction == 0:
            return
        delta = float(direction) * self.gripper_v_per_chunk
        self.latched_gripper_psm1 = float(
            np.clip(
                self.latched_gripper_psm1 + delta,
                self.gripper_closed_psm1,
                self.gripper_open_psm1,
            )
        )

    def step_psm2_gripper(self, direction: int) -> None:
        """Apply one chunk's worth of PSM2 gripper motion in ``direction``."""
        if direction == 0:
            return
        delta = float(direction) * self.gripper_v_per_chunk
        self.latched_gripper_psm2 = float(
            np.clip(
                self.latched_gripper_psm2 + delta,
                self.gripper_closed_psm2,
                self.gripper_open_psm2,
            )
        )

    def next_action_chunk(
        self,
        *,
        num_frames: int = 12,
        psm1_translate_keys: frozenset[str] = frozenset(),
        psm1_rotation_keys: frozenset[str] = frozenset(),
        psm2_translate_keys: frozenset[str] = frozenset(),
        psm2_rotation_keys: frozenset[str] = frozenset(),
    ) -> np.ndarray:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        chunk = np.zeros((num_frames, ACTION_DIM_NORMALISED), dtype=np.float32)

        self._write_translate(
            chunk,
            num_frames=num_frames,
            keys=psm1_translate_keys,
            mapping=_PSM1_TRANSLATE_KEY_TO_DIM_AND_SIGN,
            slice_start=0,
            pos=self._psm1_pos,
        )
        self._write_translate(
            chunk,
            num_frames=num_frames,
            keys=psm2_translate_keys,
            mapping=_PSM2_TRANSLATE_KEY_TO_DIM_AND_SIGN,
            slice_start=10,
            pos=self._psm2_pos,
        )

        self._write_rotation(
            chunk,
            num_frames=num_frames,
            tokens=psm1_rotation_keys,
            mean=self.psm1_rot6d_mean,
            std=self.psm1_rot6d_std,
            identity_norm=self._psm1_identity_rot6d_norm,
            slice_start=3,
        )
        self._write_rotation(
            chunk,
            num_frames=num_frames,
            tokens=psm2_rotation_keys,
            mean=self.psm2_rot6d_mean,
            std=self.psm2_rot6d_std,
            identity_norm=self._psm2_identity_rot6d_norm,
            slice_start=13,
        )

        # Grippers: constant latched value across the chunk (per actions.md).
        chunk[:, PSM1_GRIPPER_DIM] = self.latched_gripper_psm1
        chunk[:, PSM2_GRIPPER_DIM] = self.latched_gripper_psm2

        return chunk

    def _write_translate(
        self,
        chunk: np.ndarray,
        *,
        num_frames: int,
        keys: frozenset[str],
        mapping: dict[str, tuple[int, float]],
        slice_start: int,
        pos: np.ndarray,
    ) -> None:
        """Write translate slice and update tracked position in place.

        ``pos`` is mutated: after the call it holds the arm position at the
        end of this chunk so the next chunk can start from there.
        """
        v_xyz = np.zeros(3, dtype=np.float64)
        for key in keys:
            if key not in mapping:
                continue
            dim, sign = mapping[key]
            v_xyz[dim] += sign * self.translate_v_per_frame
        write_translate_ramp(
            chunk, num_frames=num_frames, v_xyz=v_xyz, slice_start=slice_start,
            start_pos=pos.copy(),
        )
        pos[:] += num_frames * v_xyz

    def _write_rotation(
        self,
        chunk: np.ndarray,
        *,
        num_frames: int,
        tokens: frozenset[str],
        mean: np.ndarray,
        std: np.ndarray,
        identity_norm: np.ndarray,
        slice_start: int,
    ) -> None:
        omega = np.zeros(3, dtype=np.float64)
        for token in tokens:
            axis = _ROTATION_TOKEN_TO_AXIS.get(token)
            if axis is None:
                continue
            omega += axis * self.rotate_theta_per_frame
        write_rotation_ramp(
            chunk,
            num_frames=num_frames,
            omega=omega,
            mean=mean,
            std=std,
            identity_norm=identity_norm,
            slice_start=slice_start,
        )
