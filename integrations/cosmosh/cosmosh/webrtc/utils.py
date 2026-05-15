"""Shared low-level helpers used by both the keyboard and Quest control paths.

Anything that lives here must genuinely be used by both
:mod:`cosmosh.webrtc.controls` (keyboard) and
:mod:`cosmosh.webrtc.controls_quest` (Quest). Anything used by only one of
them belongs in that module instead.
"""

from __future__ import annotations

import numpy as np

# 20-dim normalised action layout, matching the combined ``action`` block of
# ``stats_cosmos.json``: [PSM1 xyz | PSM1 rot6d | PSM1 gripper | PSM2 xyz |
# PSM2 rot6d | PSM2 gripper]. Both control paths write into a chunk of this
# width.
ACTION_DIM_NORMALISED = 20

# Identity rot6d in column-major flatten layout. Used as the "no rotation" anchor that
# both control paths subtract from per-frame rot6d to make zero input → zero
# normalised output.
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def write_translate_ramp(
    chunk: np.ndarray,
    *,
    num_frames: int,
    v_xyz: np.ndarray,
    slice_start: int,
) -> None:
    """Write a per-frame translate ramp ``(f+1) · v_xyz`` into ``chunk``.

    Shared between keyboard (where ``v_xyz`` comes from a key-derived sum)
    and Quest (where it comes from ``dpos * translate_scale``). No-op for an
    all-zero ``v_xyz`` so an idle frame doesn't touch the chunk.
    """
    if not np.any(v_xyz):
        return
    for f in range(num_frames):
        chunk[f, slice_start : slice_start + 3] = (f + 1) * v_xyz


def rotvec_to_matrix(omega: np.ndarray) -> np.ndarray:
    """3D rotation vector (axis × angle) → 3×3 rotation matrix.

    For ``|omega|`` below a tiny threshold returns the identity matrix to
    avoid divide-by-zero.
    """
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    k = omega / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Column-major flatten of the first two columns: ``[R[:,0], R[:,1]]``."""
    return np.concatenate([R[:, 0], R[:, 1]])


def write_rotation_ramp(
    chunk: np.ndarray,
    *,
    num_frames: int,
    omega: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    identity_norm: np.ndarray,
    slice_start: int,
) -> None:
    """Write a per-frame rot6d ramp ``rot6d(exp((f+1)·ω))`` into ``chunk``.

    No-op for ``|ω| == 0``. Output is normalised by ``(mean, std)`` and the
    identity baseline ``identity_norm`` is subtracted so an all-zero
    rotation produces exact zeros in the slice. Shared between keyboard
    (where ``ω`` comes from a key-derived axis-angle sum) and Quest (where
    it comes from ``drot * rotate_scale``).
    """
    if np.linalg.norm(omega) <= 0.0:
        return
    for f in range(num_frames):
        R_f = rotvec_to_matrix(omega * float(f + 1))
        rot6d_f = matrix_to_rot6d(R_f)
        normalised = (rot6d_f - mean) / std
        chunk[f, slice_start : slice_start + 6] = (normalised - identity_norm).astype(
            np.float32
        )
