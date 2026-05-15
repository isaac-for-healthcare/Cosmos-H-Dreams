# Cosmos-H-Surgical-Simulator: Action Space for Open-H Multi-Embodiment Training

## JHU dVRK


| Indices | Size | Contents                         | Camera-frame side |
| ------- | ---- | -------------------------------- | ----------------- |
| 0-2     | 3    | PSM1 xyz_rel  (relative xyz)     | RIGHT arm  |
| 3-8     | 6    | PSM1 rot6d_rel (6D rel rotation) | RIGHT arm |
| 9       | 1    | PSM1 gripper  (absolute opening) | RIGHT arm |
| 10-12   | 3    | PSM2 xyz_rel                     | LEFT  arm |
| 13-18   | 6    | PSM2 rot6d_rel                   | LEFT  arm |
| 19      | 1    | PSM2 gripper                     | LEFT  arm |
| 20-43   | 24   | zero-padding to MAX_ACTION_DIM=44 |- |

Values are in mean-std normalised space (mean ≈ 0, std ≈ 1 per dim) — same space the model was trained in.

## Convention: chunk-anchor-relative (12-frame chunks)

- `xyz_rel`, `rot6d_rel`: displacement from the chunk's anchor pose. Anchor = the conditional first frame of each outer block in `run_cosmosh.py`; resets every 12 frames in lockstep with the runtime's outer-block boundary.
- `gripper`: absolute opening in normalised space — NOT a delta from anchor.

Source of truth — training-time stats script `compute_openh_action_stats.py` (cosmos-h-surgical-simulator-rt repo) calls `convert_to_hybrid_relative(action_data, eef_pose=ref_pose, ...)` once per sample window with `ref_pose = state[base_idx]` (state at window start). The whole `(T_action, 9)` stack is rebased against that single anchor.

Verified empirically on `sf_inference_data/260206/suturebot_inference_data/episode_001867_actions.npy`: PSM1-x row 11 = 1.388, row 12 = 0.275 — sawtooth at every 12th row, consistent with per-chunk anchor reset.

## Per-frame semantics

| dims | all-zero means | held key over one chunk (per-frame velocity `v`) |
| ---- | -------------- | ------------------------------------------------ |
| `xyz_rel`, `rot6d_rel` | "stay at chunk anchor" (frozen) | ramp `[1·v, 2·v, …, 12·v]`; resets to `1·v` next chunk |
| `gripper` | "average opening across training set" — NOT "hold current" | constant latched value; no ramp; idle must latch user's last setting |

A flat `[v, v, …, v]` within a chunk = "jump to +v and freeze" — wrong for continuous motion.

## Stats file

Use `sf_inference_data/260206/stats_cosmos.json`. Keys: `action.psm1_pose` (9), `action.psm1_gripper` (1), `action.psm2_pose` (9), `action.psm2_gripper` (1). The combined `"action"` block at the bottom is the 20-dim layout that matches the inference `.npy` exactly.

Do NOT use `suturebot_inference_stats.json` — it's 16-dim raw absolute (xyz + quat + gripper × 2 arms), i.e. pre-`convert_to_hybrid_relative`. Wrong representation for the keyboard layer.

The inference `.npy` is in **normalised** space: `value = (raw − stats.action.mean) / stats.action.std`. Verified by reverse-normalising episode 001867's PSM1 rot6d_rel mean `[0.31, 0.04, −0.19, −0.03, 0.35, −0.04]` back to identity `[1, 0, 0, 0, 1, 0]` using `stats_cosmos.action.psm1_pose` mean/std on dims 3-8. This is *why* the all-zero-vector freeze rule works: the unnormalised rel-action means are already at the freeze pose (xyz_rel ≈ 0, rot6d_rel ≈ identity), so normalised 0 ≈ freeze.

Physical scale per 1.0 stddev/frame:

| dim             | PSM1 std (m)            | PSM2 std (m)            |
| --------------- | ----------------------- | ----------------------- |
| xyz_rel x / y / z | 0.0023 / 0.0024 / 0.0020 | 0.0018 / 0.0012 / 0.0018 |

Gripper raw stats and endpoints in normalised space (norm = (raw − mean) / std):

| arm  | raw mean | raw std | closed (raw q01 → norm) | open (raw q99 → norm) |
| ---- | -------- | ------- | ----------------------- | --------------------- |
| PSM1 | −0.190   | 0.354   | −0.349 → **−0.45**      | +0.994 → **+3.34**    |
| PSM2 | +0.050   | 0.293   | −0.349 → **−1.36**      | +0.925 → **+2.99**    |

`rot6d_rel`: no clean stddev-to-angle conversion; tune `v` empirically.

Held-key starting point: `v ≈ 0.3 stddev/frame` on PSM1 xyz gives end-of-chunk reach ≈ 8 mm (≈ 7 mm/s at fps 10) — surgical-pace. PSM2 needs a larger `v` to match physically because its stds are smaller.

`stats_cosmos.json["timestep_interval"] = 3` — actions are subsampled every 3rd raw frame; raw demos are 30 fps, the action stream is 10 fps (matches `run_cosmosh.py --fps 10`). Only matters if driving the model off-default fps.