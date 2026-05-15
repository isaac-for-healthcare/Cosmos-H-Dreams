# `cosmosh`

Cosmosh integration package for `flashdreams` that exposes a minimal WebRTC
keyboard viewer for the surgical-simulator (CosmosH) recipe.

## What It Provides (Phase 1)

- `GET /request_session` serves a standalone viewer page (`HTML/CSS/JS`).
- `POST /api/webrtc/offer` performs SDP offer/answer signaling.
- Runtime/model preloading during server startup (CosmosH pipeline + Wan2.1
  VAE encoder + decoder + CR1 text embeddings + conditional first frame).
- A single active WebRTC session per server process.
- Continuous render loop on the server:
  1. browser sends `keydown` / `keyup` / `space` / `reset` events over DataChannel — these update server-side state,
  2. while the WebRTC connection is up, the server runs CosmosH outer blocks (12 actions → 12 generated frames) back-to-back, picking up the latest pressed-keys / gripper state at the start of each block,
  3. each generated chunk is enqueued on the WebRTC track and announced via `chunk_done`. A small backpressure cap (~2 chunks) keeps the loop from running ahead of playback.

Phase 1 is single-arm (PSM1) translate only. Rotation, gripper toggle, and
PSM2 are tracked in `PLAN.md` for later phases.

## Run

All experiment parameters live in YAML configs under `configs/`. CLI flags
are limited to `--config PATH` plus runtime knobs (`--host`, `--port`,
`--debug`). The schema is documented in `cosmosh/webrtc/config_loader.py`.

Keyboard server — suturebot episode 001867:

```bash
uv run --package flash-cosmosh python -m cosmosh.webrtc.server \
  --config integrations/cosmosh/configs/keyboard_episode_001867.yaml
```

Keyboard server — chole:

```bash
uv run --package flash-cosmosh python -m cosmosh.webrtc.server \
  --config integrations/cosmosh/configs/keyboard_chole.yaml
```

Quest server (pick a config; default port is 8443, TLS required for WebXR):

```bash
uv run --package flash-cosmosh python -m cosmosh.webrtc.server_quest \
  --config integrations/cosmosh/configs/quest_episode_001867.yaml
```

Then open:

- [http://localhost:8080/request_session](http://localhost:8080/request_session) — keyboard viewer
- `https://<workstation-ip>:8443/quest_session` — Quest viewer (HTTPS)
- `/healthz` on either port reports `runtime_ready`

Override ports/hosts without editing YAML by passing `--host` / `--port` on
the CLI.

## Test

From repository root:

```bash
uv run --package flash-cosmosh --extra dev pytest integrations/cosmosh/tests
```

## Runtime Requirements

- CUDA-capable GPU for CosmosH inference.
- `runtime.ckpt_path: null` in the config falls back to
  `AVAILABLE_COSMOSH_CHECKPOINT_PATHS["default"]`.
- `runtime.cr1_embeddings_path` is required (precomputed CR1 text
  embeddings `.pt`).
- `runtime.input_path` is required and accepts either a video or a still
  image. For videos the conditional first frame is read at
  `runtime.start_frame_idx` (any mediapy-readable format); for images
  (`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`) the file is loaded directly
  and `start_frame_idx` is ignored. The frame re-anchors per-block during
  the rollout.
- `runtime.stats_path` is required (`stats_cosmos.json`); supplies the
  dataset mean/std used to normalise rot6d output.

## DataChannel Message Format

Browser -> server:

```json
{
  "type": "action",
  "action": {
    "event": "keydown",
    "key": "w"
  }
}
```

- Supported message types:
  - `{type: "action", action: {event: "keydown" | "keyup", key: ...}}` — held-key bookkeeping (`key` ∈ `w,a,s,d,r,f`) and `space` keydown to toggle the gripper.
  - `{type: "reset"}` — clear keyboard state, integrator, AR index, and re-anchor on the initial conditional frame. Server drains pending video frames and replies `{type: "reset_done", dropped_frames: N}`. The render loop continues on the reset state.
  - `{type: "action", action: {event: "step"}}` — accepted as a no-op for backwards compatibility; the server renders continuously, so explicit step requests are unnecessary.
- Key mapping (each arm uses the keyboard region under the matching hand):
  - **PSM1 (right arm, right-hand keys):**
    - `↑/↓`: translate +y / −y
    - `←/→`: translate +x / −x
    - `PgUp/PgDn`: translate +z / −z
    - `Shift + ↑/↓`: pitch ± (Shift held → arrows rotate instead of translate)
    - `Shift + ←/→`: yaw ±
    - `,/.`: roll ± (no Shift required)
    - `;` / `'` (held): open / close gripper progressively, clipped to PSM1 endpoints.
  - **PSM2 (left arm, left-hand keys):**
    - `w/s`: translate +y / −y
    - `a/d`: translate +x / −x
    - `r/f`: translate +z / −z
    - `Shift + w/s`: pitch ±
    - `Shift + a/d`: yaw ±
    - `q/e`: roll ±
    - `space` / `c` (held): open / close gripper progressively, clipped to PSM2 endpoints.
  - Latest-pressed wins per axis when both keys of a pair are held; Shift gates the WASD/arrow keys between translate and rotate.
- If multiple key events arrive before the next chunk starts, the server
  aggregates them and applies latest-pressed precedence per axis.

Server -> browser:

```json
{
  "type": "chunk_done",
  "chunk_index": 3,
  "num_frames": 12,
  "enqueued_frames": 12
}
```

## Notes

- Camera-frame axis labels in the model state aren't documented; the
  key→dim assignment in
  `cosmosh/webrtc/controls.py::_PSM1_TRANSLATE_KEY_TO_DIM_AND_SIGN` was
  set empirically so that pressing W/A/R produces the expected physical
  motion. Within a held key, sign (positive/negative) is also empirical.
- Keyboard translate sensitivity is `keyboard.translate_v_per_frame` in
  the YAML (default `0.1` stddev/frame ≈ 2.3 mm/s on PSM1 at fps=10).
  Raise for a more responsive demo, lower for finer control. The Quest
  equivalents are `vr.translate_scale` and `vr.rotate_scale`.
