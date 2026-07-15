<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `Cosmos-H-Dreams`

Package that wraps the Cosmos-H-Dreams action-conditioned Video2World model in a
streaming WebRTC server for live surgical simulation.

## What It Provides

- `GET /keyboard` — standalone keyboard viewer page (HTML/CSS/JS).
- `GET /quest` — Meta Quest WebXR viewer (HTTPS required).
- `POST /api/webrtc/offer` — SDP offer/answer signaling endpoint.
- Runtime/model preloading during server startup (CosmosH pipeline + Wan2.1
  VAE encoder + decoder + CR1 text embeddings + conditional first frame).
- A single active WebRTC session per server process.
- Continuous render loop on the server:
  1. Browser or headset sends action events over the DataChannel — these update server-side state.
  2. While the WebRTC connection is up, the server runs CosmosH outer blocks back-to-back, picking up the latest pressed-keys / gripper state at the start of each block.
  3. Each generated chunk is enqueued on the WebRTC track and announced via `chunk_done`. A small backpressure cap (~2 chunks) keeps the loop from running ahead of playback.

For full setup instructions (container, deps, config selection, offline inference) see [`GUIDE.md`](GUIDE.md).

## Run

All parameters live in YAML configs under `configs/`. CLI flags are limited
to `--config PATH` plus runtime knobs (`--host`, `--port`, `--debug`).

**Keyboard server:**

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server \
  --config cosmosHDreams/configs/keyboard_tabletop.yaml
```

**Quest server** (TLS required for WebXR — see [`GUIDE.md`](GUIDE.md) for cert setup):

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server_quest \
  --config cosmosHDreams/configs/quest_tabletop.yaml
```

**Unified server** (keyboard + Quest on one port, takeover semantics):

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server_unified \
  --config cosmosHDreams/configs/unified_tabletop.yaml
```

Then open:

- <http://0.0.0.0:8080/keyboard> — keyboard viewer (click **Connect session** to start)
- `https://<host-ip>:8443/quest` — Quest viewer (HTTPS)
- `/healthz` on either port reports `runtime_ready`

Override ports/hosts without editing YAML by passing `--host` / `--port` on the CLI.

> **First-input latency.** The DiT is compiled with `torch.compile` on the first
> forward pass triggered by user input. Expect the first generation to take
> significantly longer than steady-state; subsequent generations run at normal speed.

## Test

From repository root:

```bash
uv run --package flash-cosmosHDreams --extra dev pytest cosmosHDreams/tests
```

## Runtime Requirements

- CUDA-capable GPU with at least 12 GB of VRAM.
- `runtime.ckpt_path: null` in the config falls back to
  `AVAILABLE_COSMOSH_CHECKPOINT_PATHS["default"]`.
- `runtime.cr1_embeddings_path` — required; precomputed CR1 text embeddings `.pt`.
- `runtime.input_path` — required; accepts a video or a still image. For videos
  the conditional first frame is read at `runtime.start_frame_idx` (any
  mediapy-readable format); for images (`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`)
  the file is loaded directly and `start_frame_idx` is ignored.
- `runtime.stats_path` — required (`stats_cosmos.json`); supplies the dataset
  mean/std used to normalise rot6d output.

## DataChannel Message Format

**Browser → server:**

```json
{
  "type": "action",
  "action": {
    "event": "keydown",
    "key": "w"
  }
}
```

Supported message types:

- `{type: "action", action: {event: "keydown" | "keyup", key: ...}}` — held-key bookkeeping (`key` ∈ `w,a,s,d,r,f`) and `space` keydown to toggle the gripper.
- `{type: "reset"}` — clear keyboard state, integrator, AR index, and re-anchor on the initial conditional frame. Server pushes a fresh anchor frame to the video track and replies `{type: "reset_done"}`.
- `{type: "action", action: {event: "step"}}` — accepted as a no-op for backwards compatibility; the server renders continuously.

**Server → browser:**

```json
{
  "type": "chunk_done",
  "chunk_index": 3,
  "num_frames": 12,
  "enqueued_frames": 12
}
```

## Key Bindings

Key mapping (each arm uses the keyboard region under the matching hand):

**PSM1 (right arm, right-hand keys):**

| Key | Action |
|-----|--------|
| `↑` / `↓` | Translate +y / −y |
| `←` / `→` | Translate +x / −x |
| `PgUp` / `PgDn` | Translate +z / −z |
| `Shift + ↑/↓` | Pitch ± |
| `Shift + ←/→` | Yaw ± |
| `,` / `.` | Roll ± |
| `;` / `'` (held) | Open / close gripper progressively |

**PSM2 (left arm, left-hand keys):**

| Key | Action |
|-----|--------|
| `w` / `s` | Translate +y / −y |
| `a` / `d` | Translate +x / −x |
| `r` / `f` | Translate +z / −z |
| `Shift + w/s` | Pitch ± |
| `Shift + a/d` | Yaw ± |
| `q` / `e` | Roll ± |
| `space` / `c` (held) | Open / close gripper progressively |

Latest-pressed wins per axis when both keys of a pair are held. Shift gates
the WASD/arrow keys between translate and rotate. If multiple key events
arrive before the next chunk starts, the server aggregates them and applies
latest-pressed precedence per axis.

Keyboard translate sensitivity is `keyboard.translate_v_per_frame` in the
YAML (default `0.1` stddev/frame). The Quest equivalents are
`vr.translate_scale` and `vr.rotate_scale`.
