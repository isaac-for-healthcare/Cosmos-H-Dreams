<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running Cosmos-H-Dreams inference

This guide walks through running the CosmosH action-conditioned streaming
Video2World recipe end to end: build a container, drop into it, and run
one of the bundled configurations.

The recipe takes a Cosmos-H-Dreams autoregressive checkpoint, conditions it on a
first frame + a precomputed CR1 text embedding, and rolls it forward
in blocks of generated frames.

The guide has two parts after the shared setup:

- **Common setup** (sections 1–4) — build the container, mount your
  assets, sync deps, pick one of the bundled configurations.
- **Mode A — Offline batch inference** (sections 5–6) — feed a JSON
  manifest of `{input_video, input_action, output_video}` entries and
  drive the recipe with action `.npy` trajectories. Outputs are MP4 +
  raw-tensor files. Reach for this when you want reproducible rollouts
  or benchmark numbers.
- **Mode B — Interactive WebRTC** (section 7) — start the WebRTC server
  and drive the rollout live from either a keyboard (browser) or a Meta
  Quest headset (WebXR), or both at once through the unified server. No
  action `.npy` needed; the browser / headset is the input device.

Both modes share the same recipe runtime — the same checkpoint, the
same outer-block render loop, the same first-block warmup. Pick the
mode that matches what you want to do and skip past the other.

---

## Common setup (sections 1–4)

These four steps are identical regardless of which mode you intend to
run.

## 1. Build the container

The container recipe lives in `docker/`. It pins CUDA, cuDNN, Python,
and every dep needed for inference; you should not need to install
anything on the host.

From the repo root:

```bash
docker build -t cosmos-h-dreams:latest docker/
```

---

## 2. Start the container

Download the checkpoints from [Cosmos-H-Dreams model repo](https://huggingface.co/nvidia/Cosmos-H-Dreams) in HF, then place them under the repo root before launching:

- `checkpoints/` — the CosmosH `.pt` checkpoint(s)

(The exact names aren't enforced — keep them anywhere under the repo
root; the CLI paths below assume these.)

```bash
docker run --rm -it \
  --network host \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v .:/workspace/cosmos-h-dreams \
  -w /workspace/cosmos-h-dreams \
  cosmos-h-dreams:latest /bin/bash
```

All commands below are run from the repo root.

---

## 3. Sync dependencies

The first time you run anything in a fresh container, sync the project
extras so the runners' lazy imports (mediapy, opencv, …) are available:

```bash
uv sync --extra dev --extra runners --group lint
```

`uv` auto-activates the project venv; prefix every command with
`uv run` and you do not need to source anything.

---

## 4. Pick a configuration

Configurations follow the pattern `cosmosHDreams-chunk{N}-[2steps-]{encoder}-{decoder}`.
The `chunk{N}` prefix is the number of latent frames per DiT forward pass;
`chunk3` is recommended for throughput. Omitting the step count (i.e. no
`2steps`) defaults to 4-step, which is also available as an explicit
`4steps` slug.

| Config name | Chunk | Schedule | Encoder | Decoder | Notes |
|---|---|---|---|---|---|
| `cosmosHDreams-chunk3-vae-vae` ⭐ | 3 | 4-step | full Wan2.1 VAE | full Wan2.1 VAE | Recommended default. |
| `cosmosHDreams-chunk3-4steps-vae-vae` | 3 | 4-step | full Wan2.1 VAE | full Wan2.1 VAE | Explicit 4-step alias. |
| `cosmosHDreams-chunk3-vae-lighttae` | 3 | 4-step | full Wan2.1 VAE | TAEHV `lighttae` | Faster decode, modest quality drop. |
| `cosmosHDreams-chunk3-2steps-vae-vae` | 3 | 2-step | full Wan2.1 VAE | full Wan2.1 VAE | ~2× DiT speedup. |
| `cosmosHDreams-chunk3-2steps-vae-lighttae` | 3 | 2-step | full Wan2.1 VAE | TAEHV `lighttae` | Fastest overall. |
| `cosmosHDreams-chunk2-vae-vae` | 2 | 4-step | full Wan2.1 VAE | full Wan2.1 VAE | Smaller chunk; more DiT calls. |
| `cosmosHDreams-chunk2-vae-lighttae` | 2 | 4-step | full Wan2.1 VAE | TAEHV `lighttae` | |
| `cosmosHDreams-chunk2-2steps-vae-vae` | 2 | 2-step | full Wan2.1 VAE | full Wan2.1 VAE | |
| `cosmosHDreams-chunk2-2steps-vae-lighttae` | 2 | 2-step | full Wan2.1 VAE | TAEHV `lighttae` | |

(Plus explicit `4steps` aliases for every non-`2steps` config above — 12 configs total.)

All configs share the same action-conditioned streaming loop. The only knobs the config picks are
the chunk size, the denoising schedule length, and the VAE pair.

A few rules of thumb:

- Start with `cosmosHDreams-chunk3-vae-vae` to validate the integration,
  then move toward `lighttae` decoders for throughput once outputs look right.
- The 4-step variants are closer to the reference upstream output; the
  2-step variants trade some fidelity for ~2× DiT speedup.
- `lighttae` decoders are roughly 10× faster than the full Wan decoder.

To see every runner the CLI knows about, run `uv run flashdreams-run --help`.

---

## Mode A — Offline batch inference (sections 5–6)

You'll need a JSON manifest of entries, each pointing at an input
video and an action `.npy`. Outputs are written to disk as MP4 +
tensor files. Skip this part if you only want to drive the recipe
interactively — jump to [section 7](#7-mode-b--interactive-webrtc-keyboard--quest).

## 5. Run a configuration

Minimum required inputs:

- **`--input-json`** — a JSON list of `{"input_video": ..., "input_action": ..., "output_video": ...}` entries. `start_frame_idx` and `resolution` may be set per-entry.
- **`--cr1-embeddings-path`** — a precomputed CR1 (Cosmos-Reason1) text-embedding `.pt`. CosmosH does not run the text encoder online; it expects a one-shot embedding.

Optional:

- **`--root-dir`** — prefixed to every `input_video` / `input_action` path in the JSON.
- **`--total-blocks`** — how many outer blocks to roll. Each block consumes one chunk worth of action steps; the loop exits early if the action trajectory is shorter.
- **`--start-frame-idx`** — default conditional-frame index when the JSON doesn't specify it.
- **`--save-comparison True`** — also write `*_comparison.mp4` (input on the left, prediction on the right).
- **`--fps`** — output MP4 frame rate (default `10`).
- **`--output-dir`** — where outputs land. Per-entry `output_video` paths are anchored under this when relative; absolute paths in the JSON are honored as-is.
- **`--resolution H,W`** — override the bundle's default output resolution (default `288,512`). Both dimensions must be multiples of 8 (the Wan2.1 VAE spatial compression ratio).

Example — full Wan VAE on both sides, comparison MP4 on, 4 diffusion steps, override the
checkpoint to a domain-specific one:

```bash
uv run flashdreams-run cosmosHDreams-chunk3-vae-vae \
  --input-json assets/example_data/offline/suturebot_inference_manifest.json \
  --cr1-embeddings-path checkpoints/cr1_empty_string_text_embeddings.pt \
  --root-dir . \
  --total-blocks 20 \
  --save-comparison True \
  --pipeline.diffusion-model.transformer.checkpoint-path checkpoints/model_ema_bf16_jhutabletop_288x512.pt
```

To switch configuration, swap the first positional. Everything else can stay the same:

```bash
uv run flashdreams-run cosmosHDreams-chunk3-2steps-vae-lighttae \
  --input-json ... \
  --cr1-embeddings-path ... \
  --total-blocks 20
```

> **Note — first-run latency.** The very first block of the very first
> run is significantly slower than later ones: the DiT is compiled
> with `torch.compile` and the CUDA graphs are captured on the first
> forward pass. The runner reports this block as `[WARMUP]` in the
> per-block log. The compiled artifacts are cached by PyTorch's
> Inductor cache and reused on subsequent runs of the same
> configuration / resolution / GPU, so repeated invocations of the
> same `cosmosHDreams-*` config start much faster — only the CUDA-graph
> capture (~a few seconds) runs again. Changing `--resolution`, the
> checkpoint, or the config forces a recompile.

### Overriding pinned knobs

Any field on the pipeline / DiT / scheduler is reachable through tyro's
nested-flag syntax (`--pipeline.<path>.<field> VALUE`). The most common
overrides:

| Knob | Flag |
|---|---|
| Checkpoint path | `--pipeline.diffusion-model.transformer.checkpoint-path /path/to/ckpt.pt` |
| RNG seed | `--pipeline.diffusion-model.seed N` |
| KV-cache window (latent frames) | `--pipeline.diffusion-model.transformer.window-size-t N` |
| Disable torch.compile on the DiT | `--pipeline.diffusion-model.transformer.compile-network False` |

(Output resolution is a first-class runner flag — use `--resolution H,W`
from section 5 rather than reaching into the transformer config.)

Booleans expect explicit `True`/`False` (the CLI runs with tyro's
`FlagConversionOff`).

To see every available flag for a given runner:

```bash
uv run flashdreams-run cosmosHDreams-chunk3-vae-vae --help
```

---

## 6. Outputs

For each entry, the runner writes (next to the JSON's `output_video`
path, anchored under `--output-dir` when relative):

| File | Contents |
|---|---|
| `<name>.mp4` | Predicted video at `--fps`. |
| `<name>_annotated.mp4` | Same video with `Frame N` overlays. |
| `<name>.npy` | Raw float tensor of the full video, shape `[3, T, H, W]`, range `[-1, 1]`. |
| `<name>_comparison.mp4` | Side-by-side input vs. prediction (only when `--save-comparison True`). |

Per-block timing logs also stream to stdout. The first outer block of
each entry is reported as `[WARMUP]` because it absorbs `torch.compile`
JIT and the first CUDA-graph capture; steady-state FPS in the summary
excludes it.

---

## Mode B — Interactive WebRTC (keyboard / Quest)

The `cosmosHDreams/` package wraps the recipe in a small WebRTC
server that streams generated frames to a browser (keyboard) or a
Meta Quest headset (WebXR).

Server configs live in `configs/`. Pick the one that matches the
dataset / episode you want to condition on; ports and asset paths are
documented in the YAML itself.

> **Note — first-input latency.** In interactive mode the DiT is compiled
> with `torch.compile` on the first forward pass triggered by user input.
> Expect the first generation to take significantly longer than
> steady-state; subsequent generations run at normal speed.

## 7. Start the WebRTC server

### Keyboard

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server \
  --config cosmosHDreams/configs/keyboard_tabletop.yaml
```

Then open **<http://0.0.0.0:8080/keyboard>** in any
browser. The viewer page handles SDP signaling on its own — once it
loads, you should click on "Connect session" and you should see the conditional first frame and can start
the simulation with the keys documented in `README.md`
("DataChannel Message Format" → key bindings).

### Quest (WebXR)

WebXR requires HTTPS, so the Quest path needs a TLS cert. One-time
setup:

1. **Put the headset in developer mode.** Pair it with the Meta
   Quest mobile app, enable developer settings (Meta's official docs cover the latest
   flow.)
2. **Generate a self-signed certificate** on the host that will run
   the server. Replace `<host-ip>` with the LAN IP of that
   host so the Quest can verify the cert against the address it
   connects to:

   ```bash
   openssl req -x509 -newkey rsa:2048 -nodes \
     -keyout key.pem -out cert.pem -days 365 \
     -subj "/CN=quest" \
     -addext "subjectAltName=IP:<host-ip>"
   ```

   The Quest config (`configs/quest_*.yaml`) points at these
   `key.pem` / `cert.pem` paths — keep them next to the configs or
   update the YAML to match.

Launch the Quest server:

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server_quest \
  --config cosmosHDreams/configs/quest_tabletop.yaml
```

On the Quest browser, open **`https://<host-ip>:8443/quest`**.
Accept the self-signed-cert warning, click **Enter VR**, and you're
in. **Hold `B` on the Meta Quest controller to reset the simulation**
(server drains the queue and re-anchors on the initial conditional
frame). **Hold `Y` on the left controller for 1 s to exit immersive
mode** and drop back to the 2D landing page.

### Unified — keyboard + Quest on one port

For a demo where users can drive either from a browser or a
Quest headset, the unified server mounts both demos on a single HTTPS
port sharing one rollout. Whichever side connects most recently
drives; the other side's connection is closed automatically (takeover
semantics — kicked Quest clients see a "Take over" button to come
back). One-time setup is the same as the Quest path (developer-mode
headset + self-signed cert), since WebXR requires HTTPS regardless.

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server_unified \
  --config cosmosHDreams/configs/unified_tabletop.yaml
```

URLs once running (HTTPS on the configured port — default `8443`):

| Path | What |
|---|---|
| `/` | Landing page with links to the demos. |
| `/keyboard` | Keyboard demo (WebRTC, any browser). |
| `/quest` | Quest demo (open in the Quest browser; WebXR). |
| `/viewer` | Admin / spectator view — shows who's driving, the active scene, and the rendered stream. |

The YAML schema is the union of the keyboard and Quest schemas
(`runtime` + `scenes` + `keyboard` + `vr` + `video` + `server`); the
`scenes:` list lets users switch scenes live from the in-page
dropdown on either side. See `configs/unified_tabletop.yaml`
for a working example.

---

## 8. Troubleshooting

- **`ImportError: cannot import name '…' from 'flashdreams.infra.encoder'`** — you're running outside `uv run` against a stale environment. Re-run with `uv run flashdreams-run …` from inside the synced project.
- **`invalid choice '…' for argument '--save-comparison'`** — pass `True` (or `False`) explicitly; the CLI has flag-conversion disabled globally.
- **`Unrecognized options: --pipeline.diffusion-model.transformer.checkpoint-path`** — make sure your `cosmosHDreams/` source has the runner refactor applied (this guide assumes it). After the refactor, the override works directly with no subcommand traversal.
- **Out-of-memory** — try a `lighttae` decoder variant (`cosmosHDreams-chunk3-vae-lighttae` or `cosmosHDreams-chunk3-2steps-vae-lighttae`); they cut decoder activation memory considerably.
- **WebRTC stream flickers with wrong colors** — the GPU is saturated by the diffusion process and cannot simultaneously run NVENC reliably. Force the CPU encoder in the server YAML: `video: encoder: cpu_libav`.
