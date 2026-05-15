# Running CosmosH inference

This guide walks through running the CosmosH action-conditioned streaming
Video2World recipe end to end: build a container, drop into it, and run
one of the bundled configurations.

The recipe takes a CosmosH-2B autoregressive policy, conditions it on a
first frame + a precomputed CR1 text embedding, and rolls it forward
in blocks of 12 generated frames.

The guide has two parts after the shared setup:

- **Common setup** (sections 1–4) — build the container, mount your
  assets, sync deps, pick one of the six bundled configurations.
- **Mode A — Offline batch inference** (sections 5–6) — feed a JSON
  manifest of `{input_video, input_action, output_video}` entries and
  drive the recipe with action `.npy` trajectories. Outputs are MP4 +
  raw-tensor files. Reach for this when you want reproducible rollouts
  or benchmark numbers.
- **Mode B — Interactive WebRTC** (section 7) — start the
  `integrations/cosmosh/` WebRTC server and drive the rollout live
  from either a keyboard (browser) or a Meta Quest headset (WebXR).
  No action `.npy` needed; the browser / headset is the input device.

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
docker build -t flashdreams:public docker/
```

That tags the image as `flashdreams:public` locally — the tag the
`docker run` line below expects. Rebuild only when `docker/Dockerfile`
or its pinned deps change.

---

## 2. Start the container

Before launching, drop the assets you'll need inside the repo so they
are reachable from inside the container via the single bind mount:

- `checkpoints/` — the CosmosH `.pt` checkpoint(s)
- `sf_inference_data/` — your input-video manifest, action `.npy`s,
  and the precomputed CR1 text embedding

(The exact names aren't enforced — keep them anywhere under the repo
root; the CLI paths below assume these.)

```bash
docker run --rm -it \
  --network host \
  --gpus all \
  -v .:/workspace/flashdreams \
  -w /workspace/flashdreams \
  flashdreams:public /bin/bash
```

What each flag does:

| Flag | Why |
|---|---|
| `--rm -it` | Throwaway interactive shell. |
| `--network host` | Lets HuggingFace / S3 / proxy traffic reach the host's network. |
| `--gpus all` | Exposes every visible GPU; restrict with `--gpus '"device=0,1"'` if you need to pin. |
| `-v .:/workspace/flashdreams` | Mounts the checkout (including `checkpoints/` and `sf_inference_data/`) into the container so edits + assets are live. |
| `-w /workspace/flashdreams` | Drops you into the repo root (where the top-level `pyproject.toml` lives) so `uv run …` works out of the box. |

The shell lands directly at the repo root — no further `cd` needed.
All commands below are run from there.

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

The CosmosH recipe ships six **named runner configurations**. The slug
encodes the schedule length and the VAE encoder/decoder pairing on
either side of the DiT:

| Slug | Schedule | First-frame encoder | Output decoder | Notes |
|---|---|---|---|---|
| `cosmosh-vae-vae` | 4-step | full Wan2.1 VAE | full Wan2.1 VAE | Highest fidelity. Reasonable default. |
| `cosmosh-vae-lighttae` | 4-step | full Wan2.1 VAE | TAEHV `lighttae` | Faster decode, modest quality drop. |
| `cosmosh-lightvae-lighttae` | 4-step | distilled `lightvae` | TAEHV `lighttae` | Fastest 4-step preset. |
| `cosmosh-2steps-vae-vae` | 2-step | full Wan2.1 VAE | full Wan2.1 VAE | Half the denoising hops; same VAEs as 4-step. |
| `cosmosh-2steps-vae-lighttae` | 2-step | full Wan2.1 VAE | TAEHV `lighttae` | |
| `cosmosh-2steps-lightvae-lighttae` | 2-step | distilled `lightvae` | TAEHV `lighttae` | Fastest overall. |

All six share the same CosmosH-2B checkpoint, the same DiT geometry
(latent 60×80 → 480×640 pixels per frame), and the same action-conditioned
streaming loop. The only knobs the slug picks are the denoising
schedule length and the VAE pair.

A few rules of thumb:

- Start with `cosmosh-vae-vae` to validate the integration, then move
  toward `lighttae` decoders for throughput once outputs look right.
- The 4-step variants are closer to the reference upstream output; the
  2-step variants trade some fidelity for ~2× DiT speedup.
- `lighttae` decoders are roughly 10× faster than the full Wan decoder.

To see every runner the CLI knows about (including the non-CosmosH
ones), run `uv run flashdreams-run --help` from inside the container.

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

Optional but useful:

- **`--root-dir`** — prefixed to every `input_video` / `input_action` path in the JSON.
- **`--total-blocks`** — how many 12-frame outer blocks to roll. Each block consumes 12 action steps; the loop exits early if the action trajectory is shorter.
- **`--start-frame-idx`** — default conditional-frame index when the JSON doesn't specify it.
- **`--save-comparison True`** — also write `*_comparison.mp4` (input on the left, prediction on the right).
- **`--fps`** — output MP4 frame rate (default `10`).
- **`--output-dir`** — where outputs land. Per-entry `output_video` paths are anchored under this when relative; absolute paths in the JSON are honored as-is.
- **`--resolution H,W`** — override the bundle's default output resolution (default `480,640`). Both dimensions must be multiples of 8 (the Wan2.1 VAE spatial compression ratio). Example: `--resolution 704,1280`.

Example — full Wan VAE on both sides, comparison MP4 on, override the
checkpoint to a domain-specific one:

```bash
uv run flashdreams-run cosmosh-vae-vae \
  --input-json sf_inference_data/260206/suturebot_inference_manifest.json \
  --cr1-embeddings-path sf_inference_data/cr1_empty_string_text_embeddings.pt \
  --root-dir . \
  --total-blocks 20 \
  --save-comparison True \
  --pipeline.diffusion-model.transformer.checkpoint-path checkpoints/model_ema_jhutabletop_bf16.pt
```

To switch configuration, swap the first positional. Everything else
stays the same:

```bash
uv run flashdreams-run cosmosh-2steps-lightvae-lighttae \
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
> same `cosmosh-*` slug start much faster — only the CUDA-graph
> capture (~a few seconds) runs again. Changing `--resolution`, the
> checkpoint, or the slug forces a recompile.

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
uv run flashdreams-run cosmosh-vae-vae --help
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
| `<name>_latents.npy` | Per-block VAE latents, shape `[num_blocks, 1, T_lat, C_lat, H_lat, W_lat]`. |
| `<name>_comparison.mp4` | Side-by-side input vs. prediction (only when `--save-comparison True`). |

Per-block timing logs also stream to stdout. The first outer block of
each entry is reported as `[WARMUP]` because it absorbs `torch.compile`
JIT and the first CUDA-graph capture; steady-state FPS in the summary
excludes it.

---

## Mode B — Interactive WebRTC (keyboard / Quest)

The `integrations/cosmosh/` package wraps the recipe in a small WebRTC
server that streams generated frames to a browser (keyboard) or a
Meta Quest headset (WebXR). No action `.npy` manifest needed — the
client device produces the action stream live.

Server configs live in `integrations/cosmosh/configs/`. Pick the one
that matches the dataset / episode you want to condition on; ports
and asset paths are documented in the YAML itself.

## 7. Start the WebRTC server

### Keyboard

```bash
uv run --package flash-cosmosh python -m cosmosh.webrtc.server \
  --config integrations/cosmosh/configs/keyboard_episode_001867.yaml
```

Then open **<http://0.0.0.0:8080/request_session>** in any
browser. The viewer page handles SDP signaling on its own — once it
loads, you should see the conditional first frame and can start
driving with the keys documented in `integrations/cosmosh/README.md`
("DataChannel Message Format" → key bindings).

### Quest (WebXR)

WebXR requires HTTPS, so the Quest path needs a TLS cert. One-time
setup:

1. **Put the headset in developer mode.** Pair it with the Meta
   Quest mobile app, enable developer settings (Meta's official docs cover the latest
   flow.)
2. **Generate a self-signed certificate** on the host that will run
   the server. Replace `<bridge-pc-lan-ip>` with the LAN IP of that
   host so the Quest can verify the cert against the address it
   connects to:

   ```bash
   openssl req -x509 -newkey rsa:2048 -nodes \
     -keyout key.pem -out cert.pem -days 365 \
     -subj "/CN=quest" \
     -addext "subjectAltName=IP:<bridge-pc-lan-ip>"
   ```

   The Quest config (`configs/quest_*.yaml`) points at these
   `key.pem` / `cert.pem` paths — keep them next to the configs or
   update the YAML to match.

Launch the Quest server:

```bash
uv run --package flash-cosmosh python -m cosmosh.webrtc.server_quest \
  --config integrations/cosmosh/configs/quest_episode_001867.yaml
```

On the Quest browser, open **`https://<bridge-pc-lan-ip>:8443/quest_session`**.
Accept the self-signed-cert warning, click **Enter VR**, and you're
in. **Hold `B` on the Meta Quest controller to reset the simulation**
(server drains the queue and re-anchors on the initial conditional
frame).

---

## 8. Troubleshooting

- **`ImportError: cannot import name '…' from 'flashdreams.infra.encoder'`** — you're running outside `uv run` against a stale environment. Re-run with `uv run flashdreams-run …` from inside the synced project.
- **`invalid choice '…' for argument '--save-comparison'`** — pass `True` (or `False`) explicitly; the CLI has flag-conversion disabled globally.
- **`Unrecognized options: --pipeline.diffusion-model.transformer.checkpoint-path`** — make sure your `cosmosh/` source has the runner refactor applied (this guide assumes it). After the refactor, the override works directly with no subcommand traversal.
- **Out-of-memory** — try a `lighttae` decoder variant (`cosmosh-vae-lighttae` or one of the `2steps-*-lighttae` slugs); they cut decoder activation memory considerably.
