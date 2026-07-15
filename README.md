# Cosmos-H-Dreams

[![License](https://img.shields.io/badge/Code-Apache_2.0-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-NVIDIA_Open_Model-green.svg)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](TODO)
[![Paper](https://img.shields.io/badge/arXiv-TODO-red.svg)](TODO)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)

Real-time action-conditioned surgical video simulation via WebRTC, built on [FlashDreams](https://github.com/NVIDIA/flashdreams).

## Overview

Cosmos-H-Dreams is a fine-tuned variant of Cosmos-H-Surgical-Simulator, with its own checkpoint and a serving layer in a streaming server, enabling live surgical simulation driven by keyboard or Meta Quest controller input. Given a conditional first frame from a surgical procedure and a live stream of instrument action vectors, the model rolls forward in blocks of generated frames and streams the output to a browser or VR headset in real time via WebRTC.

The system is built on top of [FlashDreams](https://github.com/NVIDIA/flashdreams), NVIDIA's high-performance inference and serving library for autoregressive video models. It uses a fine-tuned checkpoint from [Cosmos-H-Surgical-Simulator](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator) and supports two modes:

- **Offline batch inference** — feed a JSON manifest of `{input_video, input_action, output_video}` entries and produce MP4 + raw tensor outputs.
- **Interactive WebRTC** — drive the rollout live from a browser (keyboard) or a Meta Quest headset (WebXR), with no action `.npy` needed.

## News

- **[July, 2026]** — Initial release of Cosmos-H-Dreams

## Runner Configurations

Slugs follow the pattern `cosmosHDreams-[chunk{N}-][2steps-]{encoder}-{decoder}` across four independent axes:


| Axis                        | Choices            | Notes                                                                                                                             |
| --------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **Chunk size** (`chunk{N}`) | `chunk2`, `chunk3` | Latent frames per DiT forward pass. Higher = fewer DiT calls for the same output. `chunk3` is recommended for throughput.         |
| **Schedule**                | `4steps`, `2steps` | 4-step is closer to the training distribution; 2-step gives ~2× DiT speedup at some fidelity cost. Omitting defaults to `4steps`. |
| **Encoder**                 | `vae`              | Full Wan2.1 VAE.                                                                                                                  |
| **Decoder**                 | `vae`, `lighttae`  | Full Wan2.1 VAE vs. TAEHV `lighttae` (~10× faster decode, modest quality drop).                                                   |


The recommended chunk3 variants:


| Slug                                       | Schedule | Encoder | Decoder  |
| ------------------------------------------ | -------- | ------- | -------- |
| `cosmosHDreams-chunk3-4steps-vae-vae` ⭐    | 4-step   | vae     | vae      |
| `cosmosHDreams-chunk3-4steps-vae-lighttae` | 4-step   | vae     | lighttae |
| `cosmosHDreams-chunk3-2steps-vae-vae`      | 2-step   | vae     | vae      |
| `cosmosHDreams-chunk3-2steps-vae-lighttae` | 2-step   | vae     | lighttae |


All 12 configs share the same checkpoint and DiT geometry. Run `uv run flashdreams-run --help` to list every available slug.

## Quick Start

### 1. Build the container

```bash
docker build -t cosmos-h-dreams:latest docker/
```

### 2. Start the container

Place the following assets under the repo root before launching:

- `checkpoints/` — CosmosH `.pt` checkpoint(s)
- `sf_inference_data/` — input manifests, action `.npy` files, and the precomputed CR1 text embeddings `.pt`

```bash
docker run --rm -it \
  --network host \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v .:/workspace/cosmos-h-dreams \
  -w /workspace/cosmos-h-dreams \
  cosmos-h-dreams:latest /bin/bash
```

### 3. Install dependencies

```bash
uv sync --extra dev --extra runners --group lint
```

### 4. Mode A — Offline batch inference

```bash
uv run flashdreams-run cosmosHDreams-chunk3-vae-vae \
  --input-json assets/example_data/offline/suturebot_inference_manifest.json \
  --cr1-embeddings-path sf_inference_data/cr1_empty_string_text_embeddings.pt \
  --root-dir . \
  --total-blocks 20 \
  --save-comparison True \
  --resolution 288,512 \
  --pipeline.diffusion-model.transformer.checkpoint-path checkpoints/model_ema_bf16_jhutabletop_288x512_h73.pt
```

Each entry produces `<name>.mp4`, `<name>_annotated.mp4`, `<name>.npy` (raw `[3, T, H, W]` tensor in `[-1, 1]`), and `<name>_latents.npy`.

> **Note — first-run latency.** The first block absorbs `torch.compile` JIT and the initial CUDA-graph capture (reported as `[WARMUP]`). Compiled artifacts are cached and reused on subsequent runs of the same configuration and resolution.

### 5. Mode B — Interactive WebRTC

**Keyboard (any browser):**

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server \
  --config cosmosHDreams/configs/keyboard_tabletop.yaml
```

Open **[http://0.0.0.0:8080/keyboard](http://0.0.0.0:8080/keyboard)** in any browser to start controlling the surgical robot.

**Meta Quest (WebXR):**

WebXR requires HTTPS. Generate a self-signed certificate once, replacing `<host-ip>` with the LAN IP of the server:

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=quest" \
  -addext "subjectAltName=IP:<host-ip>"
```

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server_quest \
  --config cosmosHDreams/configs/quest_tabletop.yaml
```

Open `https://<host-ip>:8443/quest` in the Quest browser and click **Enter VR**. Hold `B` to reset the simulation; hold `Y` for 1 s to exit immersive mode.

**Unified (keyboard + Quest on one port):**

```bash
uv run --package flash-cosmosHDreams python -m cosmosHDreams.webrtc.server_unified \
  --config cosmosHDreams/configs/unified_tabletop.yaml
```

Serves `/keyboard`, `/quest`, `/viewer`, and `/` on a single HTTPS port (default `8443`). Whichever client connects most recently drives; the other is paused until it reconnects (takeover semantics).

> **Note — first-input latency.** In interactive mode the DiT is compiled with `torch.compile` on the first forward pass triggered by user input. Expect the first generation to take significantly longer than steady-state; subsequent generations run at normal speed.

## Documentation


| Guide | Description |
|-------|-------------|
| [`cosmosHDreams/GUIDE.md`](cosmosHDreams/GUIDE.md) | Container setup, dependency sync, config selection, full flag reference for offline inference and all three WebRTC servers |
| [`cosmosHDreams/README.md`](cosmosHDreams/README.md) | Package reference: run commands, DataChannel protocol, key bindings |
| [`cosmosHDreams/configs/`](cosmosHDreams/configs/) | Annotated YAML schemas (runtime, scenes, server, video settings) |


## System Requirements

- NVIDIA GPU with at least 12GB of VRAM.
- NVIDIA driver **R580 series or newer** (CUDA 13.x)
- **Python >= 3.12**
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## License


| Component                     | License                                                                                                     |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Source code                   | [Apache 2.0](LICENSE)                                                                                       |
| Cosmos-H-Dreams model weights | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |


## Resources

- [Paper](TODO) — Cosmos-H-Dreams technical report
- [HuggingFace](TODO) — Model weights and checkpoints
- [Cosmos-H-Surgical-Simulator](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator) — Base model that Cosmos-H-Dreams is fine-tuned from (offline inference and fine-tuning)
- [FlashDreams](https://github.com/NVIDIA/flashdreams) — Underlying high-performance inference runtime
- [Open-H Dataset](https://huggingface.co/datasets/nvidia/Open-H) — Multi-embodiment surgical benchmark used for training
- [NVIDIA Cosmos Platform](https://www.nvidia.com/en-us/ai/cosmos) — Product website

## Known Issues

**WebRTC stream flickers with wrong colors** — the video stream in the browser or Quest headset shows color artifacts or flickers between frames. This happens when the GPU is saturated by the diffusion process and cannot simultaneously run the NVENC hardware encoder reliably. Fix: force the CPU encoder in the server YAML:

```yaml
video:
  encoder: cpu_libav
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs and submitting changes.