# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action-conditioned streaming Video2World runner for the CosmosH recipe.

Mirrors the per-entry pipeline of the upstream
``cosmos_predict2._src.predict2.interactive.inference.action_video2world_streaming_deterministic``
script so outputs are directly comparable:

- Reads each input video, takes ``start_frame_idx`` as the conditional
  first frame, and resizes to the pipeline's locked latent dims times
  the Wan2.1 spatial compression ratio.
- Encodes that frame via the Wan2.1 VAE encoder to produce the
  ``image_embeddings`` consumed by :class:`CosmoshPipeline`.
- Runs ``total_blocks`` outer blocks, each generating 12 pixel frames
  (4 latent frames at the Wan2.1 temporal compression of 4). Each
  outer block re-anchors on the previous block's last decoded frame.
- Decodes per-block latents back to pixels via the configured VAE
  decoder and writes an MP4 + ``.npy`` per entry.

The CosmosH pipeline keeps ``decoder=None`` (per-block latents are
decoded externally), so the Wan2.1 VAE encoder + decoder are carried
on the runner config alongside the streaming pipeline.

  uv run flashdreams-run cosmosh-vae-vae \
    --input-json /localhome/local-javierg/sf_inference_data/260206/suturebot_inference_manifest.json \
    --cr1-embeddings-path /localhome/local-javierg/sf_inference_data/cr1_empty_string_text_embeddings.pt \
    --root-dir /localhome/local-javierg/ \
    --total-blocks 20 \
    --save-comparison True \
    --pipeline.diffusion-model.transformer.checkpoint-path /localhome/local-javierg/checkpoints/model_ema_jhutabletop_bf16.pt

  uv run flashdreams-run cosmosh-2steps-vae-lighttae \
    --input-json /localhome/local-javierg/sf_inference_data/260206/suturebot_inference_manifest.json \
    --cr1-embeddings-path /localhome/local-javierg/sf_inference_data/cr1_empty_string_text_embeddings.pt \
    --root-dir /localhome/local-javierg/ \
    --total-blocks 20 \
    --save-comparison True \
    --pipeline.diffusion-model.transformer.checkpoint-path /localhome/local-javierg/checkpoints/model_ema_jhutabletop_bf16.pt

  uv run flashdreams-run cosmosh-chunk3-vae-vae   \
    --input-json sf_inference_data/cmr/trajectories/hyst_exp4_test/hyst_exp4_inference_manifest.json \
    --cr1-embeddings-path sf_inference_data/cr1_empty_string_text_embeddings.pt \
    --root-dir . \
    --total-blocks 20 \
    --save-comparison True \
    --pipeline.diffusion-model.transformer.checkpoint-path checkpoints/cmr/hyst/model_ema_bf16_cmr_hyst_exp5.pt
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from flashdreams.infra.decoder import DecoderConfig
from flashdreams.infra.runner import Runner, RunnerConfig
from cosmosh.encoder.action import ActionEncoderConfig
from cosmosh.pipeline import CosmoshPipeline
from cosmosh.transformer import (
    CosmosHTransformer,
    CosmosHTransformerConfig,
)
from cosmosh.utils import load_cr1_text_embeddings, pad_actions, pixel_frame_to_neg1_pos1
from flashdreams.recipes.wan.autoencoder.vae import WanVAEEncoderConfig

# Wan2.1 VAE temporal / spatial compression ratios. Each generated latent
# frame decodes to ``WAN_TCR`` pixel frames; the conditional frame decodes to 1.
WAN_TCR = 4
WAN_SCR = 8

# Per-outer-block sizes are derived at runtime from the transformer's latent
# frames per step (``_pT == len_t``) and the runner's ``steps_per_block``:
#   generated latent frames/block = steps_per_block * len_t - 1   (the -1 is the
#                                   single conditional frame at AR step 0)
#   actions consumed/block        = (steps_per_block * len_t - 1) * A
#   decoded pixel frames/block    = (steps_per_block * len_t - 1) * WAN_TCR
# At the default len_t=1, steps_per_block=4 this is 3 latent frames -> 12 pixels,
# matching the original fixed 12-action / 13-pixel (1 cond + 12 gen) block.


@dataclass(kw_only=True)
class CosmoshRunnerConfig(RunnerConfig):
    """Runner config for any CosmosH variant.

    The CosmosH pipeline accumulates per-block latents and decodes them
    externally, so the Wan2.1 VAE encoder (for the conditional first
    frame) and the per-block decoder live on the runner rather than
    inside the pipeline.
    """

    _target: type = field(default_factory=lambda: CosmoshRunner)

    vae_encoder: WanVAEEncoderConfig
    """Wan2.1 VAE encoder for the conditional first frame."""

    vae_decoder: DecoderConfig
    """Decoder for per-block output latents. Either the full Wan2.1 VAE
    or the TAEHV ``lighttae`` distilled drop-in."""

    input_json: Path | None = None
    """JSON listing per-entry inputs: ``input_video``, ``input_action``,
    ``output_video`` (plus optional ``start_frame_idx`` / ``resolution``
    per-entry overrides). Required at ``run()`` time."""

    cr1_embeddings_path: Path | None = None
    """Path to a precomputed CR1 text-embeddings ``.pt`` file. Required
    at ``run()`` time."""

    root_dir: Path = Path("")
    """Root directory ``input_video`` and ``input_action`` are resolved
    against. Empty means absolute paths in the JSON."""

    total_blocks: int = 20
    """Number of outer blocks to attempt. The loop stops early once the action
    stream is consumed."""

    len_t: int = 1
    """Latent frames generated per feed-forward (`generate()`) step. Sets the
    transformer's ``len_t`` (and the action encoder's ``latent_frames_per_step``)
    before the pipeline is built. Each generated latent frame decodes to
    ``WAN_TCR`` pixel frames. ``window_size_t`` must stay a whole multiple of
    ``len_t``; the product ``steps_per_block * len_t`` must not exceed it."""

    steps_per_block: int = 4
    """Number of ``generate()`` calls per outer block. AR step 0 leads with the
    conditional (image-anchored) frame, so a block generates
    ``steps_per_block * len_t - 1`` latent frames and re-anchors on the last
    decoded pixel frame. Default ``4`` with ``len_t=1`` reproduces the original
    12-generated-pixel block."""

    start_frame_idx: int = 0
    """Default index of the input-video frame used as the conditional
    first frame. Per-entry ``start_frame_idx`` in the JSON wins."""

    fps: float = 10.0
    """Output MP4 frame rate."""

    save_comparison: bool = False
    """Also write a side-by-side comparison MP4 (input on left, predicted
    on right)."""

    resolution: str | None = None
    """Output resolution as ``"H,W"`` in pixels (e.g. ``"704,1280"``).
    Both dimensions must be multiples of 8 (the Wan2.1 VAE spatial
    compression ratio). ``None`` resolves to the bundle's pinned default
    in ``__post_init__`` so the dumped config always shows an explicit
    ``H,W``."""

    def __post_init__(self) -> None:
        # Resolve a ``None`` resolution to the bundle's pinned pixel dims
        # so the config dump prints the effective resolution rather than
        # ``None``. Idempotent across ``derive_config`` clones because
        # the same H,W round-trips identically.
        if self.resolution is None:
            tcfg = self.pipeline.diffusion_model.transformer
            if isinstance(tcfg, CosmosHTransformerConfig):
                self.resolution = f"{tcfg.height * WAN_SCR},{tcfg.width * WAN_SCR}"


class CosmoshRunner(Runner[CosmoshRunnerConfig, CosmoshPipeline]):
    """Streaming action-conditioned I2V driver for CosmosH."""

    config: CosmoshRunnerConfig

    def __init__(self, config: CosmoshRunnerConfig) -> None:
        # The pipeline's locked latent dims are baked at ``setup()`` time
        # (RoPE tables + KV-cache shape), so any user override has to land
        # on the config *before* the base ``Runner.__init__`` builds the
        # pipeline. ``__post_init__`` guarantees ``resolution`` is set.
        assert config.resolution is not None
        try:
            h_str, w_str = config.resolution.split(",")
            new_h, new_w = int(h_str), int(w_str)
        except ValueError as exc:
            raise ValueError(
                f"--resolution must be 'H,W' (e.g. '704,1280'); got "
                f"{config.resolution!r}."
            ) from exc
        assert new_h % WAN_SCR == 0 and new_w % WAN_SCR == 0, (
            f"--resolution dims must be multiples of {WAN_SCR} "
            f"(Wan2.1 VAE spatial ratio); got {new_h}x{new_w}."
        )
        tcfg = config.pipeline.diffusion_model.transformer
        assert isinstance(tcfg, CosmosHTransformerConfig)
        tcfg.height = new_h // WAN_SCR
        tcfg.width = new_w // WAN_SCR
        # Latent frames generated per feed-forward step. Like the resolution,
        # this is baked into RoPE tables + KV-cache geometry at ``setup()``, so
        # it has to land on the config *before* the base ``Runner.__init__``
        # builds the pipeline. Keep the action encoder's per-step frame count in
        # lockstep — ``CosmoshPipeline.__init__`` asserts they match ``_pT``.
        assert config.len_t >= 1, f"len_t must be >= 1, got {config.len_t}"
        assert config.steps_per_block >= 1, (
            f"steps_per_block must be >= 1, got {config.steps_per_block}"
        )
        assert config.steps_per_block * config.len_t >= 2, (
            "steps_per_block * len_t must be >= 2 so each outer block generates "
            f"at least one frame (got steps_per_block={config.steps_per_block}, "
            f"len_t={config.len_t})."
        )
        tcfg.len_t = config.len_t
        encoder_cfg = config.pipeline.encoder
        assert isinstance(encoder_cfg, ActionEncoderConfig)
        encoder_cfg.latent_frames_per_step = config.len_t
        # ``__post_init__`` derives ``_pT / _pH / _pW / _steady_ar_idx`` from
        # the latent dims and runs only once at dataclass construction (during
        # bundle build). The DiT reads those derived fields when ``setup()``
        # builds RoPE tables + the patchifier, so the mutation above is a
        # no-op for the network unless we re-run the post-init to refresh
        # them. Without this, the VAE encoder produces a latent at the new
        # resolution while the DiT stays sized for the bundle default and
        # ``_maybe_inject_image`` fails with a token-count mismatch.
        tcfg.__post_init__()

        super().__init__(config)
        # The base ``Runner`` already built + pinned the pipeline to
        # ``cuda:LOCAL_RANK`` under torchrun (else ``config.device``).
        # Mirror that for the external VAE encoder / decoder, which the
        # CosmosH pipeline does not own.
        device = self.pipeline.device
        self.vae_encoder = self.config.vae_encoder.setup().to(device).eval()
        self.vae_decoder = self.config.vae_decoder.setup().to(device).eval()

    @torch.inference_mode()
    def run(self) -> None:
        cfg = self.config
        assert cfg.input_json is not None, (
            "CosmoshRunner requires --input_json (a JSON list of "
            "{input_video, input_action, output_video} entries)."
        )
        assert cfg.cr1_embeddings_path is not None, (
            "CosmoshRunner requires --cr1_embeddings_path "
            "(precomputed CR1 text embeddings .pt)."
        )

        tcfg = self.pipeline.diffusion_model.transformer.config
        Ht, Wt = tcfg.height * WAN_SCR, tcfg.width * WAN_SCR

        if self.is_rank_zero:
            logger.info("=" * 60)
            logger.info("COSMOSH ACTION-CONDITIONED STREAMING INFERENCE")
            logger.info("=" * 60)
            logger.info(f"  runner_name: {cfg.runner_name}")
            logger.info(f"  input_json: {cfg.input_json}")
            logger.info(f"  cr1_embeddings_path: {cfg.cr1_embeddings_path}")
            logger.info(f"  total_blocks: {cfg.total_blocks}")
            logger.info(f"  output_dir: {cfg.output_dir}")
            logger.info(
                f"  resolution: {Ht}x{Wt} pixels (latent {tcfg.height}x{tcfg.width})"
            )
            logger.info("=" * 60)

        text_embeddings_cpu = load_cr1_text_embeddings(str(cfg.cr1_embeddings_path))
        if self.is_rank_zero:
            logger.info(
                f"Loaded CR1 embeddings shape={tuple(text_embeddings_cpu.shape)} "
                f"from {cfg.cr1_embeddings_path}"
            )

        with open(cfg.input_json, "r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"{cfg.input_json} must contain a list of entries")
        if self.is_rank_zero:
            logger.info(f"Processing {len(entries)} entries")

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        all_stats: list[dict[str, float]] = []
        for idx, entry in enumerate(entries):
            if not all(
                k in entry for k in ("input_video", "input_action", "output_video")
            ):
                if self.is_rank_zero:
                    logger.warning(
                        f"Entry #{idx} missing required keys; needs input_video, "
                        "input_action, output_video. Skipping."
                    )
                continue
            if self.is_rank_zero:
                logger.info("-" * 60)
                logger.info(f"Entry {idx + 1}/{len(entries)}: {entry['input_video']}")
                logger.info("-" * 60)
            stats = self._run_entry(entry, text_embeddings_cpu=text_embeddings_cpu)
            if stats is None:
                continue
            all_stats.append(stats)
            if self.is_rank_zero:
                steady_fps = (
                    stats["steady_frames"] / stats["steady_time"]
                    if stats["steady_time"] > 0
                    else 0.0
                )
                logger.info(
                    f"  total {stats['total_time']:.2f}s "
                    f"(warmup {stats['warmup_time']:.2f}s + "
                    f"steady {stats['steady_time']:.2f}s for "
                    f"{int(stats['steady_frames'])} frames "
                    f"= {steady_fps:.2f} FPS), "
                    f"encode {stats['encode_time']:.2f}s, "
                    f"gen {stats['generation_time']:.2f}s, "
                    f"decode {stats['decode_time']:.2f}s, "
                    f"frames {int(stats['frames_generated'])}"
                )

        if all_stats and self.is_rank_zero:
            total_frames = sum(int(s["frames_generated"]) for s in all_stats)
            total_time = sum(s["total_time"] for s in all_stats)
            steady_total_time = sum(s["steady_time"] for s in all_stats)
            steady_total_frames = sum(int(s["steady_frames"]) for s in all_stats)
            steady_total_fps = (
                steady_total_frames / steady_total_time
                if steady_total_time > 0
                else 0.0
            )
            cold_fps = total_frames / total_time if total_time > 0 else 0.0
            logger.info("=" * 60)
            logger.info(
                f"DONE: {len(all_stats)} entries, {total_frames} generated frames"
            )
            logger.info(
                f"  cold (incl. compile + capture): {total_time:.2f}s wall, "
                f"{cold_fps:.2f} FPS"
            )
            logger.info(
                f"  steady-state (blocks 1+):       {steady_total_time:.2f}s wall, "
                f"{steady_total_fps:.2f} FPS  ({steady_total_frames} frames)"
            )
            logger.info("=" * 60)

    @torch.inference_mode()
    def _run_entry(
        self,
        entry: dict,
        *,
        text_embeddings_cpu: torch.Tensor,
    ) -> dict[str, float] | None:
        """Run one input_json entry end-to-end. Returns timing stats."""
        import mediapy  # noqa: PLC0415 -- under the ``runners`` extras

        cfg = self.config
        input_video_path = str(Path(cfg.root_dir) / entry["input_video"])
        input_action_path = str(Path(cfg.root_dir) / entry["input_action"])
        output_video_path = entry["output_video"]
        # When the entry's output_video is relative, anchor it to cfg.output_dir;
        # absolute paths are honored as-is for parity with the upstream script.
        if not Path(output_video_path).is_absolute():
            output_video_path = str(cfg.output_dir / output_video_path)
        out_dir = Path(output_video_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        transformer = self.pipeline.diffusion_model.transformer
        assert isinstance(transformer, CosmosHTransformer)
        tcfg: CosmosHTransformerConfig = transformer.config
        device = self.pipeline.device
        dtype = tcfg.dtype
        A = tcfg.network.num_action_per_latent_frame
        L = tcfg._pT  # latent frames generated per feed-forward step (== len_t)
        S = cfg.steps_per_block  # generate() calls per block

        # The pipeline's latent dims drive the output pixel size; per-entry
        # ``resolution`` overrides (if present) must agree.
        Ht = tcfg.height * WAN_SCR
        Wt = tcfg.width * WAN_SCR
        if (
            "resolution" in entry
            and isinstance(entry["resolution"], list)
            and len(entry["resolution"]) == 2
        ):
            entry_h, entry_w = int(entry["resolution"][0]), int(entry["resolution"][1])
            assert (entry_h, entry_w) == (Ht, Wt), (
                f"per-entry resolution {(entry_h, entry_w)} disagrees with the "
                f"pipeline's locked ({Ht}, {Wt}); override the pipeline's "
                "transformer.height / .width to switch resolutions."
            )

        video_array = mediapy.read_video(input_video_path)  # [T, H, W, 3] uint8
        start_frame_idx = int(entry.get("start_frame_idx", cfg.start_frame_idx))
        cond_frame = video_array[start_frame_idx]
        if (cond_frame.shape[0], cond_frame.shape[1]) != (Ht, Wt):
            cond_frame = mediapy.resize_image(cond_frame, (Ht, Wt))
        cond_pixels = pixel_frame_to_neg1_pos1(cond_frame, device=device, dtype=dtype)

        actions_np = np.load(input_action_path)
        actions_np = pad_actions(actions_np, target_dim=tcfg.network.action_dim)

        # Fully flat AR: only one conditional prefill (global AR step 0).
        # Each block of S AR steps produces S*L latent frames except block 0,
        # which produces S*L - 1 (one prefill). Total generated latents across
        # ar_total blocks: ar_total * S * L - 1. Actions needed: (ar_total * S *
        # L - 1) * A. Solving for ar_total from available actions:
        #   ar_total <= (N_actions // A + 1) / (S * L)
        ar_total = min(
            cfg.total_blocks,
            (actions_np.shape[0] // A + 1) // (S * L),
        )
        if ar_total <= 0:
            raise ValueError(
                f"actions_np has {actions_np.shape[0]} entries; need at least "
                f"{(S * L - 1) * A} for one outer block "
                f"(steps_per_block={S}, len_t={L})."
            )
        total_ar_steps = ar_total * S
        total_generated_latents = total_ar_steps * L - 1  # one prefill at AR 0
        total_actions_needed = total_generated_latents * A

        if self.is_rank_zero:
            logger.info(
                f"Entry {input_video_path}: resolution {Ht}x{Wt}, "
                f"{ar_total} blocks x {S} steps x {L} latent frames/step "
                f"({total_ar_steps} total AR steps, "
                f"{total_generated_latents} generated latents, "
                f"{total_generated_latents * WAN_TCR} generated pixel frames)"
            )

        text_embeddings = text_embeddings_cpu.to(device=device, dtype=dtype)

        # Encode the conditioning frame exactly once — no per-block re-encoding.
        # Move the encoder to GPU (no-op after the first entry), encode, then
        # immediately offload it back to CPU so the generation phase reclaims
        # its VRAM.
        self.vae_encoder.to(device)
        torch.cuda.synchronize()
        t = time.perf_counter()
        image_embeddings = self.vae_encoder(input=cond_pixels)
        torch.cuda.synchronize()
        t_encode = time.perf_counter() - t
        self.vae_encoder.cpu()
        torch.cuda.empty_cache()

        # Load the full action trajectory for the entire rollout at once.
        all_actions = (
            torch.from_numpy(actions_np[:total_actions_needed])
            .to(device=device, dtype=dtype)
            .unsqueeze(0)
        )

        # Initialize the pipeline cache once — no reinitialisation across blocks.
        cache = self.pipeline.initialize_cache(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            actions=all_actions,
        )

        # Persistent decoder cache carries temporal state across block decodes.
        # Block 0 (fresh cache) uses AR-0 causal semantics: first latent decodes
        # to 1 pixel frame; we skip it since cond_pixels is already frame 0.
        # Blocks 1+ (warm cache) use AR-1+ semantics: every latent decodes to
        # WAN_TCR pixel frames with no wasted frame.
        decoder_cache = self.vae_decoder.initialize_autoregressive_cache()

        final_video_pixels: list[torch.Tensor] = [
            cond_pixels.permute(0, 2, 1, 3, 4).contiguous()  # [1, 3, 1, H, W]
        ]
        latents_per_block: list[torch.Tensor] = []

        t_total_start = time.perf_counter()
        block_gen: list[float] = []
        block_decode: list[float] = []
        block_total: list[float] = []

        latent_buffer: list[torch.Tensor] = []
        block_gen_accum: float = 0.0
        block_start: float = t_total_start

        for global_ar_idx in range(total_ar_steps):
            block_idx = global_ar_idx // S
            is_last_step = global_ar_idx == total_ar_steps - 1
            is_block_start = global_ar_idx % S == 0
            is_block_end = (global_ar_idx + 1) % S == 0

            if is_block_start:
                block_start = time.perf_counter()
                block_gen_accum = 0.0

            torch.cuda.synchronize()
            t = time.perf_counter()
            # input=True triggers the ActionEncoder; the encoder ignores the
            # value and slices actions from cache.encoder_cache.actions.
            out_5d = self.pipeline.generate(global_ar_idx, cache, input=True)
            if not is_last_step:
                self.pipeline.finalize(global_ar_idx, cache)
            torch.cuda.synchronize()
            block_gen_accum += time.perf_counter() - t

            latent_buffer.append(out_5d)

            if is_block_end:
                block_latent = torch.cat(latent_buffer, dim=1)
                latent_buffer = []
                latents_per_block.append(block_latent.detach().float().cpu())

                torch.cuda.synchronize()
                t = time.perf_counter()
                block_pixels = self.vae_decoder(
                    input=block_latent, cache=decoder_cache
                )
                print("block_pixels.shape", block_pixels.shape)
                block_pixels = block_pixels.clamp(min=-1.0, max=1.0)
                torch.cuda.synchronize()
                dec_t = time.perf_counter() - t

                block_pixels_b3thw = block_pixels.permute(0, 2, 1, 3, 4).contiguous()
                if block_idx == 0:
                    # Skip the first decoded pixel frame: it is the VAE's
                    # reconstruction of the conditional AR-0 latent, already
                    # held in final_video_pixels[0] as cond_pixels.
                    final_video_pixels.append(block_pixels_b3thw[:, :, 1:])
                else:
                    # All decoded frames are newly generated.
                    final_video_pixels.append(block_pixels_b3thw)

                total_t = time.perf_counter() - block_start
                block_gen.append(block_gen_accum)
                block_decode.append(dec_t)
                block_total.append(total_t)

                if self.is_rank_zero:
                    tag = "WARMUP" if block_idx == 0 else "steady"
                    logger.info(
                        f"  outer block {block_idx + 1}/{ar_total} [{tag}]: "
                        f"gen={block_gen_accum:.2f}s decode={dec_t:.2f}s "
                        f"total={total_t:.2f}s"
                    )

        total_time = time.perf_counter() - t_total_start
        t_gen_total = sum(block_gen)
        t_decode_total = sum(block_decode)

        warmup_time = block_total[0] if block_total else 0.0
        steady_blocks_total = sum(block_total[1:])
        steady_blocks_count = max(0, len(block_total) - 1)
        # Steady blocks use the warm decoder cache (AR-1+ semantics): each of
        # the S latents decodes to WAN_TCR pixel frames with no skipped frame.
        steady_frames = steady_blocks_count * S * L * WAN_TCR
        steady_fps = (
            steady_frames / steady_blocks_total if steady_blocks_total > 0 else 0.0
        )
        if steady_blocks_count > 0 and self.is_rank_zero:
            logger.info(
                f"  warmup block: {warmup_time:.2f}s "
                f"(includes torch.compile + first CUDA-graph capture)"
            )
            logger.info(
                f"  steady-state: {steady_blocks_count} blocks in "
                f"{steady_blocks_total:.2f}s -> {steady_fps:.2f} FPS "
                f"({steady_frames} generated frames)"
            )

        # Per-stage averages in milliseconds. Prefer steady-state blocks when
        # we have them so the numbers aren't dragged by the warmup block's
        # compile + CUDA-graph capture cost; fall back to all blocks for
        # single-block entries.
        if self.is_rank_zero and block_total:
            if steady_blocks_count > 0:
                stage_slice = slice(1, None)
                avg_count = steady_blocks_count
                avg_label = "steady block"
            else:
                stage_slice = slice(None)
                avg_count = len(block_total)
                avg_label = "block (warmup only)"
            avg_gen_ms = sum(block_gen[stage_slice]) / avg_count * 1000.0
            avg_decode_ms = sum(block_decode[stage_slice]) / avg_count * 1000.0
            avg_total_ms = sum(block_total[stage_slice]) / avg_count * 1000.0
            logger.info(
                f"  avg per {avg_label}: "
                f"gen={avg_gen_ms:.1f}ms decode={avg_decode_ms:.1f}ms "
                f"total={avg_total_ms:.1f}ms"
            )

        if not self.is_rank_zero:
            return None

        full_video_b3thw = torch.cat(final_video_pixels, dim=2)
        full_video_thw3_uint8 = (
            ((1.0 + full_video_b3thw[0].permute(1, 2, 3, 0)) / 2 * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        save_fp = (
            output_video_path[:-4]
            if output_video_path.endswith(".mp4")
            else output_video_path
        )
        mediapy.write_video(save_fp + ".mp4", full_video_thw3_uint8, fps=cfg.fps)
        logger.info(f"Saved video to {save_fp}.mp4")

        annotated = _annotate_frame_numbers(full_video_thw3_uint8)
        mediapy.write_video(save_fp + "_annotated.mp4", annotated, fps=cfg.fps)
        logger.info(f"Saved annotated video to {save_fp}_annotated.mp4")

        full_float = full_video_b3thw[0].detach().float().cpu().numpy()
        np.save(save_fp + ".npy", full_float)
        logger.info(
            f"Saved float video tensor to {save_fp}.npy "
            f"(shape {full_float.shape}, [-1,1] CTHW)"
        )

        latents_stacked = torch.stack(latents_per_block, dim=0)
        np.save(save_fp + "_latents.npy", latents_stacked.numpy())
        logger.info(
            f"Saved latents to {save_fp}_latents.npy "
            f"(shape {tuple(latents_stacked.shape)}; dim 0 is outer block index)"
        )

        if cfg.save_comparison:
            try:
                original_video = mediapy.read_video(input_video_path)
                min_len = min(len(original_video), full_video_thw3_uint8.shape[0])
                original_video = original_video[:min_len]
                predicted = full_video_thw3_uint8[:min_len]
                if original_video.shape[1:3] != predicted.shape[1:3]:
                    from PIL import Image  # noqa: PLC0415

                    target_h, target_w = predicted.shape[1], predicted.shape[2]
                    resized: list[np.ndarray] = []
                    for f in original_video:
                        resized.append(
                            np.array(
                                Image.fromarray(f).resize(
                                    (target_w, target_h), Image.BILINEAR
                                )
                            )
                        )
                    original_video = np.stack(resized, axis=0)
                comparison = np.concatenate(
                    [
                        _annotate_frame_numbers(original_video),
                        _annotate_frame_numbers(predicted),
                    ],
                    axis=2,
                )
                comp_path = save_fp + "_comparison.mp4"
                mediapy.write_video(comp_path, comparison, fps=cfg.fps)
                logger.info(f"Saved comparison video to {comp_path}")
            except Exception as e:  # noqa: BLE001 -- comparison is best-effort
                logger.warning(f"Failed to create comparison video: {e}")

        frames_generated = total_generated_latents * WAN_TCR
        return {
            "total_time": total_time,
            "encode_time": t_encode,
            "generation_time": t_gen_total,
            "decode_time": t_decode_total,
            "frames_generated": float(frames_generated),
            "warmup_time": warmup_time,
            "steady_time": steady_blocks_total,
            "steady_frames": float(steady_frames),
        }


__all__ = [
    "CosmoshRunner",
    "CosmoshRunnerConfig",
]


## Helpers


def _annotate_frame_numbers(frames_uint8: np.ndarray) -> np.ndarray:
    """Burn ``Frame N`` labels into ``[T, H, W, 3]`` uint8 frames (returns a copy)."""
    import cv2  # noqa: PLC0415

    out = frames_uint8.copy()
    T, H, W = out.shape[0], out.shape[1], out.shape[2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(np.clip(min(H, W) / 720.0, 0.5, 1.4))
    thickness = max(1, int(round(scale * 2)))
    margin = int(8 + 6 * scale)
    for i in range(T):
        label = f"Frame {i}"
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        x0, y0 = margin, margin + th
        cv2.rectangle(
            out[i],
            (x0 - 4, y0 - th - 4),
            (x0 + tw + 4, y0 + baseline + 4),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            out[i],
            label,
            (x0, y0),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out
