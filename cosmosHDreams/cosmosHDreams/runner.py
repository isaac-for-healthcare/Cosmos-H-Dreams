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

"""Action-conditioned streaming Video2World runner for Cosmos-H-Dreams."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from flashdreams.infra.runner import Runner, RunnerConfig
from cosmosHDreams.pipeline import CosmoshPipeline, CosmoshPipelineCache
from cosmosHDreams.transformer import CosmosHTransformer, CosmosHTransformerConfig
from cosmosHDreams.utils import (
    load_comparison_frames,
    load_conditional_frame,
    load_cr1_text_embeddings,
    pad_actions,
    pixel_frame_to_neg1_pos1,
    resolve_manifest_input_path,
)

# Wan2.1 VAE spatial compression ratio. The output pixel resolution must be a
# multiple of it so the latent grid is integer-sized.
WAN_SCR = 8

DEFAULT_RUNNER_RESOLUTION = "288,512"


@dataclass(kw_only=True)
class CosmoshRunnerConfig(RunnerConfig):
    """Runner config for any CosmosH variant.

    The VAE first-frame encoder + decoder now live inside the pipeline (see
    :class:`cosmosHDreams.pipeline.CosmoshPipeline`), so the runner only carries the
    user-facing I/O. ``len_t`` is a per-variant pipeline property (the
    ``cosmosHDreams-chunk{1,2,3}-*`` slugs), not a runner knob.
    """

    _target: type = field(default_factory=lambda: CosmoshRunner)

    input_json: Path | None = None
    """JSON listing per-entry inputs: exactly one of ``input``,
    ``input_video``, or ``input_image`` (still image or video; type is
    inferred from the file extension), plus ``input_action`` and
    ``output_video`` (and optional per-entry ``start_frame_idx`` /
    ``resolution`` overrides). Required at ``run()`` time."""

    cr1_embeddings_path: Path | None = None
    """Path to a precomputed CR1 text-embeddings ``.pt`` file. Required at
    ``run()`` time."""

    root_dir: Path = Path("")
    """Root directory manifest input paths and ``input_action`` are resolved
    against. Empty means absolute paths in the JSON."""

    total_blocks: int = 20
    """Maximum number of AR steps to attempt per entry. The loop stops early
    once the action stream is consumed."""

    start_frame_idx: int = 0
    """Default index of the input-video frame used as the conditional first
    frame. Per-entry ``start_frame_idx`` in the JSON wins."""

    fps: float = 10.0
    """Output MP4 frame rate."""

    save_comparison: bool = False
    """Also write a side-by-side comparison MP4 (input on left, predicted
    on right)."""

    resolution: str = DEFAULT_RUNNER_RESOLUTION
    """Output resolution as ``"H,W"`` in pixels (default ``"288,512"``). Both
    dims must be multiples of 8 (the Wan2.1 VAE spatial compression ratio)."""


class CosmoshRunner(Runner[CosmoshRunnerConfig, CosmoshPipeline]):
    """Streaming action-conditioned I2V driver for CosmosH."""

    config: CosmoshRunnerConfig

    def __init__(self, config: CosmoshRunnerConfig) -> None:
        # The pipeline's locked latent dims are baked at ``setup()`` time (RoPE
        # tables + KV-cache shape), so a ``--resolution`` override has to land on
        # the transformer config *before* the base ``Runner.__init__`` builds the
        # pipeline. ``resolution`` defaults to 288x512.
        try:
            h_str, w_str = config.resolution.split(",")
            new_h, new_w = int(h_str), int(w_str)
        except ValueError as exc:
            raise ValueError(
                f"--resolution must be 'H,W' (e.g. '288,512'); got "
                f"{config.resolution!r}."
            ) from exc
        assert new_h % WAN_SCR == 0 and new_w % WAN_SCR == 0, (
            f"--resolution dims must be multiples of {WAN_SCR} "
            f"(Wan2.1 VAE spatial ratio); got {new_h}x{new_w}."
        )
        tcfg = config.pipeline.diffusion_model.transformer
        assert isinstance(tcfg, CosmosHTransformerConfig)
        if (tcfg.height, tcfg.width) != (new_h // WAN_SCR, new_w // WAN_SCR):
            tcfg.height = new_h // WAN_SCR
            tcfg.width = new_w // WAN_SCR
            # ``derive_config`` / direct setattr above doesn't re-run
            # ``__post_init__``, so refresh ``_pT / _pH / _pW / _steady_ar_idx``
            # before ``setup()`` builds RoPE tables + the patchifier. Without
            # this the VAE encoder produces a latent at the new resolution while
            # the DiT stays sized for the literal's default.
            tcfg.__post_init__()

        super().__init__(config)

    @torch.inference_mode()
    def run(self) -> None:
        cfg = self.config
        assert cfg.input_json is not None, (
            "CosmoshRunner requires --input_json (a JSON list of entries with "
            "exactly one of input / input_video / input_image, plus "
            "input_action and output_video)."
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
            logger.info(f"  runner_name: {cfg.runner_name}")
            logger.info(f"  input_json: {cfg.input_json}")
            logger.info(f"  total_blocks (max AR steps): {cfg.total_blocks}")
            logger.info(f"  output_dir: {cfg.output_dir}")
            logger.info(
                f"  resolution: {Ht}x{Wt} pixels (latent {tcfg.height}x{tcfg.width}), "
                f"len_t={tcfg.len_t}"
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
            try:
                input_path = resolve_manifest_input_path(entry)
            except ValueError as exc:
                if self.is_rank_zero:
                    logger.warning(f"Entry #{idx} invalid input fields: {exc}")
                continue
            if "input_action" not in entry or "output_video" not in entry:
                if self.is_rank_zero:
                    logger.warning(
                        f"Entry #{idx} missing required keys (input_action, "
                        "output_video). Skipping."
                    )
                continue
            if self.is_rank_zero:
                logger.info("-" * 60)
                logger.info(f"Entry {idx + 1}/{len(entries)}: {input_path}")
            stats = self._run_entry(
                entry,
                input_path=input_path,
                text_embeddings_cpu=text_embeddings_cpu,
            )
            if stats is not None:
                all_stats.append(stats)

        if all_stats and self.is_rank_zero:
            total_frames = sum(int(s["frames_generated"]) for s in all_stats)
            total_time = sum(s["total_time"] for s in all_stats)
            cold_fps = total_frames / total_time if total_time > 0 else 0.0
            logger.info("=" * 60)
            logger.info(
                f"DONE: {len(all_stats)} entries, {total_frames} generated frames "
                f"in {total_time:.2f}s ({cold_fps:.2f} FPS incl. compile + capture)"
            )
            logger.info("=" * 60)

    @torch.inference_mode()
    def _run_entry(
        self,
        entry: dict,
        *,
        input_path: str,
        text_embeddings_cpu: torch.Tensor,
    ) -> dict[str, float] | None:
        """Run one input_json entry end-to-end. Returns timing stats."""
        cfg = self.config
        input_media_path = str(Path(cfg.root_dir) / input_path)
        input_action_path = str(Path(cfg.root_dir) / entry["input_action"])
        output_video_path = entry["output_video"]
        # if not Path(output_video_path).is_absolute():
        #     output_video_path = str(cfg.output_dir / output_video_path)
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

        transformer = self.pipeline.diffusion_model.transformer
        assert isinstance(transformer, CosmosHTransformer)
        tcfg: CosmosHTransformerConfig = transformer.config
        device = self.pipeline.device
        dtype = tcfg.dtype
        Ht, Wt = tcfg.height * WAN_SCR, tcfg.width * WAN_SCR

        if (
            "resolution" in entry
            and isinstance(entry["resolution"], list)
            and len(entry["resolution"]) == 2
        ):
            entry_h, entry_w = int(entry["resolution"][0]), int(entry["resolution"][1])
            assert (entry_h, entry_w) == (Ht, Wt), (
                f"per-entry resolution {(entry_h, entry_w)} disagrees with the "
                f"pipeline's locked ({Ht}, {Wt}); override --resolution instead."
            )

        start_frame_idx = int(entry.get("start_frame_idx", cfg.start_frame_idx))
        cond_frame = load_conditional_frame(input_media_path, start_frame_idx)
        if (cond_frame.shape[0], cond_frame.shape[1]) != (Ht, Wt):
            import mediapy  # noqa: PLC0415 -- under the ``runners`` extras

            cond_frame = mediapy.resize_image(cond_frame, (Ht, Wt))
        cond_pixels = pixel_frame_to_neg1_pos1(
            cond_frame, device=device, dtype=dtype
        )  # [1, 1, 3, H, W]

        actions_np = np.load(input_action_path)
        actions_np = pad_actions(actions_np, target_dim=tcfg.network.action_dim)
        all_actions = (
            torch.from_numpy(actions_np).to(device=device, dtype=dtype).unsqueeze(0)
        )  # [1, T_actions, action_dim]
        n_actions = all_actions.shape[1]

        # Initialize the cache once (encodes the conditional frame inside the
        # pipeline) — no per-step re-encoding, no external decoder cache.
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_start = time.perf_counter()
        cache: CosmoshPipelineCache = self.pipeline.initialize_cache(
            text_embeddings=text_embeddings_cpu.to(device=device, dtype=dtype),
            image=cond_pixels,
        )

        # Prepend the real conditional frame; AR step 0 re-decodes it (1 pixel
        # frame for len_t==1) and we drop that reconstruction below.
        frames: list[torch.Tensor] = [cond_pixels]  # each [1, T, 3, H, W]
        offset = 0
        ar_steps = 0
        for ar_idx in range(cfg.total_blocks):
            need = self.pipeline.get_num_actions(ar_idx)
            if offset + need > n_actions:
                break
            chunk = all_actions[:, offset : offset + need] if need > 0 else None
            pixels = self.pipeline.generate(ar_idx, cache, actions=chunk)
            self.pipeline.finalize(ar_idx, cache)
            offset += need
            ar_steps += 1
            # AR-0's first decoded frame is the VAE reconstruction of the
            # conditional latent (already shown as the prepended cond frame).
            frames.append(pixels[:, 1:] if ar_idx == 0 else pixels)

        torch.cuda.synchronize() if device.type == "cuda" else None
        total_time = time.perf_counter() - t_start

        if ar_steps == 0:
            if self.is_rank_zero:
                logger.warning(
                    f"Entry {input_media_path}: {n_actions} actions is too few for "
                    f"a single AR step (need {self.pipeline.get_num_actions(1)}). "
                    "Skipping."
                )
            return None

        full_video = torch.cat(frames, dim=1)  # [1, T, 3, H, W]
        frames_generated = full_video.shape[1] - 1  # minus the prepended cond frame

        if self.is_rank_zero:
            logger.info(
                f"Entry {input_media_path}: {ar_steps} AR steps, "
                f"{frames_generated} generated frames, {Ht}x{Wt}, "
                f"{total_time:.2f}s"
            )

        if not self.is_rank_zero:
            return None

        self._write_outputs(
            full_video,
            output_video_path,
            input_media_path,
            start_frame_idx=start_frame_idx,
        )
        return {
            "total_time": total_time,
            "frames_generated": float(frames_generated),
        }

    def _write_outputs(
        self,
        full_video: torch.Tensor,
        output_video_path: str,
        input_media_path: str,
        *,
        start_frame_idx: int,
    ) -> None:
        """Write the MP4 + annotated MP4 + ``.npy`` (and optional comparison)."""
        import mediapy  # noqa: PLC0415

        cfg = self.config
        # [1, T, 3, H, W] -> [T, H, W, 3] uint8 in [0, 255].
        video_thwc = full_video[0].permute(0, 2, 3, 1)
        video_uint8 = (
            ((video_thwc + 1.0) / 2.0 * 255.0)
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
        mediapy.write_video(save_fp + ".mp4", video_uint8, fps=cfg.fps)
        logger.info(f"Saved video to {save_fp}.mp4")

        annotated = _annotate_frame_numbers(video_uint8)
        mediapy.write_video(save_fp + "_annotated.mp4", annotated, fps=cfg.fps)

        # [T, 3, H, W] float in [-1, 1] (CTHW after the transpose below).
        full_float = full_video[0].permute(1, 0, 2, 3).detach().float().cpu().numpy()
        np.save(save_fp + ".npy", full_float)

        if cfg.save_comparison:
            try:
                original_video = load_comparison_frames(
                    input_media_path,
                    start_frame_idx=start_frame_idx,
                    num_frames=video_uint8.shape[0],
                )
                min_len = min(original_video.shape[0], video_uint8.shape[0])
                original_video = original_video[:min_len]
                predicted = video_uint8[:min_len]
                if original_video.shape[1:3] != predicted.shape[1:3]:
                    from PIL import Image  # noqa: PLC0415

                    target_h, target_w = predicted.shape[1], predicted.shape[2]
                    original_video = np.stack(
                        [
                            np.array(
                                Image.fromarray(f).resize(
                                    (target_w, target_h), Image.BILINEAR
                                )
                            )
                            for f in original_video
                        ],
                        axis=0,
                    )
                comparison = np.concatenate(
                    [
                        _annotate_frame_numbers(original_video),
                        _annotate_frame_numbers(predicted),
                    ],
                    axis=2,
                )
                mediapy.write_video(
                    save_fp + "_comparison.mp4", comparison, fps=cfg.fps
                )
                logger.info(f"Saved comparison video to {save_fp}_comparison.mp4")
            except Exception as e:  # noqa: BLE001 -- comparison is best-effort
                logger.warning(f"Failed to create comparison video: {e}")


__all__ = [
    "CosmoshRunner",
    "CosmoshRunnerConfig",
    "DEFAULT_RUNNER_RESOLUTION",
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
