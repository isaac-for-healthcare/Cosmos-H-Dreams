"""Feed recorded .npy actions into CosmoshInferenceRuntime._render_chunk_from_actions.

Debug tool: exercises the interactive flat-AR path with deterministic
.npy inputs instead of live keyboard / VR input.  Useful for comparing
against the offline runner — if the runner produces good video but this
script does not, the bug is in the interactive path; if both fail the
same way, the flat-AR code itself is the culprit.

Action layout: the .npy file is expected to contain normalised actions
in the same layout as the training stats file (PSM1 xyz | PSM1 rot6d |
PSM1 gripper | PSM2 xyz | PSM2 rot6d | PSM2 gripper | …).  Only the
first 20 dims (ACTION_DIM_NORMALISED) are passed to
``_render_chunk_from_actions``; the session pads the remaining dims with
the resting-neutral fill, exactly as the keyboard path does.

Usage:
    python -m cosmosh.webrtc.replay \\
        --input-video  path/to/video.mp4 \\
        --input-action path/to/actions.npy \\
        --output-video path/to/output.mp4 \\
        --stats-path   path/to/stats_cosmos.json \\
        --cr1-embeddings-path path/to/cr1_embeddings.pt \\
        --total-chunks 20

    python -m cosmosh.webrtc.replay \
        --config-name vae_vae \
        --input-video  sf_inference_data/cmr/trajectories/hyst_exp4_test/episodes/episode_007193.mp4 \
        --input-action sf_inference_data/cmr/trajectories/hyst_exp4_test/episodes/episode_007193_actions.npy \
        --output-video outputs_interactive/episode_007193_output.mp4 \
        --stats-path   sf_inference_data/cmr/stats_cosmos-28D-exp1.json \
        --cr1-embeddings-path sf_inference_data/cr1_empty_string_text_embeddings.pt \
        --ckpt-path checkpoints/cmr/hyst/model_ema_bf16_cmr_hyst_exp5.pt \
        --total-chunks 20 --no-compile --len-t 3
"""

from __future__ import annotations
import argparse
import asyncio
import logging
from pathlib import Path

import mediapy
import numpy as np
import torch

from cosmosh.runner import _annotate_frame_numbers
from cosmosh.utils import pad_actions
from cosmosh.webrtc.session import CosmoshInferenceRuntime, CosmoshRuntimeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("cosmosh.replay")


async def run_replay(
    *,
    input_video: str,
    input_action: str,
    output_video: str,
    stats_path: str,
    cr1_embeddings_path: str,
    ckpt_path: str | None,
    config_name: str,
    len_t: int,
    start_frame_idx: int,
    total_chunks: int,
    actions_per_chunk: int,
    fps: float,
    device: str,
    compile_network: bool,
    no_skip_cond_frame: bool,
) -> None:
    config = CosmoshRuntimeConfig(
        config_name=config_name,
        compile_network=compile_network,
        device=device,
        ckpt_path=ckpt_path,
        cr1_embeddings_path=cr1_embeddings_path,
        input_path=input_video,
        stats_path=stats_path,
        start_frame_idx=start_frame_idx,
        actions_per_chunk=actions_per_chunk,
        len_t=len_t,
    )

    LOGGER.info("Initialising CosmoshInferenceRuntime …")
    runtime = CosmoshInferenceRuntime(config=config)
    await runtime.initialize()

    # S = AR steps per outer block (= generate() calls per block).
    # L = latent frames per generate() call (= len_t).
    # A = raw actions per generated latent frame.
    # _inner_steps_per_block was computed for L=1; we recompute S and A here
    # using the actual L so the loop mirrors the runner for any len_t.
    transformer = runtime._pipeline.diffusion_model.transformer
    L = int(transformer.config._pT)
    A = int(transformer.config.network.num_action_per_latent_frame)
    # steps_per_block: each block runs S generate() calls. From the session
    # formula _inner_steps_per_block = 1 + apc//A, apc = (S*L - 1)*A at block
    # 0. For L=1 that gives S directly. The runner always uses steps_per_block=4.
    S = runtime._inner_steps_per_block  # == 1 + apc//A; equals steps_per_block for L=1
    dev = runtime._device
    dtype = runtime._dtype
    LOGGER.info("L=%d  S=%d  A=%d  action_dim=%d", L, S, A, runtime._action_target_dim)

    # Load and pad to the model's full action_dim — same path as the offline runner.
    actions_np = np.load(input_action)
    if actions_np.ndim != 2:
        raise ValueError(f"Expected 2-D actions [T, D], got {actions_np.shape}")
    actions_np = pad_actions(actions_np.astype(np.float32), target_dim=runtime._action_target_dim)

    # Mirror runner's ar_total formula exactly.
    ar_total = min(total_chunks, (actions_np.shape[0] // A + 1) // (S * L))
    if ar_total <= 0:
        raise ValueError(
            f"Not enough actions ({actions_np.shape[0]} rows) for even one block "
            f"(need at least {(S * L - 1) * A} actions, S={S}, L={L}, A={A})."
        )
    total_ar_steps = ar_total * S
    total_generated_latents = total_ar_steps * L - 1  # one prefill latent at AR 0
    total_actions_needed = total_generated_latents * A
    LOGGER.info(
        "%d blocks × %d AR steps × L=%d = %d total AR steps, "
        "%d generated latents, %d actions consumed",
        ar_total, S, L, total_ar_steps, total_generated_latents, total_actions_needed,
    )

    with torch.inference_mode():
        # Encode the conditioning frame once, then offload the encoder.
        image_embeddings = runtime._encoder(input=runtime._cond_pixels)
        runtime._encoder.cpu()
        torch.cuda.empty_cache()

        all_actions = (
            torch.from_numpy(actions_np[:total_actions_needed])
            .to(device=dev, dtype=dtype)
            .unsqueeze(0)  # [1, T_actions, action_dim]
        )

        cache = runtime._pipeline.initialize_cache(
            text_embeddings=runtime._text_embeddings,
            image_embeddings=image_embeddings,
            actions=all_actions,
        )
        decoder_cache = runtime._decoder.initialize_autoregressive_cache()

        final_video_pixels: list[torch.Tensor] = []
        if not no_skip_cond_frame:
            final_video_pixels.append(
                runtime._initial_cond_pixels.permute(0, 2, 1, 3, 4).contiguous()
            )

        latent_buffer: list[torch.Tensor] = []
        for global_ar_idx in range(total_ar_steps):
            block_idx = global_ar_idx // S
            is_last_step = global_ar_idx == total_ar_steps - 1
            is_block_end = (global_ar_idx + 1) % S == 0

            out_5d = runtime._pipeline.generate(global_ar_idx, cache, input=True)
            print("out_5d.shape", out_5d.shape)
            if not is_last_step:
                runtime._pipeline.finalize(global_ar_idx, cache)
            latent_buffer.append(out_5d)

            if is_block_end:
                block_latent = torch.cat(latent_buffer, dim=1)
                latent_buffer = []

                block_pixels = runtime._decoder(
                    input=block_latent, cache=decoder_cache
                ).clamp(-1.0, 1.0)
                block_pixels_b3thw = block_pixels.permute(0, 2, 1, 3, 4).contiguous()

                if block_idx == 0:
                    # AR 0 decoded the conditioning frame — skip it.
                    final_video_pixels.append(block_pixels_b3thw[:, :, 1:])
                else:
                    final_video_pixels.append(block_pixels_b3thw)

                LOGGER.info(
                    "Block %d/%d — pixel shape %s",
                    block_idx + 1, ar_total,
                    tuple(block_pixels_b3thw.shape),
                )

    full_b3thw = torch.cat(final_video_pixels, dim=2)  # [1, 3, T, H, W]
    full_thw3 = (
        ((1.0 + full_b3thw[0].permute(1, 2, 3, 0)) / 2.0 * 255.0)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(output_video, full_thw3, fps=fps)
    LOGGER.info("Saved %d frames to %s", full_thw3.shape[0], output_video)

    annotated = _annotate_frame_numbers(full_thw3)
    stem = output_video[:-4] if output_video.endswith(".mp4") else output_video
    mediapy.write_video(stem + "_annotated.mp4", annotated, fps=fps)
    LOGGER.info("Saved annotated video to %s_annotated.mp4", stem)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feed .npy actions into CosmoshInferenceRuntime for debugging."
    )
    p.add_argument("--input-video", required=True, help="Conditioning video / image path.")
    p.add_argument("--input-action", required=True, help="Path to actions .npy file [T, D].")
    p.add_argument("--output-video", required=True, help="Output MP4 path.")
    p.add_argument("--stats-path", required=True, help="stats_cosmos*.json path.")
    p.add_argument("--cr1-embeddings-path", required=True, help="CR1 text embeddings .pt path.")
    p.add_argument("--ckpt-path", default=None, help="Model checkpoint path (or use default).")
    p.add_argument(
        "--config-name",
        default="lightvae_lighttae",
        help="CosmoshRuntimeConfig config_name (default: lightvae_lighttae).",
    )
    p.add_argument(
        "--len-t",
        type=int,
        default=1,
        help="Latent frames per generate() call. Must match the checkpoint "
             "(chunk1=1, chunk2=2, chunk3=3). Default: 1.",
    )
    p.add_argument("--start-frame-idx", type=int, default=0, help="Conditioning frame index.")
    p.add_argument("--total-chunks", type=int, default=20, help="Max outer blocks to generate.")
    p.add_argument(
        "--actions-per-chunk",
        type=int,
        default=12,
        help="Actions / generated pixel frames per outer block (default: 12).",
    )
    p.add_argument("--fps", type=float, default=10.0, help="Output MP4 frame rate.")
    p.add_argument("--device", default="cuda:0", help="Torch device.")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (faster cold start, slower steady-state).",
    )
    p.add_argument(
        "--no-skip-cond-frame",
        action="store_true",
        help="Do not prepend the conditioning anchor frame to the output video.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_replay(
            input_video=args.input_video,
            input_action=args.input_action,
            output_video=args.output_video,
            stats_path=args.stats_path,
            cr1_embeddings_path=args.cr1_embeddings_path,
            ckpt_path=args.ckpt_path,
            config_name=args.config_name,
            len_t=args.len_t,
            start_frame_idx=args.start_frame_idx,
            total_chunks=args.total_chunks,
            actions_per_chunk=args.actions_per_chunk,
            fps=args.fps,
            device=args.device,
            compile_network=not args.no_compile,
            no_skip_cond_frame=args.no_skip_cond_frame,
        )
    )
