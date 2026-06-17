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

"""Feed recorded .npy actions through the interactive flat-AR path.

Debug tool: drives :meth:`CosmoshInferenceRuntime._render_chunk_from_actions`
(the exact code path the keyboard / VR session uses) with deterministic .npy
inputs instead of live input. Useful for comparing against the offline
``flashdreams-run cosmosh-*`` runner — if the runner produces good video but
this script does not, the bug is in the interactive path; if both fail the same
way, the shared pipeline is the culprit.

Action layout: the .npy is expected to hold normalised actions in the training
stats layout (PSM1 xyz | PSM1 rot6d | PSM1 gripper | PSM2 xyz | PSM2 rot6d |
PSM2 gripper | ...). Only the first ``ACTION_DIM_NORMALISED`` dims are passed
to ``_render_chunk_from_actions``; the runtime fills the remaining dims with the
resting-neutral fill, exactly as the keyboard path does.

``len_t`` (latent frames per feed-forward step) is selected by the config name:
``cosmosh-chunk1-*`` -> 1, ``cosmosh-chunk2-*`` -> 2, ``cosmosh-chunk3-*`` -> 3.

Usage:
    python -m cosmosh.webrtc.replay \\
        --input-video  sf_inference_data/.../episode_007193.mp4 \\
        --input-action sf_inference_data/.../episode_007193_actions.npy \\
        --output-video outputs_interactive/episode_007193_output.mp4 \\
        --stats-path   sf_inference_data/cmr/stats_cosmos-28D-exp1.json \\
        --cr1-embeddings-path sf_inference_data/cr1_empty_string_text_embeddings.pt \\
        --ckpt-path checkpoints/cmr/hyst/model_ema_bf16_cmr_hyst_exp5.pt \\
        --config-name cosmosh-chunk3-vae-vae \\
        --total-chunks 20 --no-compile

    python -m cosmosh.webrtc.replay \
        --input-video  sf_inference_data/cmr/trajectories/hyst_exp4_test/episodes/episode_007193.mp4 \
        --input-action sf_inference_data/cmr/trajectories/hyst_exp4_test/episodes/episode_007193_actions.npy \
        --output-video outputs_interactive/hyst_exp4_test_output.mp4 \
        --stats-path   sf_inference_data/cmr/stats_cosmos-28D-exp1.json \
        --cr1-embeddings-path sf_inference_data/cr1_empty_string_text_embeddings.pt \
        --ckpt-path checkpoints/cmr/hyst/model_ema_bf16_cmr_hyst_exp5_iter2600.pt \
        --config-name cosmosh-chunk3-vae-vae \
        --total-chunks 20
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
from cosmosh.webrtc.session import CosmoshInferenceRuntime, CosmoshRuntimeConfig
from cosmosh.webrtc.utils import ACTION_DIM_NORMALISED

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
    )

    LOGGER.info("Initialising CosmoshInferenceRuntime (config_name=%s) ...", config_name)
    runtime = CosmoshInferenceRuntime(config=config)
    await runtime.initialize()

    apc = runtime.actions_per_chunk
    # Take the driven prefix; the runtime pads the rest with the resting-neutral
    # fill (same as the keyboard path).
    actions = np.load(input_action)
    if actions.ndim != 2:
        raise ValueError(f"Expected 2-D actions [T, D], got {actions.shape}")
    actions = actions[:, :ACTION_DIM_NORMALISED].astype(np.float32)
    LOGGER.info(
        "Replaying up to %d chunks x %d actions (%d action rows available).",
        total_chunks,
        apc,
        actions.shape[0],
    )

    # Drive the real interactive code path one chunk at a time.
    #
    # IMPORTANT: _render_chunk_from_actions consumes only as many actions as
    # the pipeline's AR steps need within ``apc``.  For len_t > 1 AR step 0
    # needs fewer actions than ``apc`` (e.g. chunk3: step 0 = 8, step >=1 =
    # 12), so a naive ``c * apc`` stride silently drops the leftover actions
    # and feeds all subsequent steps from the wrong offset.  Instead we
    # simulate the inner while-loop to compute the exact consumed count per
    # outer block, and advance a single cursor by that amount.
    frames: list[torch.Tensor] = []
    if not no_skip_cond_frame:
        frames.append(runtime.initial_frame_chunk())  # [1, 3, 1, H, W]
    action_cursor = 0
    for c in range(total_chunks):
        # Simulate _render_chunk_from_actions's while-loop to find how many
        # actions it will actually consume from the current global AR index.
        local_ar_idx = runtime._global_ar_idx
        consumed = 0
        while True:
            need = runtime._pipeline.get_num_actions(local_ar_idx)
            if consumed + need > apc:
                break
            consumed += need
            local_ar_idx += 1
        if consumed == 0 or action_cursor + consumed > actions.shape[0]:
            break
        # Build the apc-shaped block expected by _render_chunk_from_actions:
        # fill the consumed prefix from the recording, leave the rest zeroed
        # (it will never be read by the pipeline).
        block = np.zeros((apc, ACTION_DIM_NORMALISED), dtype=np.float32)
        block[:consumed] = actions[action_cursor : action_cursor + consumed]
        result = runtime._render_chunk_from_actions(block)
        frames.append(result.video_chunk)  # [1, 3, T, H, W] on CPU
        LOGGER.info(
            "Chunk %d/%d -> %d frames (consumed %d/%d actions, cursor %d→%d)",
            c + 1,
            total_chunks,
            result.num_frames,
            consumed,
            apc,
            action_cursor,
            action_cursor + consumed,
        )
        action_cursor += consumed

    full_b3thw = torch.cat(frames, dim=2)  # [1, 3, T, H, W]
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

    await runtime.close()


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
        default="cosmosh-lightvae-lighttae",
        help="COSMOSH_RUNNERS slug (len_t is baked into the chunk variants; "
        "e.g. cosmosh-chunk3-vae-vae). Default: cosmosh-lightvae-lighttae.",
    )
    p.add_argument("--start-frame-idx", type=int, default=0, help="Conditioning frame index.")
    p.add_argument("--total-chunks", type=int, default=20, help="Max chunks to replay.")
    p.add_argument(
        "--actions-per-chunk",
        type=int,
        default=12,
        help="Actions per outer chunk fed to the session (default: 12).",
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
            start_frame_idx=args.start_frame_idx,
            total_chunks=args.total_chunks,
            actions_per_chunk=args.actions_per_chunk,
            fps=args.fps,
            device=args.device,
            compile_network=not args.no_compile,
            no_skip_cond_frame=args.no_skip_cond_frame,
        )
    )
