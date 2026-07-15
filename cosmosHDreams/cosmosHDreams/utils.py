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

"""Shared I/O and tensor helpers used by the runner and the WebRTC session."""

from __future__ import annotations

from pathlib import Path

import mediapy
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Common single-image extensions ``mediapy.read_image`` understands. Any
# other extension is treated as a video and read via ``mediapy.read_video``.
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

MANIFEST_INPUT_KEYS = ("input", "input_video", "input_image")


def load_cr1_text_embeddings(path: str) -> torch.Tensor:
    """Load CR1 text embeddings and normalise to ``[1, T, D]`` (CPU).

    Accepts a bare ``Tensor`` or a list/tuple whose first element is the
    tensor, and promotes 2-D ``[T, D]`` to a batch of one. Always returns
    ``[1, T, D]`` — multi-batch files are sliced to the first entry so the
    caller never has to guard against a batch-dim mismatch with
    ``image_embeddings`` (always ``B=1``).
    """
    emb = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(emb, (list, tuple)):
        emb = emb[0]
    if not torch.is_tensor(emb):
        raise ValueError(f"CR1 embeddings file is not a torch.Tensor: {path}")
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)
    elif emb.dim() != 3:
        raise ValueError(
            f"CR1 embeddings must be [T, D] or [B, T, D]; got {tuple(emb.shape)}"
        )
    return emb[:1]


def pad_actions(actions_np: np.ndarray, target_dim: int) -> np.ndarray:
    """Right-pad the last dim with zeros so the action width matches ``target_dim``."""
    if actions_np.shape[-1] >= target_dim:
        return actions_np
    pad_width = target_dim - actions_np.shape[-1]
    pad_shape = list(actions_np.shape[:-1]) + [pad_width]
    zeros = np.zeros(pad_shape, dtype=actions_np.dtype)
    return np.concatenate([actions_np, zeros], axis=-1)


def is_image_path(path: str | Path) -> bool:
    """Return ``True`` when ``path`` looks like a still image by extension."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def resolve_manifest_input_path(entry: dict) -> str:
    """Return the single declared manifest input path from an entry.

    Exactly one of ``input``, ``input_video``, or ``input_image`` must be set.
    """
    present = [key for key in MANIFEST_INPUT_KEYS if key in entry]
    if not present:
        raise ValueError(
            f"manifest entry must declare exactly one of {MANIFEST_INPUT_KEYS}; "
            f"got keys {sorted(entry)}"
        )
    if len(present) > 1:
        raise ValueError(
            f"manifest entry must declare only one of {MANIFEST_INPUT_KEYS}; "
            f"got {present}"
        )
    value = entry[present[0]]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"manifest entry field {present[0]!r} must be a non-empty string; "
            f"got {value!r}"
        )
    return value


def load_conditional_frame(path: str, start_frame_idx: int) -> np.ndarray:
    """Load the conditional first frame from either a still image or a video.

    Image files (extension in :data:`IMAGE_SUFFIXES`) are read directly via
    ``mediapy.read_image``; ``start_frame_idx`` is ignored. Anything else is
    treated as a video and indexed at ``start_frame_idx``. Returns ``[H, W, 3]``
    uint8.
    """
    if is_image_path(path):
        frame = mediapy.read_image(path)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"Image at {path} did not yield a [H, W, 3] frame; got shape "
                f"{frame.shape}"
            )
        return frame
    video = mediapy.read_video(path)
    if start_frame_idx < 0 or start_frame_idx >= video.shape[0]:
        raise ValueError(
            f"start_frame_idx={start_frame_idx} is out of range for video "
            f"with {video.shape[0]} frames at {path}"
        )
    return video[start_frame_idx]


def load_comparison_frames(
    path: str,
    *,
    start_frame_idx: int,
    num_frames: int,
) -> np.ndarray:
    """Load left-hand comparison frames. Still images are tiled to ``num_frames``."""
    if is_image_path(path):
        frame = load_conditional_frame(path, start_frame_idx)
        return np.repeat(frame[np.newaxis, ...], num_frames, axis=0)
    video = mediapy.read_video(path)
    return video[:num_frames]


def pixel_frame_to_neg1_pos1(
    frame_uint8: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a single ``[H, W, 3]`` uint8 frame to ``[1, 1, 3, H, W]`` in ``[-1, 1]``.

    Uses the same ``x / 128 - 1`` mapping as the upstream deterministic
    script so VAE-encoder inputs match bit-for-bit.
    """
    frame_uint8 = np.clip(np.round(frame_uint8), 0, 255).astype(np.uint8)
    t = TF.to_tensor(Image.fromarray(frame_uint8))  # [3, H, W] in [0, 1]
    t = t * 255.0 / 128.0 - 1.0
    return t.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
