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

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


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
