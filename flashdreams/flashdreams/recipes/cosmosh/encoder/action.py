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

"""Per-AR-step action encoder for CosmosH.

The encoder is stateless w.r.t. learned parameters: the full action
trajectory for the rollout is stashed on the per-rollout cache at
``initialize_autoregressive_cache``.

Slicing semantics match the upstream
``ActionVideo2WorldModelTrigflowSelfForcingDMD2.generate_streaming_video``
loop:

- ``ar_idx == 0`` is a **conditional prefill** step (the first frame's VAE
  latent seeds the K/V cache at zero noise). The reference passes
  ``action=None`` here so the action MLPs are skipped — there is no
  motion to drive yet. ``forward`` returns ``None``.
- ``ar_idx >= 1`` is a **generation** step. The reference uses
  ``action[(t_idx - start_idx) * A : (t_idx - start_idx + 1) * A]`` with
  ``start_idx == 1``; we mirror it as ``actions[(ar_idx - 1) * A : ar_idx * A]``.

The action MLPs (``action_embedder_B_D`` / ``B_3D``) live inside the
network so the upstream checkpoint loads with no key remapping; this
encoder only handles slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.encoder import Encoder, StreamingEncoderCache


@dataclass(kw_only=True)
class ActionEncoderCache(StreamingEncoderCache):
    """Per-rollout cache holding the full action trajectory."""

    actions: Tensor
    """Full-trajectory actions, shape ``[B, T_actions, action_dim]``.
    ``T_actions`` should be at least ``num_ar_steps * num_action_per_latent_frame``."""

    num_action_per_latent_frame: int
    """Number of raw actions consumed per AR step (= VAE temporal compression)."""


@dataclass(kw_only=True)
class ActionEncoderConfig(InstantiateConfig):
    """Config for the per-AR-step action encoder."""

    _target: type["ActionEncoder"] = field(default_factory=lambda: ActionEncoder)

    num_action_per_latent_frame: int = 4
    """Default number of raw actions per AR step. Pinned to the Wan2.1 VAE
    temporal compression ratio."""


class ActionEncoder(Encoder):
    """Slices the per-AR-step action chunk out of the cached trajectory.

    Pipeline call shape:

        cache = encoder.initialize_autoregressive_cache(actions=trajectory)
        chunk = encoder(input=None, autoregressive_index=0, cache=cache)
        # chunk: None  (AR 0 is the prefill / conditional step)
        chunk = encoder(input=None, autoregressive_index=1, cache=cache)
        # chunk: [B, num_action_per_latent_frame, action_dim]  (= actions[0:A])
    """

    def __init__(self, config: ActionEncoderConfig) -> None:
        super().__init__(config)
        self.config: ActionEncoderConfig = config

    def initialize_autoregressive_cache(
        self,
        *,
        actions: Tensor,
    ) -> ActionEncoderCache:
        """Stash the full action trajectory for this rollout.

        Args:
            actions: ``[B, T_actions, action_dim]`` raw actions for the entire
                rollout. Padding to a fixed ``action_dim`` (e.g. 44 for the
                CosmosH checkpoint) is the caller's responsibility.
        """
        assert actions.ndim == 3, (
            f"actions must be [B, T_actions, action_dim], got shape {tuple(actions.shape)}"
        )
        return ActionEncoderCache(
            actions=actions,
            num_action_per_latent_frame=self.config.num_action_per_latent_frame,
        )

    def forward(
        self,
        input: Tensor | None,
        autoregressive_index: int = 0,
        cache: ActionEncoderCache | None = None,
    ) -> Tensor | None:
        """Return the action chunk for AR step ``autoregressive_index``.

        AR 0 is the conditional-frame prefill: matches the upstream's
        ``action=None`` semantics so the network skips the action MLPs and
        the K/V cache is seeded purely from the clean image latent.

        Args:
            input: Ignored; the action source is the cached trajectory.
            autoregressive_index: 0-based AR step index. ``0`` returns
                ``None``; ``>=1`` returns ``actions[(ar-1)*A : ar*A]``.
            cache: Per-rollout cache populated by
                :meth:`initialize_autoregressive_cache`.

        Returns:
            ``None`` at AR step 0; otherwise the
            ``[B, num_action_per_latent_frame, action_dim]`` slice.
        """
        del input
        assert cache is not None, "ActionEncoder requires a cache for slicing"
        if autoregressive_index == 0:
            return None
        A = cache.num_action_per_latent_frame
        start = (autoregressive_index - 1) * A
        stop = start + A
        assert stop <= cache.actions.shape[1], (
            f"AR step {autoregressive_index} requires actions[{start}:{stop}], "
            f"but trajectory has only {cache.actions.shape[1]} entries."
        )
        return cache.actions[:, start:stop, :]
