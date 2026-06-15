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

When ``latent_frames_per_step > 1`` each AR step generates several latent
frames at once, so the slice widens to ``n_gen * A`` actions (``n_gen`` is
``latent_frames_per_step - 1`` at the conditional step 0, else
``latent_frames_per_step``). See :meth:`ActionEncoder.forward` for the
cumulative-offset formula. ``latent_frames_per_step == 1`` reproduces the
single-frame semantics above exactly.

The action MLPs (``action_embedder_B_D`` / ``B_3D``) live inside the
network so the upstream checkpoint loads with no key remapping; this
encoder only handles slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.infra.encoder import EncoderConfig, StreamingEncoder, StreamingEncoderCache


@dataclass(kw_only=True)
class ActionEncoderCache(StreamingEncoderCache):
    """Per-rollout cache holding the full action trajectory."""

    actions: Tensor
    """Full-trajectory actions, shape ``[B, T_actions, action_dim]``.
    ``T_actions`` should be at least ``num_ar_steps * num_action_per_latent_frame``."""

    num_action_per_latent_frame: int
    """Number of raw actions consumed per generated latent frame (= VAE temporal
    compression)."""

    latent_frames_per_step: int
    """Latent frames generated per AR step (``= transformer _pT``). AR step 0
    leads with the conditional (image-anchored) frame so it drives
    ``latent_frames_per_step - 1`` frames; later steps drive all of them."""


@dataclass(kw_only=True)
class ActionEncoderConfig(EncoderConfig):
    """Config for the per-AR-step action encoder."""

    _target: type["ActionEncoder"] = field(default_factory=lambda: ActionEncoder)

    num_action_per_latent_frame: int = 4
    """Default number of raw actions per generated latent frame. Pinned to the
    Wan2.1 VAE temporal compression ratio."""

    latent_frames_per_step: int = 1
    """Latent frames generated per AR step (``= transformer _pT``). Must match
    the transformer's ``len_t``; validated in ``CosmoshPipeline.__init__``."""


class ActionEncoder(StreamingEncoder[ActionEncoderCache]):
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
            latent_frames_per_step=self.config.latent_frames_per_step,
        )

    def forward(
        self,
        input: Tensor | None,
        autoregressive_index: int = 0,
        cache: ActionEncoderCache | None = None,
    ) -> Tensor | None:
        """Return the action chunk for AR step ``autoregressive_index``.

        Each AR step generates ``L = latent_frames_per_step`` latent frames.
        AR step 0 leads with the conditional (image-anchored) frame, so it
        drives ``L - 1`` generated frames from ``actions[0 : (L-1)*A]``; every
        later step ``k`` drives ``L`` frames starting at the cumulative
        generated-frame offset ``g = (L-1) + (k-1)*L``, i.e.
        ``actions[g*A : (g+L)*A]``.

        With ``L == 1`` this collapses to the original semantics: AR 0 returns
        ``None`` (pure conditional prefill, matching upstream ``action=None``)
        and AR ``k>=1`` returns ``actions[(k-1)*A : k*A]``.

        Args:
            input: Ignored; the action source is the cached trajectory.
            autoregressive_index: 0-based AR step index.
            cache: Per-rollout cache populated by
                :meth:`initialize_autoregressive_cache`.

        Returns:
            ``None`` when the step drives no generated frame (``L == 1`` at
            AR step 0); otherwise the ``[B, n_gen * A, action_dim]`` slice
            where ``n_gen`` is ``L - 1`` at AR step 0 and ``L`` thereafter.
        """
        del input
        assert cache is not None, "ActionEncoder requires a cache for slicing"
        A = cache.num_action_per_latent_frame
        L = cache.latent_frames_per_step
        if autoregressive_index == 0:
            start_gen = 0
            n_gen = L - 1
        else:
            start_gen = (L - 1) + (autoregressive_index - 1) * L
            n_gen = L
        if n_gen == 0:
            # L == 1 at AR step 0: pure conditional prefill, no action.
            return None
        start = start_gen * A
        stop = start + n_gen * A
        assert stop <= cache.actions.shape[1], (
            f"AR step {autoregressive_index} requires actions[{start}:{stop}], "
            f"but trajectory has only {cache.actions.shape[1]} entries."
        )
        return cache.actions[:, start:stop, :]
