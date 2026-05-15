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

"""Streaming inference pipeline for CosmosH (single-view, action-conditioned).

Phase 1 surface area:
- ``initialize_cache(text_embeddings, actions)`` for already-encoded text
  embeddings and a full action trajectory; first-frame conditioning, the
  CR1 text encoder, and the Wan VAE encoder/decoder land in Phase 2+.
- ``generate(ar_idx, cache)`` runs one AR step; the action chunk is sliced
  out of the cached trajectory by the ``ActionEncoder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from torch import Tensor

from flashdreams.infra.decoder import StreamingDecoderCache
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)

from flashdreams.recipes.cosmosh.encoder.action import (
    ActionEncoderCache,
    ActionEncoderConfig,
)
from flashdreams.recipes.cosmosh.transformer import (
    CosmosHTransformer,
    CosmosHTransformerCache,
    CosmosHTransformerConfig,
)

CosmoshPipelineCache: TypeAlias = StreamInferencePipelineCache[
    ActionEncoderCache,
    CosmosHTransformerCache,
    StreamingDecoderCache,
]


@dataclass(kw_only=True)
class CosmoshPipelineConfig(StreamInferencePipelineConfig):
    """Config for the CosmosH pipeline.

    Phase 1 expects ``encoder=ActionEncoderConfig(...)`` and
    ``decoder=None``. Phase 2 adds a Wan VAE decoder; Phase 3 adds CR1 text
    encoder + Wan VAE first-frame encoder.
    """

    _target: type["CosmoshPipeline"] = field(default_factory=lambda: CosmoshPipeline)


class CosmoshPipeline(
    StreamInferencePipeline[
        ActionEncoderCache,
        CosmosHTransformerCache,
        StreamingDecoderCache,
    ]
):
    """CosmosH streaming inference pipeline (single-view, action-conditioned).

    Examples:

        pipeline: CosmoshPipeline = config.setup().to("cuda").eval()
        cache = pipeline.initialize_cache(
            text_embeddings=text_embeddings,
            actions=action_trajectory,
        )
        chunk_0 = pipeline.generate(0, cache)
        pipeline.finalize(0, cache)
        chunk_1 = pipeline.generate(1, cache)
        pipeline.finalize(1, cache)
    """

    def __init__(self, config: CosmoshPipelineConfig) -> None:
        super().__init__(config)

        assert self.encoder is not None, (
            "CosmoshPipeline requires the per-AR-step ActionEncoder; set "
            "CosmoshPipelineConfig.encoder = ActionEncoderConfig(...)."
        )

        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, CosmosHTransformer), (
            "CosmoshPipeline requires a CosmosH transformer; "
            f"got {type(transformer).__name__}."
        )
        # Pin the per-step action count so a mismatch between the encoder and
        # the network's MLP fails loudly at construction, not at first forward.
        encoder_cfg = self.config.encoder
        assert isinstance(encoder_cfg, ActionEncoderConfig), (
            "CosmoshPipeline expects encoder to be an ActionEncoderConfig, got "
            f"{type(encoder_cfg).__name__}."
        )
        assert (
            encoder_cfg.num_action_per_latent_frame
            == transformer.config.network.num_action_per_latent_frame
        ), (
            "ActionEncoderConfig.num_action_per_latent_frame "
            f"({encoder_cfg.num_action_per_latent_frame}) must match "
            "network.num_action_per_latent_frame "
            f"({transformer.config.network.num_action_per_latent_frame})."
        )

    @property
    def device(self) -> torch.device:
        return self.diffusion_model.device

    @torch.no_grad()
    def initialize_cache(
        self,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        actions: Tensor,
        negative_text_embeddings: Tensor | None = None,
    ) -> CosmoshPipelineCache:
        """Initialize the per-rollout cache from precomputed embeddings.

        Args:
            text_embeddings: ``[B, L_ctx, D_ctx]`` text embeddings (typically
                CR1 / Cosmos-Reason1, projected later by the network's
                ``crossattn_proj`` if enabled). Moved to ``self.device``.
            image_embeddings: ``[B, 1, C_lat, H_lat, W_lat]`` first-frame VAE
                latent (``C_lat = network.in_channels``,
                ``(H_lat, W_lat) = (transformer.config.height, .width)``).
                Moved to ``self.device``.
            actions: ``[B, T_actions, action_dim]`` full-trajectory raw actions.
                Moved to ``self.device``.
            negative_text_embeddings: Optional CFG-uncond text embeddings.
                Required when the transformer config has ``guidance_scale > 1``.
        """
        text_embeddings = text_embeddings.to(device=self.device)
        image_embeddings = image_embeddings.to(device=self.device)
        actions = actions.to(device=self.device)
        if negative_text_embeddings is not None:
            negative_text_embeddings = negative_text_embeddings.to(device=self.device)

        transformer_context: dict[str, Tensor] = {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
        }
        if negative_text_embeddings is not None:
            transformer_context["negative_text_embeddings"] = negative_text_embeddings

        return super().initialize_cache(
            transformer_context=transformer_context,
            encoder_context={"actions": actions},
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: CosmoshPipelineCache,
    ) -> Tensor:
        """Generate one chunk for this AR step.

        ``input`` is fixed to a sentinel ``True`` so the base pipeline calls
        the action encoder; the encoder ignores its input and slices
        ``cache.encoder_cache.actions``. Returns the unpatchified clean latent
        (no decoder is wired in Phase 1).
        """
        # The base pipeline only invokes the encoder when ``input is not None``.
        # Pass a non-None sentinel so the encoder's slicing path runs; the
        # encoder ignores the value.
        return super().generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=True,
        )
