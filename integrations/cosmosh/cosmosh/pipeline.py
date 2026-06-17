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

Mirrors the structure of :class:`omnidreams.pipeline.OmnidreamsPipeline`: the
pipeline OWNS its VAE first-frame image encoder and its decoder, so callers feed
raw inputs and get decoded pixels back. The only model differences from
Omnidreams are that CosmosH is **action-conditioned** (the per-AR-step action
chunk is the analogue of Omnidreams' HDMap control, passed per ``generate()``
call) and **single-view**, and that its CR1 text context arrives as a
precomputed embedding tensor rather than from a live text encoder.

Lifecycle::

    pipeline: CosmoshPipeline = config.setup().to("cuda").eval()
    cache = pipeline.initialize_cache(
        text_embeddings=cr1_embeddings,   # precomputed CR1 context [B, L, D]
        image=first_frame_pixels,         # [B, 1, 3, H, W] in [-1, 1]
    )
    pixels0 = pipeline.generate(0, cache, actions=None)        # AR-0 prefill
    pipeline.finalize(0, cache)
    pixels1 = pipeline.generate(1, cache, actions=chunk_1)     # [B, A*L, action_dim]
    pipeline.finalize(1, cache)
"""

from __future__ import annotations

import gc
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
from flashdreams.recipes.taehv import TeahvVAEDecoder
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAEDecoder,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)

from cosmosh.encoder.action import (
    ActionEncoderCache,
    ActionEncoderConfig,
    num_generated_frames,
)
from cosmosh.transformer import (
    CosmosHTransformer,
    CosmosHTransformerCache,
)

CosmoshPipelineCache: TypeAlias = StreamInferencePipelineCache[
    ActionEncoderCache,
    CosmosHTransformerCache,
    StreamingDecoderCache,
]


@dataclass(kw_only=True)
class CosmoshPipelineConfig(StreamInferencePipelineConfig):
    """Config for the CosmosH pipeline.

    The base ``encoder`` slot holds the per-AR-step :class:`ActionEncoderConfig`;
    the base ``decoder`` slot holds the Wan / TAEHV VAE decoder. On top of those
    the pipeline owns a one-shot first-frame VAE image encoder via
    :attr:`image_encoder`.
    """

    _target: type["CosmoshPipeline"] = field(default_factory=lambda: CosmoshPipeline)

    image_encoder: WanVAEEncoderConfig | None = field(
        default_factory=WanVAEEncoderConfig
    )
    """One-shot Wan VAE first-frame encoder. Pin its checkpoint to the VAE the
    network was trained against. ``None`` skips loading it (use
    ``initialize_cache_from_embeddings`` with a precomputed image latent)."""


class CosmoshPipeline(
    StreamInferencePipeline[
        ActionEncoderCache,
        CosmosHTransformerCache,
        StreamingDecoderCache,
    ]
):
    """CosmosH streaming inference pipeline (single-view, action-conditioned)."""

    image_encoder: WanVAEEncoder | None

    def __init__(self, config: CosmoshPipelineConfig) -> None:
        super().__init__(config)

        self.image_encoder = (
            config.image_encoder.setup() if config.image_encoder is not None else None
        )

        assert self.encoder is not None, (
            "CosmoshPipeline requires the per-AR-step ActionEncoder; set "
            "CosmoshPipelineConfig.encoder = ActionEncoderConfig(...)."
        )
        encoder_cfg = self.config.encoder
        assert isinstance(encoder_cfg, ActionEncoderConfig), (
            "CosmoshPipeline expects encoder to be an ActionEncoderConfig, got "
            f"{type(encoder_cfg).__name__}."
        )

        transformer = self.diffusion_model.transformer
        assert isinstance(transformer, CosmosHTransformer), (
            "CosmoshPipeline requires a CosmosH transformer; "
            f"got {type(transformer).__name__}."
        )

        decoder = self.decoder
        assert isinstance(decoder, (WanVAEDecoder, TeahvVAEDecoder)), (
            "CosmoshPipeline requires a Wan or TAEHV VAE decoder; "
            f"got {type(decoder).__name__}."
        )

        # Pin the per-step action count so a mismatch between the encoder and
        # the network's MLP fails loudly at construction, not at first forward.
        assert (
            encoder_cfg.num_action_per_latent_frame
            == transformer.config.network.num_action_per_latent_frame
        ), (
            "ActionEncoderConfig.num_action_per_latent_frame "
            f"({encoder_cfg.num_action_per_latent_frame}) must match "
            "network.num_action_per_latent_frame "
            f"({transformer.config.network.num_action_per_latent_frame})."
        )
        # The encoder validates ``latent_frames_per_step`` frames' worth of
        # actions per AR step; the transformer generates exactly ``_pT`` latent
        # frames per step. Pin them together at construction.
        assert encoder_cfg.latent_frames_per_step == transformer.config._pT, (
            "ActionEncoderConfig.latent_frames_per_step "
            f"({encoder_cfg.latent_frames_per_step}) must match the "
            f"transformer's _pT ({transformer.config._pT}) "
            f"(len_t={transformer.config.len_t}, "
            f"patch_temporal={transformer.config.network.patch_temporal})."
        )

        self._len_t_latent: int = transformer.config._pT
        self._num_action_per_latent_frame: int = (
            transformer.config.network.num_action_per_latent_frame
        )

    @property
    def device(self) -> torch.device:
        return self.diffusion_model.device

    @property
    def _use_negative_text(self) -> bool:
        return self.diffusion_model.transformer.config.requires_negative_text_embeddings

    @torch.no_grad()
    def initialize_cache(
        self,
        text_embeddings: Tensor,
        image: Tensor,
        negative_text_embeddings: Tensor | None = None,
    ) -> CosmoshPipelineCache:
        """Initialize the per-rollout cache from precomputed CR1 text + raw image.

        Args:
            text_embeddings: ``[B, L_ctx, D_ctx]`` precomputed CR1 / Cosmos-Reason1
                context (projected later by the network's ``crossattn_proj``).
            image: First-frame pixels ``[B, 1, 3, H, W]`` in ``[-1, 1]``. ``H``/``W``
                must equal latent ``height``/``width`` times the VAE spatial
                compression ratio; the owned image encoder turns it into the
                ``[B, 1, C_lat, H_lat, W_lat]`` first-frame latent.
            negative_text_embeddings: Optional CFG-uncond context. Required when
                the transformer config has ``guidance_scale > 1`` (inert by
                default for CosmosH, which runs CFG off).
        """
        assert self.image_encoder is not None, (
            "initialize_cache(image=) requires image_encoder to be loaded. It is "
            "None either because the config set it to None or because "
            "release_oneshot_encoders() has been called. If you have a "
            "precomputed image latent, use initialize_cache_from_embeddings()."
        )
        image = image.to(device=self.device)
        image_embeddings = self.image_encoder(input=image)
        return self.initialize_cache_from_embeddings(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            negative_text_embeddings=negative_text_embeddings,
        )

    @torch.no_grad()
    def initialize_cache_from_embeddings(
        self,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        negative_text_embeddings: Tensor | None = None,
    ) -> CosmoshPipelineCache:
        """Initialize the per-rollout cache from precomputed embeddings.

        Args:
            text_embeddings: ``[B, L_ctx, D_ctx]`` CR1 context. Moved to
                ``self.device``.
            image_embeddings: ``[B, 1, C_lat, H_lat, W_lat]`` first-frame VAE
                latent (``C_lat = network.in_channels``;
                ``(H_lat, W_lat) = (transformer.config.height, .width)``).
            negative_text_embeddings: Optional CFG-uncond context.
        """
        text_embeddings = text_embeddings.to(device=self.device)
        image_embeddings = image_embeddings.to(device=self.device)
        if negative_text_embeddings is not None:
            negative_text_embeddings = negative_text_embeddings.to(device=self.device)

        transformer_context: dict[str, Tensor] = {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
        }
        if negative_text_embeddings is not None:
            transformer_context["negative_text_embeddings"] = negative_text_embeddings

        # Actions flow per ``generate()`` call (the per-AR-step control), so the
        # encoder cache is stateless — no encoder_context needed here.
        return super().initialize_cache(transformer_context=transformer_context)

    @torch.no_grad()
    def precompute_embeddings(
        self,
        text_embeddings: Tensor,
        image: Tensor,
    ) -> dict[str, Tensor | None]:
        """Run only the one-shot image encoder; return embeddings on CPU.

        Pair with :meth:`initialize_cache_from_embeddings`: save the returned
        dict, build a pipeline with ``image_encoder=None``, and rehydrate. The
        CR1 ``text_embeddings`` pass through unchanged (CosmosH has no live text
        encoder).

        Returns:
            ``{"text_embeddings": [B, L, D], "image_embeddings":
            [B, 1, Cl, Hl, Wl], "negative_text_embeddings": None}`` on CPU.
        """
        assert self.image_encoder is not None, (
            "precompute_embeddings requires image_encoder to be loaded; build "
            "the pipeline with image_encoder non-None."
        )
        image = image.to(device=self.device)
        image_embeddings = self.image_encoder(input=image)
        return {
            "text_embeddings": text_embeddings.cpu(),
            "image_embeddings": image_embeddings.cpu(),
            "negative_text_embeddings": None,
        }

    def release_oneshot_encoders(self) -> None:
        """Free the one-shot first-frame image encoder.

        Idempotent. Only safe for one-shot pipeline lifetimes (a single
        rollout / demo); long-lived hosts that re-encode a new conditional
        frame per scene must not call this.
        """
        self.image_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: CosmoshPipelineCache,
        actions: Tensor | None = None,
    ) -> Tensor:
        """Generate one decoded video chunk for this AR step.

        Args:
            autoregressive_index: AR step index (0-based).
            cache: Per-rollout cache from ``initialize_cache``.
            actions: Per-AR-step action chunk ``[B, n_gen * A, action_dim]`` for
                the ``n_gen = get_num_actions(ar_idx) // A`` generated frames, or
                ``None`` when the step drives no generated frame (AR step 0 with
                ``len_t == 1``). Use :meth:`get_num_actions` to size it.

        Returns:
            Decoded video chunk ``[B, T, 3, H, W]`` in ``[-1, 1]`` with
            ``T == get_num_frames(autoregressive_index)``.
        """
        return super().generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=actions,
        )

    def get_num_actions(self, autoregressive_index: int) -> int:
        """Raw actions consumed at this AR step (``n_gen * num_action_per_latent_frame``).

        ``0`` at AR step 0 when ``len_t == 1`` (pure conditional prefill, no
        actions); callers pass ``actions=None`` in that case.
        """
        n_gen = num_generated_frames(autoregressive_index, self._len_t_latent)
        return n_gen * self._num_action_per_latent_frame

    def get_num_frames(self, autoregressive_index: int) -> int:
        """Number of decoded pixel frames produced at this AR step."""
        assert self.decoder is not None
        return self.decoder.get_output_temporal_size(
            autoregressive_index, self._len_t_latent
        )
