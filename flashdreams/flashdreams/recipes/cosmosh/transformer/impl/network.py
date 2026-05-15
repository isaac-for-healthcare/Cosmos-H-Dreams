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

"""Action-conditioned single-view CosmosH DiT network for streaming inference."""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
)
from flashdreams.infra.config import InstantiateConfig

from .modules import (
    Block,
    BlockCache,
    FinalLayer,
    Mlp,
    PatchEmbed,
    TimestepEmbedding,
    Timesteps,
)


@dataclass
class CosmosHActionDiTNetworkCache:
    """Cache container for all transformer blocks."""

    block_caches: list[BlockCache]
    """Per-block self-attn KV + cross-attn KV cache, indexed by block position."""

    def __getitem__(self, index: int) -> BlockCache:
        return self.block_caches[index]

    def before_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.after_update(chunk_idx)


@dataclass
class CosmosHActionDiTNetworkConfig(InstantiateConfig):
    """Configuration for the CosmosH action-conditioned DiT network."""

    _target: type["CosmosHActionDiTNetwork"] = field(
        default_factory=lambda: CosmosHActionDiTNetwork
    )

    in_channels: int = 16
    """Number of input latent channels before patch embedding."""

    out_channels: int = 16
    """Output latent channels after the final layer."""

    patch_spatial: int = 2
    """Spatial patch size (applied to both H and W)."""

    patch_temporal: int = 1
    """Temporal patch size."""

    model_channels: int = 2048
    """Transformer hidden size (width)."""

    num_blocks: int = 28
    """Number of transformer blocks."""

    num_heads: int = 16
    """Number of attention heads."""

    mlp_ratio: float = 4.0
    """FFN inner-dim multiplier relative to ``model_channels``."""

    concat_padding_mask: bool = True
    """If ``True``, expect a padding mask channel concatenated to the input at training."""

    use_adaln_lora: bool = True
    """If ``True``, factorize AdaLN modulation through a low-rank LoRA path."""

    adaln_lora_dim: int = 256
    """Rank of the AdaLN LoRA factorization when ``use_adaln_lora`` is ``True``."""

    use_crossattn_projection: bool = True
    """If ``True``, project text embeddings through a linear before cross-attention."""

    crossattn_proj_in_channels: int = 100352
    """Input dimension of the optional cross-attention projection."""

    crossattn_emb_channels: int = 1024
    """Cross-attention key/value dimension."""

    timestep_scale: float = 0.001
    """Multiplier applied to raw timestep values before sinusoidal embedding."""

    action_dim: int = 44
    """Per-step action vector size; pinned to 44 for the canonical CosmosH checkpoint."""

    num_action_per_latent_frame: int = 4
    """Number of raw action vectors that map to one latent frame; equal to the
    VAE temporal compression ratio (4 for Wan2.1)."""

    hidden_dim_in_action_embedder: int | None = None
    """Hidden dim inside the action MLPs. ``None`` defaults to ``4 * model_channels``,
    matching the upstream ``ActionChunkConditionedMinimalV1LVGDiT`` default."""


class CosmosHActionDiTNetwork(nn.Module):
    """Action-conditioned single-view DiT with block-causal self-attn + KV cache.

    Action injection mirrors
    ``ActionChunkConditionedMinimalV1LVGDiT.forward`` (upstream): the
    ``action`` chunk is rearranged to ``[B, 1, A*action_dim]``, run through
    two MLP heads, and added to the timestep embedding and AdaLN-LoRA stream
    before the per-block ``LayerNorm``.
    """

    def __init__(self, config: CosmosHActionDiTNetworkConfig):
        super().__init__()
        self.config = config

        # add 1 for the condition mask
        in_channels = config.in_channels + 1
        # optionally add 1 for the padding mask
        if self.config.concat_padding_mask:
            in_channels += 1

        # Patch embedder
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.config.patch_spatial,
            temporal_patch_size=self.config.patch_temporal,
            in_channels=in_channels,
            out_channels=self.config.model_channels,
        )

        # Time embeddings
        self.t_embedder = nn.Sequential(
            Timesteps(self.config.model_channels),
            TimestepEmbedding(
                self.config.model_channels,
                self.config.model_channels,
                use_adaln_lora=self.config.use_adaln_lora,
            ),
        )
        self.t_embedding_norm = nn.RMSNorm(self.config.model_channels, eps=1e-6)

        # Action embedders. Two MLPs: one to model_channels (added to t_emb),
        # one to 3*model_channels (added to AdaLN-LoRA). Hidden dim default
        # 4*model_channels matches upstream so the checkpoint loads 1:1.
        hidden_dim = (
            self.config.hidden_dim_in_action_embedder
            if self.config.hidden_dim_in_action_embedder is not None
            else 4 * self.config.model_channels
        )
        action_in = self.config.action_dim * self.config.num_action_per_latent_frame
        self.action_embedder_B_D = Mlp(
            in_features=action_in,
            hidden_features=hidden_dim,
            out_features=self.config.model_channels,
        )
        self.action_embedder_B_3D = Mlp(
            in_features=action_in,
            hidden_features=hidden_dim,
            out_features=self.config.model_channels * 3,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=self.config.model_channels,
                    context_dim=self.config.crossattn_emb_channels,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    use_adaln_lora=self.config.use_adaln_lora,
                    adaln_lora_dim=self.config.adaln_lora_dim,
                )
                for _ in range(self.config.num_blocks)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=self.config.model_channels,
            spatial_patch_size=self.config.patch_spatial,
            temporal_patch_size=self.config.patch_temporal,
            out_channels=self.config.out_channels,
            use_adaln_lora=self.config.use_adaln_lora,
            adaln_lora_dim=self.config.adaln_lora_dim,
        )

        if self.config.use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(
                    self.config.crossattn_proj_in_channels,
                    self.config.crossattn_emb_channels,
                    bias=True,
                ),
                nn.GELU(),
            )

        self._is_shuffle_op_fused = False
        self._is_padding_mask_fused = False
        self._parameters_updated_after_loading_checkpoint = False

    def set_context_parallel_group(
        self,
        self_attn_group: ProcessGroup | None,
    ) -> None:
        for block in self.blocks:
            assert isinstance(block, Block)
            block.set_context_parallel_group(self_attn_group)

    def _fuse_shuffle_op_into_last_layer(self) -> None:
        """Fuse the channel-shuffle that follows the last linear into its weights.

        The Cosmos patchify pattern is
        ``b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)`` while the
        unpatchify pattern is
        ``b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)``. Folding the
        shuffle into ``final_layer.linear`` removes the explicit ``rearrange``
        from the inference path.
        """
        if self._is_shuffle_op_fused:
            return

        self.final_layer.linear.weight.data = rearrange(
            self.final_layer.linear.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
            c=self.config.out_channels,
        ).contiguous()
        if self.final_layer.linear.bias is not None:
            self.final_layer.linear.bias.data = rearrange(
                self.final_layer.linear.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.config.patch_temporal,
                kh=self.config.patch_spatial,
                kw=self.config.patch_spatial,
                c=self.config.out_channels,
            ).contiguous()

        self._is_shuffle_op_fused = True

    def _fuse_padding_mask_into_patch_embed(self) -> None:
        """Fold the always-zero inference padding mask into ``x_embedder`` in place.

        Training concatenates a ``[B, 1, T, H, W]`` padding mask channel; at
        inference it's always zero, so the matching input channels of
        ``x_embedder`` can be dropped.
        """
        if not self.config.concat_padding_mask:
            return
        if self._is_padding_mask_fused:
            return

        self.x_embedder.in_channels -= 1
        in_channels_to_keep = self.x_embedder.get_linear_in_channels()
        proj_linear = self.x_embedder.proj[1]
        assert isinstance(proj_linear, nn.Linear)
        proj_linear.weight.data = proj_linear.weight.data[
            :, :in_channels_to_keep
        ].contiguous()
        if proj_linear.bias is not None:
            proj_linear.bias.data = proj_linear.bias.data[
                :in_channels_to_keep
            ].contiguous()

        self._is_padding_mask_fused = True

    def update_parameters_after_loading_checkpoint(self) -> None:
        """Fuse load-time-known ops into weights; call once after loading the checkpoint."""
        if self._parameters_updated_after_loading_checkpoint:
            return
        self._fuse_padding_mask_into_patch_embed()
        self._fuse_shuffle_op_into_last_layer()
        self._parameters_updated_after_loading_checkpoint = True

    def patchify_and_maybe_split_cp(
        self,
        x: Tensor,
        process_group: ProcessGroup | None = None,
    ) -> Tensor:
        """Patchify and optionally CP-split the input video tensor.

        Pattern: ``b (t kt) c (h kh) (w kw) -> b (t h w) (c kt kh kw)``.

        Args:
            x: Input video tensor of shape ``[B, T, C, H, W]``.
            process_group: Optional CP process group along the flattened seq dim.

        Returns:
            Patched tensor with shape ``[B, L, D]`` where
            ``L = (T/kt)*(H/kh)*(W/kw)`` and ``D = C*kt*kh*kw``.
        """
        assert x.ndim == 5, f"x must be a 5D tensor [B, T, C, H, W], got shape {x.shape}"

        x = rearrange(
            x,
            "b (t kt) c (h kh) (w kw) -> b (t h w) (c kt kh kw)",
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
        )
        if process_group is not None:
            x = split_inputs_cp(x, seq_dim=-2, cp_group=process_group)
        return x

    def unpatchify_and_maybe_gather_cp(
        self,
        pH: int,
        pW: int,
        x: Tensor,
        process_group: ProcessGroup | None = None,
    ) -> Tensor:
        """Unpatchify and optionally CP-gather the tensor back to video shape.

        Args:
            pH: Number of patches along height (per rank, pre-gather).
            pW: Number of patches along width (per rank, pre-gather).
            x: Tensor with shape ``[B, L, D]``.
            process_group: Optional CP process group along the flattened seq dim.

        Returns:
            Unpatched tensor with shape ``[B, T, C, H, W]``.
        """
        assert x.ndim == 3, f"x must be a 3D tensor [B, L, D], got shape {x.shape}"
        if process_group is not None:
            x = cat_outputs_cp(x, seq_dim=-2, cp_group=process_group)
        x = rearrange(
            x,
            "b (t h w) (c kt kh kw) -> b (t kt) c (h kh) (w kw)",
            h=pH,
            w=pW,
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
        )
        return x

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context: Tensor,
    ) -> CosmosHActionDiTNetworkCache:
        """Build a fresh autoregressive cache for the DiT given the chunk geometry.

        Args:
            chunk_size: Tokens appended to self-attn KV per AR step
                (= ``_pT * _pH * _pW`` per rank).
            window_size: Self-attn rolling-window capacity in tokens.
            sink_size: Self-attn sink-token capacity.
            context: Cross-attention text context, shape ``[B, L_ctx, D_ctx_in]``.
        """
        if self.config.use_crossattn_projection:
            context = self.crossattn_proj(context)

        block_caches: list[BlockCache] = []
        for block in self.blocks:
            assert isinstance(block, Block)
            block_caches.append(
                block.initialize_cache(chunk_size, window_size, sink_size, context)
            )
        return CosmosHActionDiTNetworkCache(block_caches=block_caches)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        rope_freqs: Tensor,
        cache: CosmosHActionDiTNetworkCache,
        condition_video_input_mask: Tensor,
        action: Tensor | None = None,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
    ) -> Tensor:
        """Run the DiT forward.

        Args:
            x: Patchified video tokens of shape ``[B, L, D]``.
            timesteps: Scalar timestep ``[]`` or ``[1]``.
            rope_freqs: RoPE frequencies of shape ``[L, 1, 1, D_head]``.
            cache: Per-block AR cache produced by :meth:`initialize_cache`.
            condition_video_input_mask: Patchified condition mask, same shape as ``x``.
            action: Optional ``[B, A, action_dim]`` tensor of raw actions for the
                current AR step; ``A == num_action_per_latent_frame`` per generated
                latent frame. ``None`` skips action injection (e.g. unconditional
                first-frame prefill).
            current_chunk_idx: Current chunk index for the KV cache.
            eager_mode: ``True`` runs cache pre/post-update inside the forward;
                ``False`` expects the caller to drive ``before_update`` /
                ``after_update`` outside the (graph-captured) network.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "update_parameters_after_loading_checkpoint() must run before forward"
        )

        assert timesteps.ndim == 0, (
            f"timesteps must be a scalar tensor, got shape {tuple(timesteps.shape)}"
        )
        timesteps = timesteps * self.config.timestep_scale

        # Patch embedding: append condition mask channel, then linear.
        x = torch.cat([x, condition_video_input_mask], dim=-1)
        x = self.x_embedder(x)

        # Time embedding (scalar -> [D]).
        t_emb, adaln_lora = self.t_embedder(timesteps)

        # Action injection. Mirrors ActionChunkConditionedMinimalV1LVGDiT.forward:
        # rearrange action to [B, 1, A*action_dim], two MLPs, add to t_emb and adaln_lora,
        # THEN apply t_embedding_norm.
        if action is not None:
            action_flat = rearrange(action, "b a d -> b (a d)")
            action_emb_B_D = self.action_embedder_B_D(action_flat)
            action_emb_B_3D = self.action_embedder_B_3D(action_flat)
            t_emb = t_emb + action_emb_B_D
            if adaln_lora is not None:
                adaln_lora = adaln_lora + action_emb_B_3D

        t_emb = self.t_embedding_norm(t_emb)

        # Broadcast to the batch dim. ``Timesteps`` produces a scalar-style
        # ``(model_channels,)`` embedding from a 0-d timestep, and the action
        # MLP output is ``(B, model_channels)``; ``expand`` works for both.
        B = x.shape[0]
        if t_emb.ndim == 1:
            t_emb = t_emb.unsqueeze(0)
        t_emb = t_emb.expand(B, -1)
        if adaln_lora is not None:
            if adaln_lora.ndim == 1:
                adaln_lora = adaln_lora.unsqueeze(0)
            adaln_lora = adaln_lora.expand(B, -1)

        # In non-eager mode the caller drives before_update/after_update outside
        # the (graph-captured) network forward.
        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x = block(
                x=x,
                emb=t_emb,
                rope_freqs=rope_freqs,
                adaln_lora=adaln_lora,
                cache=cache[block_idx],
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        # Final layer
        x = self.final_layer(x, t_emb, adaln_lora)
        return x
