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

"""Single-view, action-conditioned Cosmos DiT for streaming CosmosH.

Implements per-AR-step action conditioning, first-frame image conditioning,
and the rectified-flow ``predict_flow`` contract consumed by
:class:`flashdreams.infra.diffusion.scheduler.fm.FlowMatchScheduler`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.compile import compile_module
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import (
    TransformerConfig,
    Transformer,
    TransformerAutoregressiveCache,
)

from .impl.network import (
    CosmosHActionDiTNetwork,
    CosmosHActionDiTNetworkCache,
    CosmosHActionDiTNetworkConfig,
)
from .impl.rope import RotaryPositionEmbedding3D


@dataclass(kw_only=True)
class CosmosHTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for the CosmosH transformer."""

    network_cache: CosmosHActionDiTNetworkCache
    """Per-block self-attn KV + cross-attn KV cache (cond branch)."""

    network_cache_uncond: CosmosHActionDiTNetworkCache | None = None
    """Unconditional cache for CFG; ``None`` disables CFG."""

    rope_adapter: RotaryPositionEmbedding3D
    """3D RoPE adapter, advanced via ``shift_t`` each AR step."""

    image_patched: Tensor
    """First-frame VAE latent, T-padded to ``len_t`` and patchified to
    ``[B, L, D]``. Stamped into the noisy / predicted latent at AR step 0
    via :meth:`CosmosHTransformer._maybe_inject_image`."""

    mask_first_block_patched: Tensor
    """Patchified ``[B, L, D_mask]`` condition-video mask used at AR step 0:
    ones on the (single) conditional latent frame, zeros elsewhere."""

    mask_other_blocks_patched: Tensor
    """Patchified all-zero counterpart used at AR step >= 1."""

    autoregressive_index: int = -1
    """AR step index for the chunk currently being processed; ``-1`` before
    the first :meth:`start` call."""

    def start(self, autoregressive_index: int) -> None:
        # Hoist per-block KV pre-update out of the (graph-captured) network
        # forward; predict_flow runs the network with eager_mode=False.
        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        self.network_cache.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.after_update(autoregressive_index)


# Outer-container keys to peel off when ``torch.load`` returns a wrapping
# dict instead of the bare state-dict (DCP exports tend to do one of these).
_KNOWN_OUTER_CONTAINER_KEYS: tuple[str, ...] = (
    "state_dict",
    "model",
    "module",
    "net",
    "ema",
    "net_ema",
    "model_ema",
)
"""Outer dict keys that may wrap the actual ``net`` state-dict in
``torch.save`` outputs from upstream training stacks."""

# Per-key prefixes commonly added by training wrappers (FSDP / DDP / EMA);
# stripped only when *every* key in the state-dict shares the prefix.
_KNOWN_KEY_PREFIXES: tuple[str, ...] = (
    "module.",
    "net.",
    "_orig_mod.",
    "model.",
)
"""Per-key prefixes stripped only when every key starts with one of them."""


def _default_state_dict_transform(
    state_dict: dict[str, Any],
) -> dict[str, Tensor]:
    """Best-effort normalization for upstream-trained checkpoints.

    Two passes:

    1. **Unwrap outer containers** — if the loaded object is a dict whose
       only / dominant entry is one of :data:`_KNOWN_OUTER_CONTAINER_KEYS`
       and the value is itself a dict of tensors, descend into it. Repeats
       until the bare state-dict is found.
    2. **Strip a shared prefix** — if every key starts with one of
       :data:`_KNOWN_KEY_PREFIXES`, strip it. Skipped if keys disagree.

    Override via ``CosmosHTransformerConfig.state_dict_transform`` for
    checkpoints that don't fit either pattern.
    """
    # Pass 1: unwrap outer container dicts up to a fixed depth so a malformed
    # dict can't loop forever.
    for _ in range(4):
        if not isinstance(state_dict, dict):
            break
        if state_dict and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
            break
        # Pick the first matching container key whose value is a dict.
        unwrapped = None
        for k in _KNOWN_OUTER_CONTAINER_KEYS:
            if k in state_dict and isinstance(state_dict[k], dict):
                logger.info(
                    f"State-dict transform: unwrapping outer container key {k!r}"
                )
                unwrapped = state_dict[k]
                break
        if unwrapped is None:
            break
        state_dict = unwrapped

    assert isinstance(state_dict, dict) and all(
        isinstance(v, torch.Tensor) for v in state_dict.values()
    ), (
        "State-dict transform could not reach a bare {str: Tensor} dict; "
        "supply a custom state_dict_transform on CosmosHTransformerConfig."
    )

    # Pass 2: strip a shared per-key prefix.
    for prefix in _KNOWN_KEY_PREFIXES:
        if all(k.startswith(prefix) for k in state_dict):
            logger.info(f"State-dict transform: stripping shared prefix {prefix!r}")
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
            break

    return state_dict


@dataclass(kw_only=True)
class CosmosHTransformerConfig(TransformerConfig):
    """Config for the CosmosH transformer (single-view, action-conditioned)."""

    _target: type["CosmosHTransformer"] = field(
        default_factory=lambda: CosmosHTransformer
    )

    network: CosmosHActionDiTNetworkConfig = field(
        default_factory=CosmosHActionDiTNetworkConfig
    )
    """Backbone DiT network config."""

    dtype: torch.dtype = torch.bfloat16
    """Network parameter / activation dtype."""

    checkpoint_path: str | None = None
    """Optional path to a pretrained checkpoint; ``None`` keeps the random init."""

    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Pre-load state-dict remap. Defaults to a ``net.`` prefix stripper."""

    batch_shape: tuple[int, ...] = (1,)
    """Batch dims of the latent (excluding ``T, HW, D``)."""

    height: int = 88
    """Latent height (post-VAE; for 720p Wan VAE that's 704/8 = 88)."""

    width: int = 160
    """Latent width (post-VAE; for 720p Wan VAE that's 1280/8 = 160)."""

    len_t: int = 1
    """Latent frames per AR chunk. Pinned to ``1`` for the streaming
    self-forcing geometry (one latent frame per AR step)."""

    cp_size: int = 1
    """Size of the THW context-parallel group. Locked to 1 in Phase 1."""

    h_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along H (3.0 @ 720p)."""

    w_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along W."""

    t_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along T."""

    window_size_t: int = 13
    """Self-attention sliding window in pre-patchify frames; matches the
    13-frame CosmosH conditioning window."""

    sink_size_t: int = 0
    """Sink-token count (pre-patchify T)."""

    compile_network: bool = False
    """``torch.compile`` the network (Phase 4)."""

    use_cuda_graph: bool = False
    """Wrap the network in :class:`CUDAGraphWrapper` (Phase 4)."""

    warmup_iters: int = 2
    """Eager calls before CUDA-graph capture."""

    skip_finalize_kv_cache: bool = False
    """Skip the KV cache finalize step."""

    guidance_scale: float = 1.0
    """CFG scale. ``1.0`` disables CFG; ``> 1.0`` requires negative text embeddings."""

    @property
    def requires_negative_text_embeddings(self) -> bool:
        """Whether cache initialization must receive negative text embeddings."""
        return self.guidance_scale > 1.0

    def __post_init__(self) -> None:
        assert (
            self.guidance_scale >= 1.0
        ), f"guidance_scale must be >= 1.0, got {self.guidance_scale}"

        kt = self.network.patch_temporal
        kh = kw = self.network.patch_spatial
        assert (
            self.len_t % kt == 0 and self.height % kh == 0 and self.width % kw == 0
        ), (
            f"({self.len_t}, {self.height}, {self.width}) must be divisible by "
            f"patch_size ({kt}, {kh}, {kw})"
        )
        self._pT = self.len_t // kt
        self._pH = self.height // kh
        self._pW = self.width // kw

        # First AR step whose forward sees a fully-filled, steady-state KV
        # cache. With ``len_t == 1`` this is exactly ``sink_size_t + window_size_t``.
        chunks_total = self.sink_size_t + self.window_size_t
        assert chunks_total % self._pT == 0, (
            f"sink_size_t + window_size_t ({chunks_total}) must be divisible "
            f"by _pT ({self._pT}) so the BlockKVCache fits a whole number of AR chunks."
        )
        self._steady_ar_idx = chunks_total // self._pT


class CosmosHTransformer(Transformer[CosmosHTransformerCache]):
    """Single-view action-conditioned Cosmos DiT as an infra transformer.

    Phase 1: per-AR-step action conditioning, no first-frame conditioning,
    no CFG. CFG plumbing (``network_cache_uncond``) and CUDA-graph wrappers
    are wired but inert under default config.
    """

    network: CosmosHActionDiTNetwork

    def __init__(
        self,
        config: CosmosHTransformerConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config)
        self.config: CosmosHTransformerConfig = config

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            assert config.cp_size == world_size, (
                f"CosmosHTransformerConfig.cp_size ({config.cp_size}) must match "
                f"torch.distributed.get_world_size() ({world_size})"
            )
            self._cp_group = (
                torch.distributed.group.WORLD if config.cp_size > 1 else None
            )
        else:
            assert config.cp_size == 1, (
                f"CosmosHTransformerConfig.cp_size must be 1 in non-distributed "
                f"mode (got {config.cp_size})"
            )
            self._cp_group = None

        self.network = CosmosHActionDiTNetwork(config=config.network)
        if device is not None:
            self.network = self.network.to(device=device)
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(self_attn_group=self._cp_group)

        if config.checkpoint_path is not None:
            transform = config.state_dict_transform or _default_state_dict_transform
            state_dict = load_checkpoint(config.checkpoint_path)
            state_dict = transform(state_dict)
            for k in list(state_dict.keys()):
                if "_extra_state" in k or "pos_embedder" in k or "accum" in k:
                    # _extra_state: Key introduced by TransformerEngine for FP8
                    # pos_embedder: Used in RoPE adapter. Check if this is a problem
                    # accum: Safely discardable. This is just a training log.
                    state_dict.pop(k)
            self.network.load_state_dict(state_dict)
        # Always fuse: the helpers are idempotent and required regardless of
        # whether weights were loaded (the network forward asserts on the flag).
        self.network.update_parameters_after_loading_checkpoint()

        if config.compile_network:
            self.network = compile_module(self.network)

        # CFG / CUDA-graph wrappers are constructed in
        # ``initialize_autoregressive_cache`` so they bind to the freshly
        # allocated KV slots of each new rollout.
        self._use_cuda_graph = config.use_cuda_graph
        self._network_call: CUDAGraphWrapper | CosmosHActionDiTNetwork = (
            CUDAGraphWrapper(self.network, warmup_iters=config.warmup_iters)
            if config.use_cuda_graph
            else self.network
        )
        self._network_call_uncond: CUDAGraphWrapper | CosmosHActionDiTNetwork = (
            CUDAGraphWrapper(self.network, warmup_iters=config.warmup_iters)
            if config.use_cuda_graph
            else self.network
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape ``[*batch_shape, L/cp_size, D]``."""
        cfg = self.config
        kt = cfg.network.patch_temporal
        kh = kw = cfg.network.patch_spatial
        D = cfg.network.in_channels * kt * kh * kw
        L = cfg._pT * cfg._pH * cfg._pW
        return (*cfg.batch_shape, L // cfg.cp_size, D)

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        # Two payloads pass through this hook in CosmosH:
        # - the noisy latent: 5D [B, T, C, H, W] -> patchify + CP-split.
        # - the action chunk (encoder output): 3D [B, A, action_dim] -> passthrough.
        # The DiffusionModel runs the encoder output through this hook once
        # before forwarding to ``predict_flow``; for action tokens patchify is
        # a no-op since they are not spatial.
        if x.ndim == 3:
            return x
        assert x.ndim == 5, (
            f"x must be 5D [B, T, C, H, W] (latent) or 3D [B, A, action_dim] "
            f"(action chunk); got shape {tuple(x.shape)}."
        )
        return self.network.patchify_and_maybe_split_cp(x, process_group=self._cp_group)

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, f"x must be a 3D tensor [B, L, D], got shape {x.shape}"
        return self.network.unpatchify_and_maybe_gather_cp(
            pH=self.config._pH,
            pW=self.config._pW,
            x=x,
            process_group=self._cp_group,
        )

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        negative_text_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> CosmosHTransformerCache:
        """Build a fully seeded cache for a new rollout.

        Args:
            text_embeddings: ``[B, L_ctx, D_ctx_in]`` text embeddings.
            image_embeddings: ``[B, 1, C, H, W]`` first-frame VAE latent. The
                channel / height / width must match
                ``(network.in_channels, height, width)``. T is padded with
                zeros to ``len_t`` (no-op for the canonical ``len_t == 1``
                streaming geometry).
            negative_text_embeddings: Required when ``guidance_scale > 1.0``.
        """
        cfg = self.config
        head_dim = cfg.network.model_channels // cfg.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=cfg._pT,
            len_h=cfg._pH,
            len_w=cfg._pW,
            head_dim=head_dim,
            h_extrapolation_ratio=cfg.h_extrapolation_ratio,
            w_extrapolation_ratio=cfg.w_extrapolation_ratio,
            t_extrapolation_ratio=cfg.t_extrapolation_ratio,
            device=self.device,
        )
        rope_adapter.set_context_parallel_group(cp_group=self._cp_group)

        num_tokens_per_step = cfg._pH * cfg._pW
        if self._cp_group is not None:
            num_tokens_per_step //= self._cp_group.size()
        chunk_size = num_tokens_per_step * cfg._pT
        # Window/sink in tokens: BlockKVCache wants tokens, not latent frames.
        window_size = num_tokens_per_step * cfg.window_size_t
        sink_size = num_tokens_per_step * cfg.sink_size_t

        network_cache = self.network.initialize_cache(
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            context=text_embeddings,
        )

        network_cache_uncond: CosmosHActionDiTNetworkCache | None = None
        if cfg.requires_negative_text_embeddings:
            assert negative_text_embeddings is not None, (
                f"guidance_scale={cfg.guidance_scale} > 1.0 requires "
                "negative_text_embeddings."
            )
            network_cache_uncond = self.network.initialize_cache(
                chunk_size=chunk_size,
                window_size=window_size,
                sink_size=sink_size,
                context=negative_text_embeddings,
            )

        # First-frame VAE latent. Expected shape [B, 1, C, H, W]; pad the T
        # dim with zeros so the rest of the chunk stays blank. For len_t == 1
        # (the canonical streaming geometry) the pad is a no-op.
        assert image_embeddings.ndim == 5, (
            "image_embeddings must be [B, 1, C, H, W]; got shape "
            f"{tuple(image_embeddings.shape)}"
        )
        B, T_img, C_img, H_img, W_img = image_embeddings.shape
        assert T_img == 1, (
            "image_embeddings must carry a single conditional frame (T=1), "
            f"got T={T_img}"
        )
        assert (C_img, H_img, W_img) == (
            cfg.network.in_channels,
            cfg.height,
            cfg.width,
        ), (
            "image_embeddings (C, H, W) must match (network.in_channels, "
            f"height, width) = ({cfg.network.in_channels}, {cfg.height}, "
            f"{cfg.width}); got ({C_img}, {H_img}, {W_img})"
        )
        assert B == int(cfg.batch_shape[0]), (
            f"image_embeddings batch ({B}) must match config.batch_shape[0] "
            f"({cfg.batch_shape[0]})"
        )
        # F.pad spec is reverse-order pairs; the last pair pads dim -4 (T)
        # on the right. With len_t == 1 every pair is (0, 0).
        image = F.pad(
            image_embeddings.to(device=self.device, dtype=cfg.dtype),
            (0, 0, 0, 0, 0, 0, 0, cfg.len_t - 1),
        )

        # Build the per-AR-step condition masks pre-patchify: ones only on the
        # first temporal latent frame at AR 0, zeros elsewhere / always at AR > 0.
        mask_first_block = torch.zeros(
            B,
            cfg.len_t,
            1,
            cfg.height,
            cfg.width,
            device=self.device,
            dtype=cfg.dtype,
        )
        mask_first_block[:, :1, :, :, :] = 1.0
        mask_other_blocks = torch.zeros_like(mask_first_block)

        # Patchify once at rollout start.
        image_patched = self.patchify_and_maybe_split_cp(image)
        mask_first_patched = self.patchify_and_maybe_split_cp(mask_first_block)
        mask_other_patched = self.patchify_and_maybe_split_cp(mask_other_blocks)

        # Reset any prior CUDA graph: it refers to slot pointers from the
        # previous cache, which the new cache invalidates.
        if self._use_cuda_graph:
            assert isinstance(self._network_call, CUDAGraphWrapper)
            self._network_call.reset()
            assert isinstance(self._network_call_uncond, CUDAGraphWrapper)
            self._network_call_uncond.reset()

        return CosmosHTransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
            rope_adapter=rope_adapter,
            image_patched=image_patched,
            mask_first_block_patched=mask_first_patched,
            mask_other_blocks_patched=mask_other_patched,
        )

    def _maybe_inject_image(
        self,
        latent: Tensor,
        cache: CosmosHTransformerCache,
    ) -> Tensor:
        """Stamp the first-frame VAE latent into ``latent`` at AR step 0.

        At AR > 0 this is a no-op. At AR == 0 the patched first-block mask
        is 1 wherever the conditional frame lives; the inject blends the
        prediction toward the cached image at exactly those positions. This
        is *separate* from the network's ``condition_video_input_mask``
        argument: upstream's
        ``ActionVideo2WorldModelTrigflowSelfForcingDMD2`` was trained with
        ``num_conditional_frames=0``, so the network always sees an
        all-zero mask channel — see :meth:`_predict_branch`.
        """
        if cache.autoregressive_index != 0:
            return latent
        # Take one feature channel of the patched mask: every patched position
        # for a given (t, h, w) shares the same conditional value across the
        # unrolled patch features.
        mask = cache.mask_first_block_patched[..., :1]
        return latent * (1.0 - mask) + cache.image_patched * mask

    def _select_network(self, cache: CosmosHTransformerCache, *, uncond: bool) -> Any:
        if not self._use_cuda_graph:
            return self.network

        network_call = self._network_call_uncond if uncond else self._network_call
        assert isinstance(network_call, CUDAGraphWrapper)
        # Filling phase -> .drain (eager); steady-state -> __call__ (capture/replay).
        return (
            network_call.drain
            if cache.autoregressive_index < self.config._steady_ar_idx
            else network_call
        )

    def _predict_branch(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosHTransformerCache,
        network_cache: CosmosHActionDiTNetworkCache,
        action: Tensor | None,
        *,
        uncond: bool,
    ) -> Tensor:
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "Cache.start(autoregressive_index) must be called before "
            "predict_flow (DiffusionModel.generate handles this)."
        )
        rope_freqs = cache.rope_adapter.shift_t(offset=ar_idx * self.config._pT)
        # Stamp the conditional frame into the noisy latent before the
        # network sees it; pairs with the masked-out condition channel so the
        # network knows which positions are "given" rather than "predict".
        noisy_latent = self._maybe_inject_image(noisy_latent, cache)
        return self._select_network(cache, uncond=uncond)(
            noisy_latent,
            timesteps=timestep,
            rope_freqs=rope_freqs,
            cache=network_cache,
            condition_video_input_mask=cache.mask_other_blocks_patched,
            action=action,
            current_chunk_idx=ar_idx,
            eager_mode=False,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosHTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Predict the flow for the current denoising step.

        Args:
            noisy_latent: Patchified noisy latent ``[B, L/cp, D]``.
            timestep: Scalar timestep tensor.
            cache: Per-rollout AR cache.
            input: Action chunk for this AR step, ``[B, A, action_dim]``, or
                ``None`` to skip action injection (e.g. unconditional first-frame
                prefill).
        """
        action = input
        flow_cond = self._predict_branch(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache,
            action=action,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        flow_uncond = self._predict_branch(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache_uncond,
            action=action,
            uncond=True,
        )
        return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: CosmosHTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Re-stamp the conditional first frame onto the predicted clean latent.

        Without this the AR 0 prediction at the conditional positions could
        drift by a small amount even though the network was told (via the
        mask channel) those positions were given. The inject also matters
        for ``DiffusionModel.finalize`` which feeds this clean latent back
        through ``finalize_kv_cache`` to seed the rolling KV cache.
        """
        del input
        return self._maybe_inject_image(clean_latent, cache)

    def finalize_kv_cache(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.config.skip_finalize_kv_cache:
            return
        super().finalize_kv_cache(*args, **kwargs)
