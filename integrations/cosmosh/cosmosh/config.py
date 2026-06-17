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

"""User-facing configs for CosmosH.

Mirrors :mod:`omnidreams.config`: module-level **literal**
:class:`CosmoshPipelineConfig` objects + ``derive_config`` variants (no
``build_*`` factories), aggregated into ``COSMOSH_CONFIGS``; per-slug
:class:`CosmoshRunnerConfig` literals aggregated into ``COSMOSH_RUNNERS``;
and the ``RUNNER_*`` module constants wired into the
``flashdreams.runner_configs`` entry-point group by this package's
``pyproject.toml``.

The shipped matrix is ``{vae, lightvae}`` encoder x ``{vae, lighttae}``
decoder x ``{4-step, 2-step}`` schedule, each at ``len_t in {1, 2, 3}``
(the ``cosmosh-chunk{1,2,3}-*`` slugs). The encoder/decoder/schedule axes
are spelled out as explicit ``derive_config`` literals; the ``len_t`` axis
is applied mechanically by :func:`_chunk_variant` (it must re-run the
transformer config's ``__post_init__`` because ``derive_config`` setattrs
without re-deriving ``_pT``).
"""

from __future__ import annotations

from typing import cast

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.taehv import (
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
    TeahvVAEDecoder,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoder,
    WanVAEDecoderConfig,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)

from cosmosh.constants import AVAILABLE_COSMOSH_CHECKPOINT_PATHS
from cosmosh.encoder.action import ActionEncoder, ActionEncoderConfig
from cosmosh.pipeline import CosmoshPipeline, CosmoshPipelineConfig
from cosmosh.runner import CosmoshRunnerConfig
from cosmosh.transformer import CosmosHTransformer, CosmosHTransformerConfig
from cosmosh.transformer.impl.network import (
    CosmosHActionDiTNetwork,
    CosmosHActionDiTNetworkConfig,
)

# ---------------------------------------------------------------------------
# Spatial defaults (public — examples / integrations import these to compute
# pixel <-> latent dims).
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_HEIGHT = 288
"""Pixel-space rollout height."""

DEFAULT_VIDEO_WIDTH = 512
"""Pixel-space rollout width."""

COSMOSH_VAE_SPATIAL_COMPRESSION = 8
"""Wan2.1 VAE spatial compression ratio."""

_HEIGHT_LAT = DEFAULT_VIDEO_HEIGHT // COSMOSH_VAE_SPATIAL_COMPRESSION
_WIDTH_LAT = DEFAULT_VIDEO_WIDTH // COSMOSH_VAE_SPATIAL_COMPRESSION

_WINDOW_SIZE_T = 11
"""KV-cache rolling window in latent frames; a whole multiple of every
shipped ``len_t`` (1, 2, 3) so no per-variant window tuning is needed."""

_SINK_SIZE_T = 1
"""Sink-token count in latent frames; a whole multiple of every shipped ``len_t`` (1, 2, 3) so no per-variant sink tuning is needed."""

_FLOWMATCH_SHIFT = 5.0
"""FlowMatch warp factor matching the CosmosH training schedule."""

# Canonical CosmosH 4-step student schedule (with ``shift=5`` the warped sigmas
# are ``[1.0, 0.9375, 0.833, 0.625]`` — the distribution the net was trained on).
_DENOISING_4STEP = [1000, 750, 500, 250]
# 2-step student schedule (alpadreams-style ``[1000, 450]``).
_DENOISING_2STEP = [1000, 450]


# ---------------------------------------------------------------------------
# Base literal: 4-step, full Wan VAE encoder + decoder, len_t=1.
# Every other variant derives from this one.
# ---------------------------------------------------------------------------

COSMOSH_VAE_VAE = CosmoshPipelineConfig(
    name="cosmosh-vae-vae",
    _target=CosmoshPipeline,
    image_encoder=WanVAEEncoderConfig(
        _target=WanVAEEncoder,
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        use_compile=False,
        use_cuda_graph=True,
    ),
    encoder=ActionEncoderConfig(
        _target=ActionEncoder,
        num_action_per_latent_frame=4,
        latent_frames_per_step=1,
    ),
    decoder=WanVAEDecoderConfig(
        _target=WanVAEDecoder,
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        use_compile=False,
        use_cuda_graph=True,
    ),
    diffusion_model=DiffusionModelConfig(
        seed=1,
        context_noise=0,
        transformer=CosmosHTransformerConfig(
            _target=CosmosHTransformer,
            network=CosmosHActionDiTNetworkConfig(
                _target=CosmosHActionDiTNetwork,
                in_channels=16,
                out_channels=16,
                patch_spatial=2,
                patch_temporal=1,
                model_channels=2048,
                num_blocks=28,
                num_heads=16,
                mlp_ratio=4.0,
                concat_padding_mask=True,
                use_adaln_lora=True,
                adaln_lora_dim=256,
                use_crossattn_projection=True,
                crossattn_proj_in_channels=100352,  # CR1-7B context width
                crossattn_emb_channels=1024,
                timestep_scale=0.001,
                action_dim=44,
                num_action_per_latent_frame=4,
                hidden_dim_in_action_embedder=None,  # default: 4 * model_channels
            ),
            height=_HEIGHT_LAT,
            width=_WIDTH_LAT,
            len_t=1,
            cp_size=1,
            h_extrapolation_ratio=3.0,
            w_extrapolation_ratio=3.0,
            t_extrapolation_ratio=1.0,
            window_size_t=_WINDOW_SIZE_T,
            sink_size_t=_SINK_SIZE_T,
            compile_network=True,
            use_cuda_graph=True,
            guidance_scale=1.0,
            checkpoint_path=AVAILABLE_COSMOSH_CHECKPOINT_PATHS["default"],
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=len(_DENOISING_4STEP),
            denoising_timesteps=list(_DENOISING_4STEP),
            warp_denoising_step=True,
            shift=_FLOWMATCH_SHIFT,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
"""Base chassis: 4-step, full Wan2.1 VAE encoder + decoder, ``len_t=1``."""


def _teahv_decoder() -> TeahvVAEDecoderConfig:
    """LightTAE drop-in decoder (a fixed sub-config, no knobs)."""
    return TeahvVAEDecoderConfig(
        _target=TeahvVAEDecoder,
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        use_compile=False,
        use_cuda_graph=True,
    )


COSMOSH_VAE_LIGHTTAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        COSMOSH_VAE_VAE,
        name="cosmosh-vae-lighttae",
        decoder=_teahv_decoder(),
    ),
)
"""4-step: full Wan2.1 VAE encoder + TAEHV ``lighttae`` decoder."""

COSMOSH_LIGHTVAE_LIGHTTAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        COSMOSH_VAE_LIGHTTAE,
        name="cosmosh-lightvae-lighttae",
        image_encoder=dict(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"]
        ),
    ),
)
"""4-step: distilled ``lightvae`` encoder + TAEHV ``lighttae`` decoder."""

COSMOSH_2STEPS_VAE_VAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        COSMOSH_VAE_VAE,
        name="cosmosh-2steps-vae-vae",
        diffusion_model=dict(
            scheduler=dict(
                num_inference_steps=len(_DENOISING_2STEP),
                denoising_timesteps=list(_DENOISING_2STEP),
            ),
        ),
    ),
)
"""2-step: full Wan2.1 VAE encoder + decoder."""

COSMOSH_2STEPS_VAE_LIGHTTAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        COSMOSH_VAE_LIGHTTAE,
        name="cosmosh-2steps-vae-lighttae",
        diffusion_model=dict(
            scheduler=dict(
                num_inference_steps=len(_DENOISING_2STEP),
                denoising_timesteps=list(_DENOISING_2STEP),
            ),
        ),
    ),
)
"""2-step: full Wan2.1 VAE encoder + TAEHV ``lighttae`` decoder."""

COSMOSH_2STEPS_LIGHTVAE_LIGHTTAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        COSMOSH_LIGHTVAE_LIGHTTAE,
        name="cosmosh-2steps-lightvae-lighttae",
        diffusion_model=dict(
            scheduler=dict(
                num_inference_steps=len(_DENOISING_2STEP),
                denoising_timesteps=list(_DENOISING_2STEP),
            ),
        ),
    ),
)
"""2-step: distilled ``lightvae`` encoder + TAEHV ``lighttae`` decoder (fastest)."""


_BASE_CONFIGS: tuple[CosmoshPipelineConfig, ...] = (
    COSMOSH_VAE_VAE,
    COSMOSH_VAE_LIGHTTAE,
    COSMOSH_LIGHTVAE_LIGHTTAE,
    COSMOSH_2STEPS_VAE_VAE,
    COSMOSH_2STEPS_VAE_LIGHTTAE,
    COSMOSH_2STEPS_LIGHTVAE_LIGHTTAE,
)

_BASE_DESCRIPTIONS: dict[str, str] = {
    "cosmosh-vae-vae": (
        "CosmosH 4-step action-conditioned I2V (Wan2.1 VAE encoder + decoder)."
    ),
    "cosmosh-vae-lighttae": (
        "CosmosH 4-step action-conditioned I2V (Wan2.1 VAE encoder, TAEHV "
        "lighttae decoder)."
    ),
    "cosmosh-lightvae-lighttae": (
        "CosmosH 4-step action-conditioned I2V (distilled lightvae encoder + "
        "TAEHV lighttae decoder)."
    ),
    "cosmosh-2steps-vae-vae": (
        "CosmosH 2-step action-conditioned I2V (Wan2.1 VAE encoder + decoder)."
    ),
    "cosmosh-2steps-vae-lighttae": (
        "CosmosH 2-step action-conditioned I2V (Wan2.1 VAE encoder, TAEHV "
        "lighttae decoder)."
    ),
    "cosmosh-2steps-lightvae-lighttae": (
        "CosmosH 2-step action-conditioned I2V (distilled lightvae encoder + "
        "TAEHV lighttae decoder; fastest overall)."
    ),
}


def _chunk_slug(base_name: str, len_t: int) -> str:
    """``cosmosh-vae-vae`` + ``len_t=2`` -> ``cosmosh-chunk2-vae-vae``."""
    rest = base_name[len("cosmosh-") :]
    return f"cosmosh-chunk{len_t}-{rest}"


def _chunk_variant(base: CosmoshPipelineConfig, len_t: int) -> CosmoshPipelineConfig:
    """Derive a ``len_t`` variant of ``base``.

    Sets ``transformer.len_t`` and ``encoder.latent_frames_per_step`` in
    lockstep, then re-runs the transformer config's ``__post_init__`` because
    ``derive_config`` assigns via ``setattr`` and would otherwise leave the
    derived ``_pT / _pH / _pW / _steady_ar_idx`` stale for the new ``len_t``.
    """
    cfg = cast(
        CosmoshPipelineConfig,
        derive_config(
            base,
            name=_chunk_slug(base.name, len_t),
            diffusion_model=dict(transformer=dict(len_t=len_t)),
            encoder=dict(latent_frames_per_step=len_t),
        ),
    )
    tcfg = cfg.diffusion_model.transformer
    assert isinstance(tcfg, CosmosHTransformerConfig)
    tcfg.__post_init__()
    return cfg


_CHUNK_LEN_TS: tuple[int, ...] = (1, 2, 3)

_CHUNK_CONFIGS: tuple[CosmoshPipelineConfig, ...] = tuple(
    _chunk_variant(base, len_t)
    for base in _BASE_CONFIGS
    for len_t in _CHUNK_LEN_TS
)


COSMOSH_CONFIGS: dict[str, CosmoshPipelineConfig] = {
    cfg.name: cfg for cfg in (*_BASE_CONFIGS, *_CHUNK_CONFIGS)
}
"""All shipped CosmosH pipeline configs, keyed by ``name``."""


## Per-variant runner-config literals (slug == ``name``).


def _runner_for(cfg: CosmoshPipelineConfig, description: str) -> CosmoshRunnerConfig:
    return CosmoshRunnerConfig(
        runner_name=cfg.name,
        description=description,
        pipeline=cfg,
    )


def _build_cosmosh_runners() -> dict[str, RunnerConfig]:
    runners: dict[str, RunnerConfig] = {}
    for cfg in _BASE_CONFIGS:
        runners[cfg.name] = _runner_for(cfg, _BASE_DESCRIPTIONS[cfg.name])
    for base in _BASE_CONFIGS:
        base_desc = _BASE_DESCRIPTIONS[base.name].rstrip(".")
        for len_t in _CHUNK_LEN_TS:
            slug = _chunk_slug(base.name, len_t)
            runners[slug] = _runner_for(
                COSMOSH_CONFIGS[slug],
                f"{base_desc}; len_t={len_t} ({len_t} latent frame(s) per "
                "feed-forward step).",
            )
    return runners


COSMOSH_RUNNERS: dict[str, RunnerConfig] = _build_cosmosh_runners()
"""All shipped CosmosH runners, keyed by ``runner_name`` (== pipeline ``name``)."""


# Module-level constants for entry-point discovery. Each is loaded by
# ``flashdreams.plugins.registry.discover_runners`` via the
# ``[project.entry-points."flashdreams.runner_configs"]`` table in
# ``integrations/cosmosh/pyproject.toml``.
RUNNER_COSMOSH_VAE_VAE = COSMOSH_RUNNERS["cosmosh-vae-vae"]
RUNNER_COSMOSH_VAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-vae-lighttae"]
RUNNER_COSMOSH_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-lightvae-lighttae"]
RUNNER_COSMOSH_2STEPS_VAE_VAE = COSMOSH_RUNNERS["cosmosh-2steps-vae-vae"]
RUNNER_COSMOSH_2STEPS_VAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-2steps-vae-lighttae"]
RUNNER_COSMOSH_2STEPS_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-2steps-lightvae-lighttae"
]

# len_t (frames-per-feed-forward) variants: chunk1/2/3 for every bundle.
RUNNER_COSMOSH_CHUNK1_VAE_VAE = COSMOSH_RUNNERS["cosmosh-chunk1-vae-vae"]
RUNNER_COSMOSH_CHUNK2_VAE_VAE = COSMOSH_RUNNERS["cosmosh-chunk2-vae-vae"]
RUNNER_COSMOSH_CHUNK3_VAE_VAE = COSMOSH_RUNNERS["cosmosh-chunk3-vae-vae"]
RUNNER_COSMOSH_CHUNK1_VAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-chunk1-vae-lighttae"]
RUNNER_COSMOSH_CHUNK2_VAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-chunk2-vae-lighttae"]
RUNNER_COSMOSH_CHUNK3_VAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-chunk3-vae-lighttae"]
RUNNER_COSMOSH_CHUNK1_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk1-lightvae-lighttae"
]
RUNNER_COSMOSH_CHUNK2_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk2-lightvae-lighttae"
]
RUNNER_COSMOSH_CHUNK3_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk3-lightvae-lighttae"
]
RUNNER_COSMOSH_CHUNK1_2STEPS_VAE_VAE = COSMOSH_RUNNERS["cosmosh-chunk1-2steps-vae-vae"]
RUNNER_COSMOSH_CHUNK2_2STEPS_VAE_VAE = COSMOSH_RUNNERS["cosmosh-chunk2-2steps-vae-vae"]
RUNNER_COSMOSH_CHUNK3_2STEPS_VAE_VAE = COSMOSH_RUNNERS["cosmosh-chunk3-2steps-vae-vae"]
RUNNER_COSMOSH_CHUNK1_2STEPS_VAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk1-2steps-vae-lighttae"
]
RUNNER_COSMOSH_CHUNK2_2STEPS_VAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk2-2steps-vae-lighttae"
]
RUNNER_COSMOSH_CHUNK3_2STEPS_VAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk3-2steps-vae-lighttae"
]
RUNNER_COSMOSH_CHUNK1_2STEPS_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk1-2steps-lightvae-lighttae"
]
RUNNER_COSMOSH_CHUNK2_2STEPS_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk2-2steps-lightvae-lighttae"
]
RUNNER_COSMOSH_CHUNK3_2STEPS_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS[
    "cosmosh-chunk3-2steps-lightvae-lighttae"
]
