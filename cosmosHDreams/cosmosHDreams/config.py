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

The shipped matrix is ``{vae}`` encoder x ``{vae, lighttae}`` decoder x
``{4-step, 2-step}`` schedule x ``len_t in {2, 3}``
(``cosmosHDreams-chunk{2,3}-*`` slugs; 8 canonical + 4 explicit ``4steps``
aliases = 12 slugs total). The base ``len_t=1`` pipeline objects exist only
as derivation anchors and are not registered as runners.

The encoder/decoder/schedule axes are spelled out as explicit ``derive_config``
literals; the ``len_t`` axis is applied mechanically by :func:`_chunk_variant`.
Every non-2steps slug also has an explicit ``4steps`` alias so users do not need
to know that 4-step is implied by the absence of ``2steps``.
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

from cosmosHDreams.constants import AVAILABLE_COSMOSH_CHECKPOINT_PATHS
from cosmosHDreams.encoder.action import ActionEncoder, ActionEncoderConfig
from cosmosHDreams.pipeline import CosmoshPipeline, CosmoshPipelineConfig
from cosmosHDreams.runner import CosmoshRunnerConfig
from cosmosHDreams.transformer import CosmosHTransformer, CosmosHTransformerConfig
from cosmosHDreams.transformer.impl.network import (
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
shipped ``len_t`` (2, 3) so no per-variant window tuning is needed."""

_SINK_SIZE_T = 1
"""Sink-token count in latent frames."""

_FLOWMATCH_SHIFT = 5.0
"""FlowMatch warp factor matching the CosmosH training schedule."""

# Canonical CosmosH 4-step student schedule (with ``shift=5`` the warped sigmas
# are ``[1.0, 0.9375, 0.833, 0.625]`` — the distribution the net was trained on).
_DENOISING_4STEP = [1000, 937, 833, 625]
# 2-step student schedule (alpadreams-style ``[1000, 450]``).
_DENOISING_2STEP = [1000, 450]


# ---------------------------------------------------------------------------
# Base pipeline configs (len_t=1). Not registered as runners; used only as
# derivation anchors for the chunk2/chunk3 variants below.
# ---------------------------------------------------------------------------

_BASE_VAE_VAE = CosmoshPipelineConfig(
    name="cosmosHDreams-vae-vae",
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


def _teahv_decoder() -> TeahvVAEDecoderConfig:
    """LightTAE drop-in decoder (a fixed sub-config, no knobs)."""
    return TeahvVAEDecoderConfig(
        _target=TeahvVAEDecoder,
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        use_compile=False,
        use_cuda_graph=True,
    )


_BASE_VAE_LIGHTTAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        _BASE_VAE_VAE, name="cosmosHDreams-vae-lighttae", decoder=_teahv_decoder()
    ),
)

_BASE_2STEPS_VAE_VAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        _BASE_VAE_VAE,
        name="cosmosHDreams-2steps-vae-vae",
        diffusion_model=dict(
            scheduler=dict(
                num_inference_steps=len(_DENOISING_2STEP),
                denoising_timesteps=list(_DENOISING_2STEP),
            ),
        ),
    ),
)

_BASE_2STEPS_VAE_LIGHTTAE = cast(
    CosmoshPipelineConfig,
    derive_config(
        _BASE_VAE_LIGHTTAE,
        name="cosmosHDreams-2steps-vae-lighttae",
        diffusion_model=dict(
            scheduler=dict(
                num_inference_steps=len(_DENOISING_2STEP),
                denoising_timesteps=list(_DENOISING_2STEP),
            ),
        ),
    ),
)

# Ordered tuple used only to generate chunk variants; not exposed as runners.
_DERIVATION_BASES: tuple[CosmoshPipelineConfig, ...] = (
    _BASE_VAE_VAE,
    _BASE_VAE_LIGHTTAE,
    _BASE_2STEPS_VAE_VAE,
    _BASE_2STEPS_VAE_LIGHTTAE,
)

_CHUNK_DESCRIPTIONS: dict[str, str] = {
    "cosmosHDreams-vae-vae": (
        "CosmosH 4-step action-conditioned I2V (Wan2.1 VAE encoder + decoder)"
    ),
    "cosmosHDreams-vae-lighttae": (
        "CosmosH 4-step action-conditioned I2V (Wan2.1 VAE encoder, TAEHV lighttae decoder)"
    ),
    "cosmosHDreams-2steps-vae-vae": (
        "CosmosH 2-step action-conditioned I2V (Wan2.1 VAE encoder + decoder)"
    ),
    "cosmosHDreams-2steps-vae-lighttae": (
        "CosmosH 2-step action-conditioned I2V (Wan2.1 VAE encoder, TAEHV lighttae decoder)"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_slug(base_name: str, len_t: int) -> str:
    """``cosmosHDreams-vae-vae`` + ``len_t=2`` -> ``cosmosHDreams-chunk2-vae-vae``."""
    rest = base_name[len("cosmosHDreams-") :]
    return f"cosmosHDreams-chunk{len_t}-{rest}"


def _4step_slug(name: str) -> str:
    """Insert explicit ``4steps`` into a chunk slug, mirroring the ``2steps`` position.

    ``cosmosHDreams-chunk3-vae-vae`` -> ``cosmosHDreams-chunk3-4steps-vae-vae``
    """
    prefix = "cosmosHDreams-"
    rest = name[len(prefix) :]
    if rest.startswith("chunk"):
        chunk_end = rest.index("-") + 1  # length of "chunk3-"
        return f"{prefix}{rest[:chunk_end]}4steps-{rest[chunk_end:]}"
    return f"{prefix}4steps-{rest}"


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


_CHUNK_LEN_TS: tuple[int, ...] = (2, 3)

_CHUNK_CONFIGS: tuple[CosmoshPipelineConfig, ...] = tuple(
    _chunk_variant(base, len_t) for base in _DERIVATION_BASES for len_t in _CHUNK_LEN_TS
)

COSMOSH_CONFIGS: dict[str, CosmoshPipelineConfig] = {
    cfg.name: cfg for cfg in _CHUNK_CONFIGS
}
"""All registered CosmosH pipeline configs, keyed by ``name``."""


# ---------------------------------------------------------------------------
# Runner registration
# ---------------------------------------------------------------------------


def _runner_for(cfg: CosmoshPipelineConfig, description: str) -> CosmoshRunnerConfig:
    return CosmoshRunnerConfig(
        runner_name=cfg.name,
        description=description,
        pipeline=cfg,
    )


def _build_cosmosHDreams_runners() -> dict[str, RunnerConfig]:
    runners: dict[str, RunnerConfig] = {}
    for base in _DERIVATION_BASES:
        base_desc = _CHUNK_DESCRIPTIONS[base.name]
        for len_t in _CHUNK_LEN_TS:
            slug = _chunk_slug(base.name, len_t)
            runners[slug] = _runner_for(
                COSMOSH_CONFIGS[slug],
                f"{base_desc}; len_t={len_t} ({len_t} latent frame(s) per "
                "feed-forward step).",
            )
    # Explicit 4-step aliases: same pipeline as the non-2steps chunk slugs, with
    # "4steps" inserted so users do not need to know 4-step is the default.
    for slug in list(runners):
        if "2steps" not in slug:
            alias = _4step_slug(slug)
            original = cast(CosmoshRunnerConfig, runners[slug])
            runners[alias] = CosmoshRunnerConfig(
                runner_name=alias,
                description=original.description.rstrip(".")
                + "; explicit 4-step alias.",
                pipeline=original.pipeline,
            )
    return runners


COSMOSHDREAMS_RUNNERS: dict[str, RunnerConfig] = _build_cosmosHDreams_runners()
"""All registered CosmosH runners, keyed by ``runner_name`` (== pipeline ``name``)."""


# ---------------------------------------------------------------------------
# Module-level constants for entry-point discovery.
# Each is loaded by ``flashdreams.plugins.registry.discover_runners`` via the
# ``[project.entry-points."flashdreams.runner_configs"]`` table in pyproject.toml.
# ---------------------------------------------------------------------------

# chunk2 variants
RUNNER_COSMOSH_CHUNK2_VAE_VAE = COSMOSHDREAMS_RUNNERS["cosmosHDreams-chunk2-vae-vae"]
RUNNER_COSMOSH_CHUNK2_VAE_LIGHTTAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk2-vae-lighttae"
]
RUNNER_COSMOSH_CHUNK2_2STEPS_VAE_VAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk2-2steps-vae-vae"
]
RUNNER_COSMOSH_CHUNK2_2STEPS_VAE_LIGHTTAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk2-2steps-vae-lighttae"
]

# chunk3 variants (recommended)
RUNNER_COSMOSH_CHUNK3_VAE_VAE = COSMOSHDREAMS_RUNNERS["cosmosHDreams-chunk3-vae-vae"]
RUNNER_COSMOSH_CHUNK3_VAE_LIGHTTAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk3-vae-lighttae"
]
RUNNER_COSMOSH_CHUNK3_2STEPS_VAE_VAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk3-2steps-vae-vae"
]
RUNNER_COSMOSH_CHUNK3_2STEPS_VAE_LIGHTTAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk3-2steps-vae-lighttae"
]

# Explicit 4-step aliases
RUNNER_COSMOSH_CHUNK2_4STEPS_VAE_VAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk2-4steps-vae-vae"
]
RUNNER_COSMOSH_CHUNK2_4STEPS_VAE_LIGHTTAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk2-4steps-vae-lighttae"
]
RUNNER_COSMOSH_CHUNK3_4STEPS_VAE_VAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk3-4steps-vae-vae"
]
RUNNER_COSMOSH_CHUNK3_4STEPS_VAE_LIGHTTAE = COSMOSHDREAMS_RUNNERS[
    "cosmosHDreams-chunk3-4steps-vae-lighttae"
]
