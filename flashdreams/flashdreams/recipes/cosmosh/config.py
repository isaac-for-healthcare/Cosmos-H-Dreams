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

"""Pipeline-config builders for CosmosH."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from flashdreams.configs.registry import register_runner
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.cosmosh.constants import (
    AVAILABLE_COSMOSH_CHECKPOINT_PATHS,
)
from flashdreams.recipes.cosmosh.encoder.action import (
    ActionEncoder,
    ActionEncoderConfig,
)
from flashdreams.recipes.cosmosh.pipeline import (
    CosmoshPipeline,
    CosmoshPipelineConfig,
)
from flashdreams.recipes.cosmosh.runner import CosmoshRunnerConfig
from flashdreams.recipes.cosmosh.transformer import (
    CosmosHTransformer,
    CosmosHTransformerConfig,
)
from flashdreams.recipes.cosmosh.transformer.impl.network import (
    CosmosHActionDiTNetwork,
    CosmosHActionDiTNetworkConfig,
)
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

# Canonical CosmosH 4-step student schedule. With ``shift=5`` the warped
# sigmas at these timesteps are ``[1.0, 0.9375, 0.833, 0.625]`` — the
# distribution the network was trained against.
_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS: tuple[int, ...] = (1000, 750, 500, 250)
"""FlowMatch ``denoising_timesteps`` for the canonical 4-step student loop."""

_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS_2STEP: tuple[int, ...] = (1000, 450)
"""FlowMatch ``denoising_timesteps`` for a 2-step student loop. Mirrors
the alpadreams ``_DEFAULT_DENOISING_TIMESTEPS`` since CosmosH and
alpadreams share a training procedure."""

_COSMOSH_FLOWMATCH_SHIFT = 5.0
"""FlowMatch warp factor matching the CosmosH training schedule."""


# ---------------------------------------------------------------------------
# VAE encoder / decoder helpers.
#
# The CosmosH pipeline keeps ``decoder=None`` (per-block latents are
# accumulated and decoded externally), so these helpers exist to give the
# example script + downstream callers a single place to pick a VAE/TAE
# variant. The naming follows the alpadreams convention:
#
#   - ``_wan_vae_encoder_config`` / ``_wan_vae_decoder_config``: full
#     Wan2.1 VAE. Highest fidelity, slowest. Default checkpoint key
#     ``"vae"``; the smaller distilled ``"lightvae"`` is also exposed as
#     an opt-in for the encoder.
#   - ``_teahv_vae_decoder_config``: TAEHV "lighttae" decoder. Small
#     distilled VAE designed to be a drop-in for the Wan decoder at
#     inference time; trades a bit of fidelity for ~10x throughput.
# ---------------------------------------------------------------------------


def _wan_vae_encoder_config(
    *,
    checkpoint_name: str = "vae",
    use_compile: bool = False,
    use_cuda_graph: bool = True,
) -> WanVAEEncoderConfig:
    """Wan2.1 VAE encoder. ``checkpoint_name`` picks ``"vae"`` (full) or
    ``"lightvae"`` (distilled)."""
    return WanVAEEncoderConfig(
        _target=WanVAEEncoder,
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS[checkpoint_name],
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )


def _wan_vae_decoder_config(
    *,
    use_compile: bool = False,
    use_cuda_graph: bool = True,
) -> WanVAEDecoderConfig:
    """Full Wan2.1 VAE decoder."""
    return WanVAEDecoderConfig(
        _target=WanVAEDecoder,
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )


def _teahv_vae_decoder_config(
    *,
    use_compile: bool = False,
    use_cuda_graph: bool = True,
) -> TeahvVAEDecoderConfig:
    """TAEHV ``"lighttae"`` decoder. Drop-in replacement for the Wan2.1
    decoder; matches the same temporal/spatial compression ratios and
    runs ~10x faster at the cost of some fidelity."""
    return TeahvVAEDecoderConfig(
        _target=TeahvVAEDecoder,
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )


# ---------------------------------------------------------------------------
# CosmosH bundle: pipeline + external VAE encoder / decoder.
#
# The CosmosH pipeline keeps ``decoder=None`` because we accumulate per-block
# latents and decode the entire block at once externally. To still let the
# user pick everything (DiT pipeline + which VAE pair) by a single name, the
# named builders below return a :class:`CosmoshRunBundle`. The example script
# instantiates each component separately from the bundle.
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CosmoshRunBundle:
    """End-to-end CosmosH inference bundle.

    Pairs the streaming :class:`CosmoshPipelineConfig` with the external
    Wan2.1 VAE encoder and decoder used to encode the conditional first
    frame and decode the per-block output latents.
    """

    pipeline: CosmoshPipelineConfig
    """Streaming DiT + scheduler pipeline."""

    vae_encoder: WanVAEEncoderConfig
    """Wan2.1 VAE encoder for the conditional first frame."""

    vae_decoder: WanVAEDecoderConfig | TeahvVAEDecoderConfig
    """Output decoder. Either the full Wan2.1 VAE or the TAEHV
    ``lighttae`` distilled drop-in."""


# ---------------------------------------------------------------------------
# Smoke (random init, FlowMatch scheduler, tiny DiT) — used by the unit tests.
# ---------------------------------------------------------------------------


def build_cosmosh_smoke(
    *,
    seed: int = 42,
    window_size_t: int = 4,
    recipe_name: str = "cosmosh-smoke",
) -> CosmoshPipelineConfig:
    """Tiny random-init pipeline for the smoke / load-bearing tests.

    Latent grid 1x16x16 with a tiny 2-block / 128-channel DiT so the test
    fits comfortably on a single GPU. Same contracts as the production
    builder (single view, ``len_t = 1``, action conditioning) but with a
    one-step :class:`FlowMatchScheduler` instead of the canonical 4-hop
    schedule.
    """
    network = CosmosHActionDiTNetworkConfig(
        _target=CosmosHActionDiTNetwork,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=128,
        num_blocks=2,
        num_heads=2,
        action_dim=44,
        num_action_per_latent_frame=4,
        # crossattn_proj_in_channels matches the upstream CR1 width when CFG
        # plumbing is exercised; for the smoke test we just feed the matching
        # context width so the projection runs cleanly.
        use_crossattn_projection=True,
        crossattn_proj_in_channels=64,
        crossattn_emb_channels=64,
    )
    transformer = CosmosHTransformerConfig(
        _target=CosmosHTransformer,
        network=network,
        height=16,
        width=16,
        len_t=1,
        cp_size=1,
        window_size_t=window_size_t,
        sink_size_t=0,
        compile_network=False,
        use_cuda_graph=False,
        guidance_scale=1.0,
    )
    return CosmoshPipelineConfig(
        recipe_name=recipe_name,
        _target=CosmoshPipeline,
        encoder=ActionEncoderConfig(
            _target=ActionEncoder, num_action_per_latent_frame=4
        ),
        decoder=None,
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=0,
            transformer=transformer,
            scheduler=FlowMatchSchedulerConfig(
                num_inference_steps=1,
                denoising_timesteps=[500],
                warp_denoising_step=True,
                shift=5.0,
                num_train_timesteps=1000,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# CosmosH 2B production builder (real checkpoint + scheduler choice).
# ---------------------------------------------------------------------------

_DEFAULT_VIDEO_HEIGHT = 480
_DEFAULT_VIDEO_WIDTH = 640
_WAN_VAE_SPATIAL_COMPRESSION = 8

_COSMOSH_HEIGHT_LAT = _DEFAULT_VIDEO_HEIGHT // _WAN_VAE_SPATIAL_COMPRESSION
_COSMOSH_WIDTH_LAT = _DEFAULT_VIDEO_WIDTH // _WAN_VAE_SPATIAL_COMPRESSION

_COSMOSH_WINDOW_SIZE_T = 13
"""KV-cache rolling window in latent frames; sized to comfortably hold a
13-frame conditioning clip's worth of latent K/V (the deterministic
script's ``cache_frame_size=-1`` means "no cap"; 13 covers the canonical
CosmosH pretraining clip)."""


def build_cosmosh(
    *,
    seed: int = 1,
    height: int = _COSMOSH_HEIGHT_LAT,
    width: int = _COSMOSH_WIDTH_LAT,
    window_size_t: int = _COSMOSH_WINDOW_SIZE_T,
    checkpoint_path: str | None = None,
    num_inference_steps: int | None = None,
    compile_network: bool = True,
    use_cuda_graph: bool = True,
    denoising_timesteps: tuple[int, ...] = _COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS,
    recipe_name: str = "cosmosh",
) -> CosmoshPipelineConfig:
    """Production CosmosH pipeline for the canonical 2B self-forcing checkpoint.

    Network sizes: 2B params, 28 blocks, 16 heads, ``in_channels=16``
    (Wan2.1 VAE latent), ``action_dim=44``,
    ``num_action_per_latent_frame=4``. The network output is treated as a
    rectified-flow velocity by :class:`FlowMatchScheduler`, which renoises
    with ``x = (1-sigma)*x0 + sigma*eps`` between hops.

    Args:
        seed: RNG seed for initial noise.
        height: Latent height. Default 88 (720p ÷ Wan2.1 spatial /8 + pad).
        width: Latent width. Default 160 (1280 ÷ 8).
        window_size_t: KV cache rolling window in latent frames.
        checkpoint_path: Override the default
            ``AVAILABLE_COSMOSH_CHECKPOINT_PATHS['default']`` location.
            Pass ``None`` to use the default.
        num_inference_steps: Number of student hops. ``None`` uses the full
            ``denoising_timesteps`` schedule. Otherwise the first ``N``
            entries are used.
        compile_network: ``torch.compile`` the DiT network. Defaults to
            ``True`` (matches the alpadreams recipe). Pass ``False`` to
            run the DiT eagerly (useful for debugging).
        use_cuda_graph: Wrap the DiT forward in a per-rollout
            :class:`CUDAGraphWrapper`. Defaults to ``True``. Filling-phase
            calls run on the eager ``.drain`` path; once the KV cache
            reaches steady state the wrapper captures and replays.
        denoising_timesteps: FlowMatch student schedule (in ``[0, 1000]``).
            Defaults to the canonical 4-step
            :data:`_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS`. Pass
            :data:`_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS_2STEP` for the
            2-step variant.

    Returns:
        A :class:`CosmoshPipelineConfig` ready for ``.setup().to('cuda').eval()``.
    """
    if checkpoint_path is None:
        checkpoint_path = AVAILABLE_COSMOSH_CHECKPOINT_PATHS["default"]

    full_schedule_len = len(denoising_timesteps)
    if num_inference_steps is not None:
        assert (
            1 <= num_inference_steps <= full_schedule_len
        ), f"num_inference_steps must be in [1, {full_schedule_len}], got {num_inference_steps}"
    n_steps = (
        num_inference_steps if num_inference_steps is not None else full_schedule_len
    )

    network = CosmosHActionDiTNetworkConfig(
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
    )

    scheduler_config = FlowMatchSchedulerConfig(
        num_inference_steps=n_steps,
        denoising_timesteps=list(denoising_timesteps[:n_steps]),
        warp_denoising_step=True,
        shift=_COSMOSH_FLOWMATCH_SHIFT,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )

    transformer = CosmosHTransformerConfig(
        _target=CosmosHTransformer,
        network=network,
        height=height,
        width=width,
        len_t=1,
        cp_size=1,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        window_size_t=window_size_t,
        sink_size_t=0,
        compile_network=compile_network,
        use_cuda_graph=use_cuda_graph,
        guidance_scale=1.0,
        checkpoint_path=checkpoint_path,
    )
    return CosmoshPipelineConfig(
        recipe_name=recipe_name,
        _target=CosmoshPipeline,
        encoder=ActionEncoderConfig(
            _target=ActionEncoder, num_action_per_latent_frame=4
        ),
        decoder=None,
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=0,
            transformer=transformer,
            scheduler=scheduler_config,
        ),
    )


# ---------------------------------------------------------------------------
# Fast-path helper.
# ---------------------------------------------------------------------------


def with_compile_and_cuda_graph(
    base: CosmoshPipelineConfig,
) -> CosmoshPipelineConfig:
    """Return ``base`` with ``compile_network`` + ``use_cuda_graph`` flipped on.

    Pairs ``torch.compile`` (in ``mode="max-autotune-no-cudagraphs"`` via
    :func:`flashdreams.infra.compile.compile_module`) with the
    :class:`CUDAGraphWrapper` static-buffer replay path. The wrapper builds
    lazily inside ``initialize_autoregressive_cache``, so each rollout gets
    a fresh capture against its own KV cache pointers.

    Use after the base builder so the caller can pick which variant to
    accelerate:

        base = build_cosmosh()
        fast = with_compile_and_cuda_graph(base)
    """
    return cast(
        CosmoshPipelineConfig,
        derive_config(
            base,
            diffusion_model=dict(
                transformer=dict(
                    compile_network=True,
                    use_cuda_graph=True,
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Named CosmosH bundle builders. Mirrors the alpadreams convention of one
# function per ``<encoder>_<decoder>`` combination so the example script
# can pick a complete configuration by a single string name.
# ---------------------------------------------------------------------------


def _build_cosmosh_bundle(
    *,
    seed: int,
    height: int,
    width: int,
    window_size_t: int,
    checkpoint_path: str | None,
    num_inference_steps: int | None,
    compile_network: bool,
    use_cuda_graph: bool,
    denoising_timesteps: tuple[int, ...],
    encoder_checkpoint_name: str,
    decoder_kind: str,
    compile_vae: bool,
    recipe_name: str,
) -> CosmoshRunBundle:
    """Shared body for the named ``build_cosmosh_*`` bundle builders."""
    pipeline = build_cosmosh(
        seed=seed,
        height=height,
        width=width,
        window_size_t=window_size_t,
        checkpoint_path=checkpoint_path,
        num_inference_steps=num_inference_steps,
        compile_network=compile_network,
        use_cuda_graph=use_cuda_graph,
        denoising_timesteps=denoising_timesteps,
        recipe_name=recipe_name,
    )
    vae_encoder = _wan_vae_encoder_config(
        checkpoint_name=encoder_checkpoint_name, use_compile=compile_vae
    )
    vae_decoder: WanVAEDecoderConfig | TeahvVAEDecoderConfig
    if decoder_kind == "vae":
        vae_decoder = _wan_vae_decoder_config(use_compile=compile_vae)
    elif decoder_kind == "lighttae":
        vae_decoder = _teahv_vae_decoder_config(use_compile=compile_vae)
    else:
        raise ValueError(f"unknown decoder_kind {decoder_kind!r}")
    return CosmoshRunBundle(
        pipeline=pipeline, vae_encoder=vae_encoder, vae_decoder=vae_decoder
    )


def _make_named_builder(
    *,
    encoder_checkpoint_name: str,
    decoder_kind: str,
    denoising_timesteps: tuple[int, ...],
    default_recipe_name: str,
) -> Callable[..., CosmoshRunBundle]:
    """Factory: returns a ``build_cosmosh_*`` function pinned to one
    encoder/decoder/schedule combination. Keeps the named builders below to
    one line each."""

    def _build(
        *,
        seed: int = 1,
        height: int = _COSMOSH_HEIGHT_LAT,
        width: int = _COSMOSH_WIDTH_LAT,
        window_size_t: int = _COSMOSH_WINDOW_SIZE_T,
        checkpoint_path: str | None = None,
        num_inference_steps: int | None = None,
        compile_network: bool = True,
        use_cuda_graph: bool = True,
        compile_vae: bool = False,
        recipe_name: str = default_recipe_name,
    ) -> CosmoshRunBundle:
        return _build_cosmosh_bundle(
            seed=seed,
            height=height,
            width=width,
            window_size_t=window_size_t,
            checkpoint_path=checkpoint_path,
            num_inference_steps=num_inference_steps,
            compile_network=compile_network,
            use_cuda_graph=use_cuda_graph,
            denoising_timesteps=denoising_timesteps,
            encoder_checkpoint_name=encoder_checkpoint_name,
            decoder_kind=decoder_kind,
            compile_vae=compile_vae,
            recipe_name=recipe_name,
        )

    return _build


# 4-step variants (canonical CosmosH student schedule).
build_cosmosh_vae_vae = _make_named_builder(
    encoder_checkpoint_name="vae",
    decoder_kind="vae",
    denoising_timesteps=_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS,
    default_recipe_name="cosmosh-vae-vae",
)
"""4-step: full Wan2.1 VAE encoder + full Wan2.1 VAE decoder (highest fidelity)."""

build_cosmosh_vae_lighttae = _make_named_builder(
    encoder_checkpoint_name="vae",
    decoder_kind="lighttae",
    denoising_timesteps=_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS,
    default_recipe_name="cosmosh-vae-lighttae",
)
"""4-step: full Wan2.1 VAE encoder + TAEHV ``lighttae`` decoder."""

build_cosmosh_lightvae_lighttae = _make_named_builder(
    encoder_checkpoint_name="lightvae",
    decoder_kind="lighttae",
    denoising_timesteps=_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS,
    default_recipe_name="cosmosh-lightvae-lighttae",
)
"""4-step: distilled ``lightvae`` encoder + TAEHV ``lighttae`` decoder
(fastest 4-step, lowest fidelity)."""

# 2-step variants (alpadreams-style ``[1000, 450]`` student schedule).
build_cosmosh_2steps_vae_vae = _make_named_builder(
    encoder_checkpoint_name="vae",
    decoder_kind="vae",
    denoising_timesteps=_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS_2STEP,
    default_recipe_name="cosmosh-2steps-vae-vae",
)
"""2-step: full Wan2.1 VAE encoder + full Wan2.1 VAE decoder."""

build_cosmosh_2steps_vae_lighttae = _make_named_builder(
    encoder_checkpoint_name="vae",
    decoder_kind="lighttae",
    denoising_timesteps=_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS_2STEP,
    default_recipe_name="cosmosh-2steps-vae-lighttae",
)
"""2-step: full Wan2.1 VAE encoder + TAEHV ``lighttae`` decoder."""

build_cosmosh_2steps_lightvae_lighttae = _make_named_builder(
    encoder_checkpoint_name="lightvae",
    decoder_kind="lighttae",
    denoising_timesteps=_COSMOSH_FLOWMATCH_DENOISING_TIMESTEPS_2STEP,
    default_recipe_name="cosmosh-2steps-lightvae-lighttae",
)
"""2-step: distilled ``lightvae`` encoder + TAEHV ``lighttae`` decoder
(fastest overall)."""


COSMOSH_CONFIG_BUILDERS: dict[str, Callable[..., CosmoshRunBundle]] = {
    "vae_vae": build_cosmosh_vae_vae,
    "vae_lighttae": build_cosmosh_vae_lighttae,
    "lightvae_lighttae": build_cosmosh_lightvae_lighttae,
    "2steps_vae_vae": build_cosmosh_2steps_vae_vae,
    "2steps_vae_lighttae": build_cosmosh_2steps_vae_lighttae,
    "2steps_lightvae_lighttae": build_cosmosh_2steps_lightvae_lighttae,
}


## Per-variant runner-config literals (slug == ``cosmosh-<bundle-name>``).

_COSMOSH_DESCRIPTIONS: dict[str, str] = {
    "cosmosh-vae-vae": (
        "CosmosH 4-step action-conditioned I2V (Wan2.1 VAE encoder + decoder)."
    ),
    "cosmosh-vae-lighttae": (
        "CosmosH 4-step action-conditioned I2V (Wan2.1 VAE encoder, TAEHV "
        "``lighttae`` decoder)."
    ),
    "cosmosh-lightvae-lighttae": (
        "CosmosH 4-step action-conditioned I2V (distilled ``lightvae`` encoder + "
        "TAEHV ``lighttae`` decoder)."
    ),
    "cosmosh-2steps-vae-vae": (
        "CosmosH 2-step action-conditioned I2V (Wan2.1 VAE encoder + decoder)."
    ),
    "cosmosh-2steps-vae-lighttae": (
        "CosmosH 2-step action-conditioned I2V (Wan2.1 VAE encoder, TAEHV "
        "``lighttae`` decoder)."
    ),
    "cosmosh-2steps-lightvae-lighttae": (
        "CosmosH 2-step action-conditioned I2V (distilled ``lightvae`` encoder + "
        "TAEHV ``lighttae`` decoder; fastest overall)."
    ),
}


def _build_cosmosh_runners() -> dict[str, RunnerConfig]:
    """Project ``COSMOSH_CONFIG_BUILDERS`` into per-variant runner literals."""
    runners: dict[str, RunnerConfig] = {}
    for slug, builder in COSMOSH_CONFIG_BUILDERS.items():
        bundle = builder()
        runner_name = f"cosmosh-{slug.replace('_', '-')}"
        assert runner_name in _COSMOSH_DESCRIPTIONS, (
            f"missing CLI description for cosmosh slug {runner_name!r}; "
            "add an entry to ``_COSMOSH_DESCRIPTIONS``."
        )
        runners[runner_name] = CosmoshRunnerConfig(
            runner_name=runner_name,
            description=_COSMOSH_DESCRIPTIONS[runner_name],
            pipeline=bundle.pipeline,
            vae_encoder=bundle.vae_encoder,
            vae_decoder=bundle.vae_decoder,
        )
    return runners


COSMOSH_RUNNERS: dict[str, RunnerConfig] = _build_cosmosh_runners()
"""All shipped CosmosH runners, keyed by ``runner_name``."""

for _name, _cfg in COSMOSH_RUNNERS.items():
    register_runner(_name, _cfg, source="builtin")
