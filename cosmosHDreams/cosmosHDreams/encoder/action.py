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

"""Per-AR-step action encoder for Cosmos-H-Dreams."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)


@dataclass(kw_only=True)
class ActionEncoderCache(StreamingEncoderCache):
    """Per-rollout action-encoder cache.

    Stateless across AR steps now that actions flow per ``generate()`` call;
    it only carries the per-step sizing knobs so :meth:`ActionEncoder.forward`
    can validate the chunk length.
    """

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


def num_generated_frames(autoregressive_index: int, latent_frames_per_step: int) -> int:
    """Generated latent frames at ``autoregressive_index``.

    AR step 0 leads with the conditional (image-anchored) frame, so it drives
    ``latent_frames_per_step - 1`` frames; every later step drives all
    ``latent_frames_per_step``. Shared by the encoder (chunk validation) and
    :meth:`CosmoshPipeline.get_num_actions` / ``get_num_frames`` so the AR-0
    offset is defined in exactly one place.
    """
    if autoregressive_index == 0:
        return latent_frames_per_step - 1
    return latent_frames_per_step


class ActionEncoder(StreamingEncoder[ActionEncoderCache]):
    """Forwards (and validates) the per-AR-step action chunk to the network.

    Pipeline call shape:

        cache = encoder.initialize_autoregressive_cache()
        chunk = encoder(input=None, autoregressive_index=0, cache=cache)
        # chunk: None  (AR 0 prefill when latent_frames_per_step == 1)
        chunk = encoder(input=actions_step1, autoregressive_index=1, cache=cache)
        # chunk: [B, latent_frames_per_step * A, action_dim]
    """

    def __init__(self, config: ActionEncoderConfig) -> None:
        super().__init__(config)
        self.config: ActionEncoderConfig = config

    def initialize_autoregressive_cache(self) -> ActionEncoderCache:
        """Build the (stateless) per-rollout cache."""
        return ActionEncoderCache(
            num_action_per_latent_frame=self.config.num_action_per_latent_frame,
            latent_frames_per_step=self.config.latent_frames_per_step,
        )

    def forward(
        self,
        input: Tensor | None,
        autoregressive_index: int = 0,
        cache: ActionEncoderCache | None = None,
    ) -> Tensor | None:
        """Validate and return the action chunk for AR step ``autoregressive_index``.

        Args:
            input: The action chunk for this step, ``[B, n_gen * A, action_dim]``
                where ``n_gen`` is the number of generated latent frames at this
                step (``L - 1`` at AR 0, else ``L``). Pass ``None`` when the step
                drives no generated frame (``L == 1`` at AR step 0).
            autoregressive_index: 0-based AR step index.
            cache: Per-rollout cache from :meth:`initialize_autoregressive_cache`.

        Returns:
            ``None`` when the step drives no generated frame; otherwise the
            ``[B, n_gen * A, action_dim]`` chunk unchanged.
        """
        assert cache is not None, "ActionEncoder requires a cache"
        A = cache.num_action_per_latent_frame
        n_gen = num_generated_frames(autoregressive_index, cache.latent_frames_per_step)
        if n_gen == 0:
            assert input is None, (
                f"AR step {autoregressive_index} drives no generated frame "
                f"(latent_frames_per_step={cache.latent_frames_per_step}); pass "
                f"actions=None, got a chunk of shape {tuple(input.shape)}."
            )
            return None
        assert input is not None, (
            f"AR step {autoregressive_index} drives {n_gen} frame(s); "
            f"expected an action chunk of {n_gen * A} rows, got None."
        )
        assert input.ndim == 3 and input.shape[1] == n_gen * A, (
            f"AR step {autoregressive_index} expects actions of shape "
            f"[B, {n_gen * A}, action_dim] (n_gen={n_gen}, A={A}); got "
            f"{tuple(input.shape)}."
        )
        return input
