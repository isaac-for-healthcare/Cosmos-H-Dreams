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

"""CosmosH recipe constants (checkpoint paths)."""

AVAILABLE_COSMOSH_CHECKPOINT_PATHS: dict[str, str] = {
    # CosmosH 2B EMA, action_dim=44, temporal_compression_ratio=4.
    # Single-file .pt extracted from the upstream DCP checkpoint.
    "default": "/localhome/local-javierg/checkpoints/model_ema_bf16.pt",
}
"""Canonical CosmosH checkpoint paths. Keys are the variant shorthand;
values are local paths or ``s3://`` URIs."""
