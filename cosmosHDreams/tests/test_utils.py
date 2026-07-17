# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared manifest/input helpers."""

from __future__ import annotations

import pytest

from cosmosHDreams.utils import (
    IMAGE_SUFFIXES,
    is_image_path,
    resolve_manifest_input_path,
)


def test_is_image_path_recognises_common_suffixes() -> None:
    for suffix in IMAGE_SUFFIXES:
        assert is_image_path(f"scene{suffix}")
        assert is_image_path(f"scene{suffix.upper()}")
    assert not is_image_path("scene.mp4")
    assert not is_image_path("scene.mov")


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"input": "frame.png"}, "frame.png"),
        ({"input_video": "clip.mp4"}, "clip.mp4"),
        ({"input_image": "still.jpg"}, "still.jpg"),
    ],
)
def test_resolve_manifest_input_path_accepts_single_key(
    entry: dict[str, str], expected: str
) -> None:
    assert resolve_manifest_input_path(entry) == expected


def test_resolve_manifest_input_path_rejects_missing_key() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        resolve_manifest_input_path({"input_action": "a.npy"})


def test_resolve_manifest_input_path_rejects_multiple_keys() -> None:
    with pytest.raises(ValueError, match="only one of"):
        resolve_manifest_input_path(
            {"input": "a.png", "input_video": "b.mp4", "input_action": "a.npy"}
        )
