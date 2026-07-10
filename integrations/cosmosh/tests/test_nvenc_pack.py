"""Unit tests for ``cosmosh.webrtc.nvenc.pack``.

Pure-PyTorch — runs on CPU. Verifies:

- The pack kernel produces the documented ``[T, H, W, 4]`` uint8 layout
  with alpha == 255 everywhere.
- Per-pixel RGB values are at most 1 LSB away from the reference
  implementation used today by
  ``cosmosh.webrtc.media.tensor_chunk_to_rgb_frames``. The slight
  rounding difference is intentional and documented in
  :func:`pack.denormalize_and_pack_argb`.
- The kernel preserves the device of its input (so on GPU it stays on
  GPU; on CPU it stays on CPU).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmosh.webrtc.nvenc.pack import (
    denormalize_and_pack_argb,
    denormalize_and_pack_argb_reference,
)

pytestmark = pytest.mark.ci_cpu


def _random_chunk(T: int = 4, H: int = 32, W: int = 32) -> torch.Tensor:
    """Random bf16 chunk in [-1, 1] with the runtime-canonical layout."""
    g = torch.Generator().manual_seed(0)
    x = (torch.rand((1, 3, T, H, W), generator=g) * 2.0 - 1.0).to(torch.bfloat16)
    return x


def test_output_shape_and_dtype():
    chunk = _random_chunk(T=4, H=8, W=16)
    argb = denormalize_and_pack_argb(chunk)
    assert argb.shape == (4, 8, 16, 4)
    assert argb.dtype == torch.uint8
    assert argb.is_contiguous()


def test_alpha_channel_is_255():
    chunk = _random_chunk()
    argb = denormalize_and_pack_argb(chunk)
    # Memory byte order is B, G, R, A — alpha is at channel index 3.
    assert torch.all(argb[..., 3] == 255)


def test_extremes_map_to_full_range():
    """Clamp to [-1, 1] then denormalize must hit 0 and 255 exactly."""
    chunk = torch.full((1, 3, 1, 1, 1), -1.0, dtype=torch.float32)
    argb = denormalize_and_pack_argb(chunk)
    # BGRA byte order — B, G, R at indices 0, 1, 2; A at index 3.
    # All input channels at -1 → all three colour bytes are 0.
    assert torch.all(argb[..., :3] == 0)
    assert torch.all(argb[..., 3] == 255)

    chunk = torch.full((1, 3, 1, 1, 1), 1.0, dtype=torch.float32)
    argb = denormalize_and_pack_argb(chunk)
    # All input channels at +1 → all three colour bytes are 255.
    assert torch.all(argb[..., :3] == 255)
    assert torch.all(argb[..., 3] == 255)


def test_clamps_out_of_range_input():
    """Values outside [-1, 1] must clamp; output stays in [0, 255]."""
    chunk = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    chunk[0, 0, 0, 0, 0] = -2.0  # R clamps to -1 → 0
    chunk[0, 1, 0, 0, 0] = 2.0   # G clamps to +1 → 255
    chunk[0, 2, 0, 0, 0] = -5.0  # B clamps to -1 → 0
    argb = denormalize_and_pack_argb(chunk)
    assert torch.all((argb >= 0) & (argb <= 255))
    # BGRA byte order: channel 0 = B (input B → 0),
    # channel 1 = G (input G → 255), channel 2 = R (input R → 0).
    assert int(argb[0, 0, 0, 0]) == 0    # B
    assert int(argb[0, 0, 0, 1]) == 255  # G
    assert int(argb[0, 0, 0, 2]) == 0    # R
    assert int(argb[0, 0, 0, 3]) == 255  # alpha


def test_within_1_lsb_of_reference():
    """Production kernel uses rounding; reference uses truncation.

    The two should agree within 1 LSB per channel for arbitrary input.
    """
    chunk = _random_chunk(T=2, H=16, W=16)
    a = denormalize_and_pack_argb(chunk)
    b = denormalize_and_pack_argb_reference(chunk)
    diff = a.to(torch.int16) - b.to(torch.int16)
    max_abs = int(diff.abs().max().item())
    assert max_abs <= 1, f"pack vs reference diverged by {max_abs} LSB"


def test_argb_channel_order_red_green_blue():
    """Memory byte order is B, G, R, A — matches NVENC's NV_ENC_BUFFER_FORMAT_ARGB.

    NVIDIA names the format by 32-bit word layout (A in MSB → B in LSB).
    On a little-endian host, the memory bytes are therefore B, G, R, A.
    Tested by feeding pure-red input and verifying the bytes:
        channel 0 (B) = 0
        channel 1 (G) = 0
        channel 2 (R) = 255
        channel 3 (A) = 255
    """
    chunk = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    chunk[0, 0, 0, 0, 0] = 1.0    # R
    chunk[0, 1, 0, 0, 0] = -1.0   # G
    chunk[0, 2, 0, 0, 0] = -1.0   # B
    argb = denormalize_and_pack_argb(chunk)
    assert int(argb[0, 0, 0, 0]) == 0     # B (channel 0)
    assert int(argb[0, 0, 0, 1]) == 0     # G (channel 1)
    assert int(argb[0, 0, 0, 2]) == 255   # R (channel 2)
    assert int(argb[0, 0, 0, 3]) == 255   # alpha (channel 3)


def test_preserves_input_device():
    chunk_cpu = _random_chunk(T=1, H=4, W=4)
    argb_cpu = denormalize_and_pack_argb(chunk_cpu)
    assert argb_cpu.device == chunk_cpu.device


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected pixel chunk"):
        denormalize_and_pack_argb(torch.zeros((3, 8, 8), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="expected pixel chunk"):
        denormalize_and_pack_argb(
            torch.zeros((1, 4, 1, 8, 8), dtype=torch.bfloat16)
        )


def test_multi_frame_independence():
    """Each frame must denormalize independently; no cross-frame leakage."""
    T, H, W = 3, 4, 4
    chunk = torch.zeros((1, 3, T, H, W), dtype=torch.float32)
    chunk[0, :, 0, :, :] = -1.0   # frame 0 all -1 -> 0
    chunk[0, :, 1, :, :] = 0.0    # frame 1 all  0 -> 128 (rounded)
    chunk[0, :, 2, :, :] = 1.0    # frame 2 all  1 -> 255
    argb = denormalize_and_pack_argb(chunk)
    assert int(argb[0, 0, 0, 1]) == 0
    assert int(argb[1, 0, 0, 1]) == 128
    assert int(argb[2, 0, 0, 1]) == 255
