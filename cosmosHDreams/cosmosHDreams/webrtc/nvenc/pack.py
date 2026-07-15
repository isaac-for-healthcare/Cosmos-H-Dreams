"""GPU-side denormalize + ARGB pack for NVENC consumption.

The CosmosH runtime hands the post-decoder pixel chunk to the encode
layer as a bf16 tensor in the value range ``[-1, 1]`` with layout
``[1, 3, T, H, W]`` (``B=1`` batch, ``C=3`` RGB channels, ``T`` frames,
``H``, ``W``). The current (CPU) path copies that to host memory and
casts to uint8 RGB per frame.

The NVENC path needs the chunk as uint8 ``ARGB`` surfaces on the
*same* GPU. PyNvVideoCodec's ``NV_ENC_BUFFER_FORMAT_ARGB`` lays out
each pixel as four bytes in order A, R, G, B (channel 0 = alpha). The
alpha channel is unused by NVENC's color converter but must be present;
we fill it with 255.

This module is pure PyTorch — no NVENC, no aiortc, no CosmosH runtime
dependency — so it can be exercised on a CPU-only dev box for parity
testing against the existing cast.
"""
from __future__ import annotations

import torch


__all__ = ["denormalize_and_pack_argb", "denormalize_and_pack_argb_reference"]


def denormalize_and_pack_argb(chunk: torch.Tensor) -> torch.Tensor:
    """Convert a CosmosH pixel chunk to NVENC-ready uint8 surfaces.

    Args:
        chunk: tensor with shape ``[1, 3, T, H, W]``, dtype usually
            bfloat16 (the runtime emits bf16 from the VAE/TAEHV
            decoder), value range ``[-1, 1]``. May reside on CPU or
            GPU; output is on the same device.

    Returns:
        Tensor with shape ``[T, H, W, 4]``, dtype ``torch.uint8``,
        contiguous, on the input's device. **Byte order in memory is
        B, G, R, A** with alpha fixed at 255.

    Despite the function name, the bytes are laid out as ``BGRA``,
    not ``ARGB``. NVIDIA's ``NV_ENC_BUFFER_FORMAT_ARGB`` names its
    formats by 32-bit DWORD layout (A in the most significant byte,
    B in the least), which is ``BGRA`` byte order in little-endian
    memory. The function name preserves the *format string* passed
    to PyNvVideoCodec's ``CreateEncoder``; the byte order is
    intentionally chosen to match what NVENC actually reads from
    that format string.

    The denormalize step mirrors the existing CPU cast in
    ``cosmosHDreams.webrtc.media.tensor_chunk_to_rgb_frames``:
    ``x_u8 = clip((x + 1) / 2 * 255, 0, 255)``. Implemented here as
    ``x * 127.5 + 127.5`` for one fewer arithmetic op.
    """
    if chunk.ndim != 5 or chunk.shape[0] != 1 or chunk.shape[1] != 3:
        raise ValueError(
            "expected pixel chunk with shape [1, 3, T, H, W]; got "
            f"{tuple(chunk.shape)}"
        )
    # [1, 3, T, H, W] -> [T, H, W, 3] with channels still in R, G, B order.
    rgb = chunk.squeeze(0).permute(1, 2, 3, 0)
    # Compute in float32 for numerical headroom on the clamp+scale, then
    # cast down. PyTorch's .to(uint8) truncates (not rounds), so we add
    # 0.5 explicitly before flooring for correct rounding-to-nearest.
    rgb = (rgb.float().clamp(-1.0, 1.0) * 127.5 + 127.5 + 0.5)
    rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    T, H, W, _ = rgb.shape
    alpha = torch.full(
        (T, H, W, 1), 255, dtype=torch.uint8, device=rgb.device
    )
    # Reorder R, G, B → B, G, R then append A. Memory byte order is
    # B, G, R, A — what NVENC's ARGB format actually expects.
    return torch.cat(
        [rgb[..., 2:3], rgb[..., 1:2], rgb[..., 0:1], alpha], dim=-1
    ).contiguous()


def denormalize_and_pack_argb_reference(chunk: torch.Tensor) -> torch.Tensor:
    """Reference implementation for parity testing.

    Uses the explicit ``(x + 1) / 2 * 255`` form (no rounding) and
    yields a slightly different uint8 rounding than the production
    kernel above. Emits the same BGRA byte order as the production
    function.
    """
    if chunk.ndim != 5 or chunk.shape[0] != 1 or chunk.shape[1] != 3:
        raise ValueError(
            "expected pixel chunk with shape [1, 3, T, H, W]; got "
            f"{tuple(chunk.shape)}"
        )
    rgb = chunk.squeeze(0).permute(1, 2, 3, 0).float()
    rgb = ((rgb + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0).to(torch.uint8)
    T, H, W, _ = rgb.shape
    alpha = torch.full(
        (T, H, W, 1), 255, dtype=torch.uint8, device=rgb.device
    )
    return torch.cat(
        [rgb[..., 2:3], rgb[..., 1:2], rgb[..., 0:1], alpha], dim=-1
    ).contiguous()
