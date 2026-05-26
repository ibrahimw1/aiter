# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.embedding.gather import (
    _embedding_gather_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gather(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Triton drop-in for `F.embedding(indices, weight)` (no padding_idx,
    no max_norm, no scale_grad).

    Replaces aten::indexSelectSmallIndex on the embedding lookup hot path.
    Caller MUST guarantee 0 <= indices[n] < weight.shape[0]; out-of-range
    rows are NOT clamped (matches F.embedding's undefined-behaviour contract).

    Args:
        indices: (N,) integer tensor of token IDs.
        weight:  (V, D) embedding matrix.

    Returns:
        out: (N, D) tensor with same dtype as `weight`. If `indices` is not
        flat, the leading shape is preserved: returned shape is
        `indices.shape + (D,)`.
    """
    assert weight.dim() == 2, f"expected (V, D) weight, got {tuple(weight.shape)}"
    out_lead_shape = indices.shape
    indices_flat = indices.contiguous().view(-1)
    # AITER Triton kernels read garbage from non-contiguous view tensors.
    # Same precedent as gpt-oss MHA/RMSNorm fixes; just hardcode the guard.
    weight = weight.contiguous()

    N = indices_flat.numel()
    V, D = weight.shape

    # 1024 columns/block: with D up to ~16k (largest LLM hidden) this is at most
    # 16 col-blocks per row; under that we fit in a single block. 1024*4B fp32 =
    # 4KB per program, well within LDS budget — leaves room for the pipeliner.
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    _LOGGER.info(
        f"EMBEDDING_GATHER: N={N} V={V} D={D} dtype={weight.dtype}"
    )

    grid = (N, triton.cdiv(D, BLOCK_D))
    _embedding_gather_kernel[grid](
        indices_flat,
        weight,
        out,
        weight.stride(0),
        out.stride(0),
        D,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out.view(*out_lead_shape, D)
