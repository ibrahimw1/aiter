# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.sample.mix_sample import (
    _mixed_sample_outer_exponential_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def mixed_sample_outer_exponential(
    out: torch.Tensor,
    input: torch.Tensor,
    exponentials: torch.Tensor,
    temperatures: torch.Tensor,
    eps: float = 1e-10,
) -> None:
    """Triton drop-in for aiter.mixed_sample_outer_exponential (HIP).

    For each row m of `input` (shape (M, N), bf16/fp16/fp32):
      * if temperatures[m] == 0 -> writes argmax_i input[m, i]  (greedy)
      * else                    -> writes argmax_i softmax(input[m]/T)_i / (exponentials[m, i] + eps)
                                   (Gumbel-Max with externally-supplied Exp(1) noise)

    Tie-breaking is lower-index-wins, matching the HIP kernel's `hipcub::ArgMax`
    for bit-exact agreement on the greedy path.

    Args:
        out:          (M,)    int32   -- output token indices
        input:        (M, N)  floating -- logits
        exponentials: (M, N)  float32  -- Exp(1) noise sampled by the caller
        temperatures: (M,)    float32  -- per-row temperature; 0.0 = greedy
        eps:          scalar           -- numerical floor for (exponentials + eps)
    """
    assert input.dim() == 2, f"expected (M, N) input, got {tuple(input.shape)}"
    M, N = input.shape
    assert out.shape == (M,), f"out shape {tuple(out.shape)} != (M={M},)"
    assert out.dtype == torch.int32, f"out dtype must be int32, got {out.dtype}"
    assert exponentials.shape == (M, N), (
        f"exponentials shape {tuple(exponentials.shape)} != {(M, N)}"
    )
    assert exponentials.dtype == torch.float32, (
        f"exponentials dtype must be float32, got {exponentials.dtype}"
    )
    assert temperatures.shape == (M,), (
        f"temperatures shape {tuple(temperatures.shape)} != (M={M},)"
    )
    assert temperatures.dtype == torch.float32, (
        f"temperatures dtype must be float32, got {temperatures.dtype}"
    )

    # AITER Triton kernels read garbage from non-contiguous view tensors.
    # See branch ibrahim/triton-swap-gpt-oss-20b, Stage 5 (TritonRMSNorm) for
    # the precedent that motivates this guard.
    input = input.contiguous()
    exponentials = exponentials.contiguous()

    _LOGGER.info(
        f"MIXED_SAMPLE_OUTER_EXPONENTIAL: M={M} N={N} dtype={input.dtype}"
    )

    # 4096 = sweet spot for MI355X (gfx950): 4096*2B + 4096*4B = 24 KB live per
    # stage; with num_stages=2 = 48 KB LDS, well under 160 KB/CU budget; ~49
    # streaming iters for N=201088 (gpt-oss vocab) — enough for the pipeliner
    # to land its overlap without loop overhead dominating.
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    num_warps = 8
    num_stages = 2

    grid = (M,)
    _mixed_sample_outer_exponential_kernel[grid](
        out,
        input,
        exponentials,
        temperatures,
        N,
        input.stride(0),
        exponentials.stride(0),
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
