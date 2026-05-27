# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import random as _py_random
import threading as _threading

import torch
import triton

from aiter.ops.triton._triton_kernels.rng.exponential import _exponential_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Module-level seed counter. Cheap (no GPU/aten side effects) and gives a
# unique seed per call within a process. Seeded once from os entropy via
# Python's random; callers wanting cross-run reproducibility should pass an
# explicit `seed=`. Choosing CPU-side counter avoids the aten Philox
# `distribution_elementwise_grid_stride_kernel<unsigned long,2,random_from_to>`
# that fires when we use torch.randint -- which would defeat the purpose of
# replacing the per-element Philox in the first place.
_SEED_LOCK = _threading.Lock()
_SEED_COUNTER = _py_random.SystemRandom().randrange(1 << 31)


def _next_seed() -> int:
    global _SEED_COUNTER
    with _SEED_LOCK:
        _SEED_COUNTER = (_SEED_COUNTER + 0x9E3779B1) & 0x7FFFFFFF  # golden-ratio step
        return _SEED_COUNTER


def exponential(
    shape,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    seed: int | None = None,
) -> torch.Tensor:
    """Triton drop-in for ``torch.empty(shape, dtype, device).exponential_(1)``.

    Returns a tensor of Exp(rate=1) samples. NOT bit-exact vs PyTorch's
    Philox-based exponential_(1) -- uses Triton's own RNG -- but
    distribution-equivalent (mean=1.0, var=1.0) and deterministic given a
    fixed seed.

    Args:
        shape:  output shape (tuple or torch.Size).
        dtype:  output dtype. Only torch.float32 is supported on the kernel
                output (matching PyTorch's exponential_ which produces fp32);
                a different dtype is cast at the end.
        device: output device.
        seed:   per-call RNG seed. If None, drawn from a CPU-side counter
                (no GPU/aten allocation). For cross-run reproducibility pass
                an explicit seed -- torch.manual_seed is NOT honored here
                because honoring it would require the very Philox kernel we
                are replacing.
    """
    if seed is None:
        seed = _next_seed()

    numel = 1
    for s in shape:
        numel *= int(s)
    # Allocate flat fp32 buffer; reshape at the end. Keeping the kernel flat
    # avoids strided pointer arithmetic for the common (M, N) case.
    buf = torch.empty(numel, dtype=torch.float32, device=device)

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _LOGGER.info(f"EXPONENTIAL: shape={tuple(shape)} numel={numel} seed={seed}")
    _exponential_kernel[grid](buf, numel, seed, BLOCK=BLOCK, num_warps=4)

    out = buf.view(*shape)
    if dtype != torch.float32:
        out = out.to(dtype)
    return out
