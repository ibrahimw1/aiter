# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _exponential_kernel(
    out_ptr,         # *fp32, flat (N,)
    N,
    seed,            # i64 / i32 scalar
    BLOCK: tl.constexpr,
):
    """Fill `out` (flat fp32) with Exp(rate=1) samples via the inverse-CDF
    method.

    tl.rand(seed, offsets) returns U(0, 1] (open at 0, closed at 1), so
    -log(u) gives Exp(1) without the log(0) -> +inf hazard.

    Stride layout is the caller's responsibility: the wrapper allocates a
    contiguous fp32 buffer of N elements and reshapes after.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    u = tl.rand(seed, offs)
    e = -tl.log(u)
    tl.store(out_ptr + offs, e, mask=mask)
