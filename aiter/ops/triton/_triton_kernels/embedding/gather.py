# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _embedding_gather_kernel(
    indices_ptr,     # *i64/i32, (N,)
    weight_ptr,      # *bf16/fp16/fp32, (V, D)
    out_ptr,         # *<weight dtype>, (N, D)
    stride_weight_row,
    stride_out_row,
    D,
    BLOCK_D: tl.constexpr,
):
    """For row n: out[n, :] = weight[indices[n], :].

    Grid is 2D: (N, cdiv(D, BLOCK_D)). One program handles one (row, col-tile)
    pair. Columns outside D are masked off; row index is trusted (caller
    guarantees 0 <= indices[n] < V — no in-range check, unlike the masked
    variant used for vocab-parallel embeddings).
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    idx = tl.load(indices_ptr + pid_row).to(tl.int64)

    col_start = pid_col * BLOCK_D
    cols = col_start + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    src = weight_ptr + idx * stride_weight_row + cols
    val = tl.load(src, mask=col_mask, other=0.0, cache_modifier=".cg")
    tl.store(out_ptr + pid_row * stride_out_row + cols, val, mask=col_mask)
