# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


# Sentinel matches the HIP kernel's `-FLT_MAX` init for thread_kvp.value.
# Using -inf would propagate NaN through the first-iteration rescale
# (-inf * exp(-inf - newmax) = -inf * 0 = NaN), corrupting the running argmax.
# Must be tl.constexpr so Triton can read it inside @triton.jit functions.
_NEG_HUGE_F32 = tl.constexpr(-3.4028235e38)


@triton.jit
def _mixed_sample_outer_exponential_kernel(
    out_ptr,            # *i32, (M,)
    in_ptr,             # *bf16/fp16/fp32, (M, N)
    exp_ptr,            # *fp32, (M, N)
    temp_ptr,           # *fp32, (M,)
    N,
    stride_in_m,
    stride_exp_m,
    eps,
    BLOCK_N: tl.constexpr,
):
    m_idx = tl.program_id(0)
    T = tl.load(temp_ptr + m_idx)

    in_row = in_ptr + m_idx * stride_in_m
    exp_row = exp_ptr + m_idx * stride_exp_m

    best_score = tl.full((), _NEG_HUGE_F32, tl.float32)
    best_idx = tl.zeros((), tl.int32)

    if T == 0.0:
        # Greedy: argmax over fp32-cast logits, lower-index-wins on ties
        # (matches HIP `argmax_impl` -> hipcub::ArgMax).
        for k in tl.range(0, N, BLOCK_N, num_stages=2):
            offs = k + tl.arange(0, BLOCK_N)
            mask = offs < N
            x = tl.load(
                in_row + offs, mask=mask, other=_NEG_HUGE_F32, cache_modifier=".cg"
            ).to(tl.float32)
            tile_max = tl.max(x, axis=0)
            tile_arg = tl.argmax(x, axis=0, tie_break_left=True)
            tile_idx_global = (k + tile_arg).to(tl.int32)
            take = (tile_max > best_score) | (
                (tile_max == best_score) & (tile_idx_global < best_idx)
            )
            best_score = tl.where(take, tile_max, best_score)
            best_idx = tl.where(take, tile_idx_global, best_idx)
    else:
        # Sampled path: HIP-faithful online softmax + division by (e + eps).
        # Per-element score = exp(x/T - running_max) / (e + eps); after a tile
        # raises running_max, previously-accumulated best is rescaled by
        # exp(old_max - new_max). Argmax across all tiles gives the Gumbel-Max
        # sampling identity.
        T_inv = 1.0 / tl.maximum(T, 1e-5)
        running_max = tl.full((), _NEG_HUGE_F32, tl.float32)

        for k in tl.range(0, N, BLOCK_N, num_stages=2):
            offs = k + tl.arange(0, BLOCK_N)
            mask = offs < N
            x = (
                tl.load(
                    in_row + offs,
                    mask=mask,
                    other=_NEG_HUGE_F32,
                    cache_modifier=".cg",
                ).to(tl.float32)
                * T_inv
            )
            # other=1.0 keeps masked positions out of the division (1.0 + eps > 0)
            # and pairs with score=-inf masking below.
            e = tl.load(
                exp_row + offs, mask=mask, other=1.0, cache_modifier=".cg"
            )

            tile_max = tl.max(x, axis=0)
            new_max = tl.maximum(running_max, tile_max)
            correction = tl.exp(running_max - new_max)
            best_score = best_score * correction
            running_max = new_max

            smx = tl.exp(x - new_max)
            score = smx / (e + eps)
            score = tl.where(mask, score, _NEG_HUGE_F32)

            tile_best = tl.max(score, axis=0)
            tile_arg = tl.argmax(score, axis=0, tie_break_left=True)
            tile_idx_global = (k + tile_arg).to(tl.int32)
            take = (tile_best > best_score) | (
                (tile_best == best_score) & (tile_idx_global < best_idx)
            )
            best_score = tl.where(take, tile_best, best_score)
            best_idx = tl.where(take, tile_idx_global, best_idx)

    tl.store(out_ptr + m_idx, best_idx)
