# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for aiter.ops.triton.sample.mix_sample.

References against a pure-PyTorch implementation of the same spec — independent
of the HIP `aiter.mixed_sample_outer_exponential` (which has a regression on
recent aiter main that produces always-token-0 output).

Greedy path (temperature == 0): bit-exact vs torch.argmax (lower-index-wins).
Stochastic path (temperature > 0): given the same exponentials, the Triton
kernel and the torch reference both compute a deterministic function of
(input, exponentials, temperatures). They must agree bit-exact on the argmax.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.sample.mix_sample import mixed_sample_outer_exponential


def _torch_reference(
    input: torch.Tensor,
    exponentials: torch.Tensor,
    temperatures: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """PyTorch implementation of the kernel's spec, lower-index-wins on ties.

    For each row m:
        T = temperatures[m]
        if T == 0: out[m] = argmin_i{ i : input[m, i] == max_j input[m, j] }
        else:      out[m] = argmin_i{ i : score[m, i] == max_j score[m, j] }
                   where score = softmax(input/T) / (exponentials + eps)

    Equivalently (and what we compute here for numerical-determinism parity with
    the kernel's online-softmax form): for T>0, use
        score = exp(input/T - max(input/T)) / (exponentials + eps)
    The omitted softmax denominator is a per-row constant — doesn't change argmax.
    Note this means the magnitudes differ from a full-softmax computation but the
    argmax is identical, which is what we test.
    """
    M, N = input.shape
    out = torch.empty(M, dtype=torch.int32, device=input.device)
    x = input.to(torch.float32)
    for m in range(M):
        T = float(temperatures[m].item())
        if T == 0.0:
            row = x[m]
        else:
            T_inv = 1.0 / max(T, 1e-5)
            xm = x[m] * T_inv
            xm = xm - xm.max()
            row = xm.exp() / (exponentials[m] + eps)
        # torch.argmax returns the smallest index on ties — matches kernel.
        out[m] = int(row.argmax().item())
    return out


def _check(
    M: int,
    N: int,
    dtype: torch.dtype,
    temps: torch.Tensor,
    seed: int,
    eps: float = 1e-10,
    logits_scale: float = 1.0,
    tag: str = "",
):
    torch.manual_seed(seed)
    input = (torch.randn(M, N, device="cuda", dtype=dtype) * logits_scale)
    exponentials = torch.empty(M, N, device="cuda", dtype=torch.float32).exponential_(1)

    out_tri = torch.empty(M, dtype=torch.int32, device="cuda")
    mixed_sample_outer_exponential(out_tri, input, exponentials, temps, eps=eps)
    out_ref = _torch_reference(input, exponentials, temps, eps)

    # Diagnostics
    diff = (out_tri != out_ref).nonzero(as_tuple=False).flatten().tolist()
    if diff:
        sample = diff[:5]
        rows = []
        for m in sample:
            T = float(temps[m].item())
            rows.append(f"row {m}: T={T:.3g} tri={int(out_tri[m].item())} "
                        f"ref={int(out_ref[m].item())}")
        msg = (f"[{tag}] M={M} N={N} dtype={dtype} mismatches={len(diff)}/{M} "
               + " | ".join(rows))
        raise AssertionError(msg)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "M,N",
    [
        (1, 32),         # tiny
        (4, 1024),       # small batch
        (16, 32000),     # llama-like vocab
        (4, 201088),     # gpt-oss vocab
    ],
)
def test_greedy_bit_exact(M: int, N: int, dtype: torch.dtype):
    """T=0 on every row → bit-exact argmax over the row."""
    temps = torch.zeros(M, device="cuda", dtype=torch.float32)
    _check(M, N, dtype, temps, seed=20, tag="greedy")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "M,N",
    [
        (1, 32),
        (4, 1024),
        (16, 32000),
        (4, 201088),
    ],
)
@pytest.mark.parametrize("T", [0.7, 1.0, 1.7])
def test_stochastic_deterministic(M: int, N: int, dtype: torch.dtype, T: float):
    """T>0 on every row, fixed exponentials → bit-exact vs torch reference."""
    temps = torch.full((M,), T, device="cuda", dtype=torch.float32)
    _check(M, N, dtype, temps, seed=21, tag=f"stoch_T{T}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("M,N", [(8, 32000), (16, 201088)])
def test_mixed_temperatures(M: int, N: int, dtype: torch.dtype):
    """Half greedy, half stochastic in the same batch — covers per-row branch."""
    temps = torch.zeros(M, device="cuda", dtype=torch.float32)
    temps[M // 2:] = 0.8
    _check(M, N, dtype, temps, seed=22, tag="mixed")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_extreme_logits(dtype: torch.dtype):
    """Very large positive + negative logits — checks online-softmax numerical stability."""
    M, N = 4, 8192
    temps = torch.full((M,), 1.0, device="cuda", dtype=torch.float32)
    _check(M, N, dtype, temps, seed=23, logits_scale=50.0, tag="extreme")


def test_first_index_wins_on_tie():
    """All-equal logits → greedy must return 0 (lower-index-wins on tie)."""
    M, N = 4, 1024
    input = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    exponentials = torch.empty(M, N, device="cuda", dtype=torch.float32).exponential_(1)
    temps = torch.zeros(M, device="cuda", dtype=torch.float32)
    out = torch.empty(M, dtype=torch.int32, device="cuda")
    mixed_sample_outer_exponential(out, input, exponentials, temps)
    assert torch.all(out == 0), f"expected all 0 (first-index-wins), got {out.tolist()}"
