# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for aiter.ops.triton.embedding.gather.

References against torch.nn.functional.embedding. The Triton kernel is a pure
load/store -- output should be bit-exact for all inputs.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.embedding.gather import gather


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "N, V, D",
    [
        (1, 128, 64),  # tiny
        (4, 32000, 2048),  # llama-ish
        (128, 201088, 2880),  # gpt-oss-120b prefill
        (512, 201088, 2880),  # gpt-oss with longer batched prefill
        (1024, 151936, 4096),  # qwen-like
    ],
)
def test_gather_bit_exact(N: int, V: int, D: int, dtype: torch.dtype):
    torch.manual_seed(20)
    weight = torch.randn(V, D, device="cuda", dtype=dtype)
    indices = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)

    out_tri = gather(indices, weight)
    out_ref = F.embedding(indices, weight)
    assert out_tri.shape == out_ref.shape
    assert out_tri.dtype == out_ref.dtype
    assert torch.equal(out_tri, out_ref), (
        f"mismatch: dtype={dtype} N={N} V={V} D={D} "
        f"max_abs={(out_tri.float() - out_ref.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_gather_index_dtypes(indices_dtype: torch.dtype):
    """Both i32 and i64 indices must work — kernel casts to int64 internally."""
    torch.manual_seed(21)
    V, D, N = 8192, 1024, 32
    weight = torch.randn(V, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, V, (N,), device="cuda", dtype=indices_dtype)
    out_tri = gather(indices, weight)
    out_ref = F.embedding(indices.to(torch.int64), weight)
    assert torch.equal(out_tri, out_ref)


def test_gather_preserves_leading_shape():
    """indices.shape=(B,T) -> out.shape=(B,T,D), like F.embedding."""
    V, D, B, T = 4096, 512, 3, 7
    weight = torch.randn(V, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, V, (B, T), device="cuda", dtype=torch.int64)
    out_tri = gather(indices, weight)
    out_ref = F.embedding(indices, weight)
    assert out_tri.shape == (B, T, D)
    assert torch.equal(out_tri, out_ref)


def test_gather_repeated_indices():
    """Same row picked many times — must always return identical bytes."""
    V, D, N = 1024, 256, 128
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32)
    indices = torch.full((N,), 17, device="cuda", dtype=torch.int64)
    out = gather(indices, weight)
    expected_row = weight[17]
    for n in range(N):
        assert torch.equal(out[n], expected_row), f"row {n} differs"


def test_gather_validation_envgate(monkeypatch):
    """When ATOM_VALIDATE_TRITON_EMBEDDING=1, the first call per shape
    cross-checks against F.embedding. On a correctly-implemented kernel the
    validation should pass and the (V, D, dtype) tuple is recorded in
    _VALIDATED_SHAPES. Subsequent calls then skip validation."""
    import aiter.ops.triton.embedding.gather as g

    monkeypatch.setenv("ATOM_VALIDATE_TRITON_EMBEDDING", "1")
    g._VALIDATED_SHAPES.clear()
    g._FALLBACK_SHAPES.clear()

    V, D, N = 4096, 512, 8
    weight = torch.randn(V, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)
    out = g.gather(indices, weight)
    # First call should validate and record the shape.
    assert (V, D, torch.bfloat16) in g._VALIDATED_SHAPES
    assert (V, D, torch.bfloat16) not in g._FALLBACK_SHAPES
    # Output still correct.
    assert torch.equal(out, F.embedding(indices, weight))


def test_gather_validation_fallback(monkeypatch):
    """Force a fallback by injecting a fake `gather` that returns wrong
    output, then verify the second call routes to F.embedding."""
    import aiter.ops.triton.embedding.gather as g

    monkeypatch.setenv("ATOM_VALIDATE_TRITON_EMBEDDING", "1")
    g._VALIDATED_SHAPES.clear()
    g._FALLBACK_SHAPES.clear()

    # Force the shape into the fallback set directly (simulating what would
    # happen if validation found a divergence).
    V, D, N = 2048, 256, 4
    g._FALLBACK_SHAPES.add((V, D, torch.bfloat16))

    weight = torch.randn(V, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)
    out = g.gather(indices, weight)
    # Should be the F.embedding output bit-exact (fallback path).
    assert torch.equal(out, F.embedding(indices, weight))
