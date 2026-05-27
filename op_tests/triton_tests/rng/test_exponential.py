# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Statistical tests for aiter.ops.triton.rng.exponential.

NOT bit-exact vs torch.exponential_ -- Triton uses its own RNG, PyTorch uses
Philox. We check the distribution is Exp(rate=1):
  - mean   ~ 1.0 (true value, asymptotic SE ~ 1/sqrt(N))
  - var    ~ 1.0
  - min    > 0  (no log(0) hazard)
  - K-S statistic small vs the analytical CDF 1 - exp(-x)
"""
import math
import pytest
import torch

from aiter.ops.triton.rng.exponential import exponential


@pytest.mark.parametrize("shape", [(1024,), (16, 32000), (2, 201088)])
def test_moments(shape):
    """Sample mean/var must approach (1.0, 1.0). With N >= 1024 the asymptotic
    SE is 1/sqrt(N) ~ 0.03; we allow 5x that = 0.15 absolute tolerance."""
    x = exponential(shape, seed=20)
    assert x.shape == shape
    assert x.dtype == torch.float32
    assert (x > 0).all(), f"got non-positive samples: min={x.min().item()}"
    assert torch.isfinite(x).all(), "got non-finite samples"

    N = x.numel()
    mean = float(x.mean().item())
    var = float(x.var(unbiased=False).item())
    se = 5.0 / math.sqrt(N)
    assert abs(mean - 1.0) < se, f"mean {mean:.4f} too far from 1.0 (5*SE={se:.4f})"
    assert abs(var - 1.0) < 5 * se, f"var {var:.4f} too far from 1.0"


def test_determinism_same_seed():
    """Same seed -> bit-exact output. Different seeds -> different output."""
    a = exponential((16, 4096), seed=42)
    b = exponential((16, 4096), seed=42)
    assert torch.equal(a, b), "same seed must produce identical output"
    c = exponential((16, 4096), seed=43)
    assert not torch.equal(a, c), "different seeds must produce different output"


def test_ks_against_analytical_cdf():
    """Kolmogorov-Smirnov 1-sample test vs the analytical Exp(1) CDF.

    For N=100k samples, the 99% critical KS statistic is ~1.63/sqrt(N) ~ 0.005.
    We allow a generous 0.01 to keep the test non-flaky across seeds.
    """
    N = 100_000
    x = exponential((N,), seed=2024)
    xs, _ = torch.sort(x.float())
    # empirical CDF at the sorted positions: F_n(x_(i)) = i/N
    empirical = torch.arange(1, N + 1, dtype=torch.float32, device=x.device) / N
    analytical = 1.0 - torch.exp(-xs)
    ks = float((empirical - analytical).abs().max().item())
    assert ks < 0.01, f"KS statistic {ks:.4f} exceeds 0.01"


def test_implicit_seed_progresses():
    """When seed is None, consecutive calls must produce different output
    (the CPU-side seed counter advances). This is the property that lets
    callers in the sampler hot path get fresh noise per decode step."""
    a = exponential((1024,))
    b = exponential((1024,))
    assert not torch.equal(a, b), (
        "implicit-seed calls returned identical output -- seed counter not advancing"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_cast(dtype):
    """Wrapper supports dtype != float32 via a final cast."""
    x = exponential((512,), dtype=dtype, seed=7)
    assert x.dtype == dtype
    assert (x > 0).all()
