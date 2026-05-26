"""Standalone statistical smoke for aiter.ops.triton.rng.exponential.

Checks: positivity, mean~=1, var~=1, KS-test vs Exp(1) CDF, same-seed determinism.
"""
import math
import sys
import torch
from aiter.ops.triton.rng.exponential import exponential


def check_moments(shape, seed):
    x = exponential(shape, seed=seed)
    N = x.numel()
    mean = float(x.mean().item())
    var = float(x.var(unbiased=False).item())
    se = 5.0 / math.sqrt(N)
    pos = bool((x > 0).all().item())
    fin = bool(torch.isfinite(x).all().item())
    ok = pos and fin and abs(mean - 1.0) < se and abs(var - 1.0) < 5 * se
    print(f"[{'OK ' if ok else 'FAIL'}] shape={str(shape):<14} N={N:>7} "
          f"mean={mean:.4f} var={var:.4f} pos={pos} fin={fin}")
    return ok


def check_ks():
    N = 100_000
    x = exponential((N,), seed=2024)
    xs, _ = torch.sort(x.float())
    emp = torch.arange(1, N + 1, dtype=torch.float32, device=x.device) / N
    ana = 1.0 - torch.exp(-xs)
    ks = float((emp - ana).abs().max().item())
    ok = ks < 0.01
    print(f"[{'OK ' if ok else 'FAIL'}] KS vs Exp(1): {ks:.5f} (threshold 0.01)")
    return ok


def check_determinism():
    a = exponential((16, 4096), seed=42)
    b = exponential((16, 4096), seed=42)
    c = exponential((16, 4096), seed=43)
    ok = torch.equal(a, b) and not torch.equal(a, c)
    print(f"[{'OK ' if ok else 'FAIL'}] determinism (same seed equal, different seed differs)")
    return ok


if __name__ == "__main__":
    all_ok = True
    for shape, seed in [((1024,), 20), ((16, 32000), 21), ((2, 201088), 22)]:
        all_ok &= check_moments(shape, seed)
    all_ok &= check_ks()
    all_ok &= check_determinism()
    print("\nALL PASS" if all_ok else "\nFAILURES")
    sys.exit(0 if all_ok else 1)
