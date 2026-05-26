"""Standalone correctness check for aiter.ops.triton.sample.mix_sample.

References against a pure-PyTorch implementation (independent of HIP, which is
currently broken on aiter main). Runs in <30s inside rocm/atom-dev.

Usage:
    python3 _mix_sample_smoke.py
"""
import sys
import torch
from aiter.ops.triton.sample.mix_sample import mixed_sample_outer_exponential


def torch_ref(input, exponentials, temperatures, eps):
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
        out[m] = int(row.argmax().item())
    return out


def run(tag, M, N, dtype, T, seed=20, logits_scale=1.0, eps=1e-10, mixed=False):
    torch.manual_seed(seed)
    input = torch.randn(M, N, device="cuda", dtype=dtype) * logits_scale
    e = torch.empty(M, N, device="cuda", dtype=torch.float32).exponential_(1)
    if mixed:
        temps = torch.zeros(M, device="cuda", dtype=torch.float32)
        temps[M // 2:] = T
    else:
        temps = torch.full((M,), T, device="cuda", dtype=torch.float32)
    out_tri = torch.empty(M, dtype=torch.int32, device="cuda")
    mixed_sample_outer_exponential(out_tri, input, e, temps, eps=eps)
    out_ref = torch_ref(input, e, temps, eps)
    diffs = (out_tri != out_ref).sum().item()
    status = "OK " if diffs == 0 else "FAIL"
    print(f"[{status}] {tag:<24} M={M:>3} N={N:>7} dtype={str(dtype):<14} "
          f"T={T:.2f} mismatches={diffs}/{M}")
    if diffs:
        idx = (out_tri != out_ref).nonzero(as_tuple=False).flatten()[:3].tolist()
        for m in idx:
            print(f"   row {m}: tri={int(out_tri[m].item())} ref={int(out_ref[m].item())}")
    return diffs == 0


if __name__ == "__main__":
    cases = []
    # Greedy bit-exact across shapes / dtypes
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        for M, N in [(1, 32), (4, 1024), (16, 32000), (4, 201088)]:
            cases.append(("greedy", M, N, dt, 0.0))
    # Stochastic (deterministic given fixed exponentials)
    for dt in (torch.bfloat16, torch.float32):
        for M, N in [(4, 1024), (16, 32000), (4, 201088)]:
            for T in (0.7, 1.0, 1.7):
                cases.append((f"stoch_T{T}", M, N, dt, T))
    # Mixed-temperature batch
    cases.append(("mixed",     8, 32000, torch.bfloat16, 0.8))
    cases.append(("mixed-vocab", 16, 201088, torch.bfloat16, 0.8))
    # Extreme logits
    cases.append(("extreme",   4, 8192, torch.bfloat16, 1.0))

    all_ok = True
    for c in cases:
        tag, M, N, dt, T = c
        ok = run(tag, M, N, dt, T,
                 mixed=tag.startswith("mixed"),
                 logits_scale=50.0 if tag == "extreme" else 1.0)
        all_ok &= ok
    # tie-break test
    print("--- tie-break (all-equal logits, greedy must return 0) ---")
    input = torch.zeros(4, 1024, device="cuda", dtype=torch.bfloat16)
    e = torch.empty(4, 1024, device="cuda", dtype=torch.float32).exponential_(1)
    temps = torch.zeros(4, device="cuda", dtype=torch.float32)
    out = torch.empty(4, dtype=torch.int32, device="cuda")
    mixed_sample_outer_exponential(out, input, e, temps)
    tie_ok = bool(torch.all(out == 0).item())
    print(f"[{'OK ' if tie_ok else 'FAIL'}] tie-break             out={out.tolist()}")
    all_ok &= tie_ok

    print("\nALL PASS" if all_ok else "\nFAILURES")
    sys.exit(0 if all_ok else 1)
