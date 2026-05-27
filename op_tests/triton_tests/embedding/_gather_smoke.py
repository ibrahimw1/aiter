"""Standalone correctness check for aiter.ops.triton.embedding.gather.

Pure load/store kernel — should be bit-exact vs F.embedding for all inputs.
"""
import sys
import torch
import torch.nn.functional as F
from aiter.ops.triton.embedding.gather import gather

CASES = [
    # (N, V, D, dtype)
    (1,     128,    64,    torch.bfloat16),
    (4,     32000,  2048,  torch.bfloat16),
    (128,   201088, 2880,  torch.bfloat16),  # gpt-oss
    (512,   201088, 2880,  torch.bfloat16),  # gpt-oss longer batch
    (1024,  151936, 4096,  torch.bfloat16),  # qwen
    (4,     32000,  2048,  torch.float16),
    (4,     32000,  2048,  torch.float32),
]


def run(N, V, D, dtype):
    torch.manual_seed(20)
    weight = torch.randn(V, D, device="cuda", dtype=dtype)
    indices = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)
    out_tri = gather(indices, weight)
    out_ref = F.embedding(indices, weight)
    ok = torch.equal(out_tri, out_ref)
    status = "OK " if ok else "FAIL"
    print(f"[{status}] N={N:>5} V={V:>7} D={D:>5} dtype={str(dtype):<14}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for c in CASES:
        all_ok &= run(*c)

    # i32 indices
    weight = torch.randn(8192, 1024, device="cuda", dtype=torch.bfloat16)
    indices_i32 = torch.randint(0, 8192, (32,), device="cuda", dtype=torch.int32)
    ok = torch.equal(gather(indices_i32, weight),
                     F.embedding(indices_i32.to(torch.int64), weight))
    print(f"[{'OK ' if ok else 'FAIL'}] i32 indices")
    all_ok &= ok

    # leading shape preservation
    weight = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 4096, (3, 7), device="cuda", dtype=torch.int64)
    out_tri = gather(indices, weight)
    out_ref = F.embedding(indices, weight)
    ok = out_tri.shape == (3, 7, 512) and torch.equal(out_tri, out_ref)
    print(f"[{'OK ' if ok else 'FAIL'}] leading shape preserved")
    all_ok &= ok

    print("\nALL PASS" if all_ok else "\nFAILURES")
    sys.exit(0 if all_ok else 1)
