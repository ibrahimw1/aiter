"""Standalone SWA smoke check — verifies the wrapper relax + kernel SLIDING_WINDOW
path against the in-tree reference on gpt-oss shapes. Runs in <30s.

Usage (inside rocm/atom-dev container with aiter mounted):
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python3 _swa_smoke.py
"""
import sys
import torch
from aiter.ops.triton.attention.mha import flash_attn_func
from aiter.test_mha_common import attention_ref

CASES = [
    # (seqlen_q, seqlen_k, window, n_q_heads, n_kv_heads, head_dim)
    (128,  128,  128, 64, 8, 64),   # gpt-oss prefill, 1 block
    (256,  256,  128, 64, 8, 64),   # window < seqlen, n-block-skip
    (1024, 1024, 128, 64, 8, 64),   # long prefill
    (511,  511,  128, 64, 8, 64),   # non-aligned
    (128,  128,  1,   64, 8, 64),   # tightest window
    (2048, 2048, 128, 64, 8, 64),   # longer
]

def run_case(sq, sk, win, hq, hkv, hd):
    torch.manual_seed(20)
    dtype = torch.bfloat16
    q = torch.randn((1, sq, hq, hd), device="cuda", dtype=dtype)
    k = torch.randn((1, sk, hkv, hd), device="cuda", dtype=dtype)
    v = torch.randn((1, sk, hkv, hd), device="cuda", dtype=dtype)
    window_size = (win - 1, 0)
    triton_out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True, window_size=window_size)
    torch_out, _, _ = attention_ref(q, k, v, dropout_p=0.0, causal=True, window_size=window_size)
    a = triton_out.detach().to(torch.float64).flatten()
    b = torch_out.detach().to(torch.float64).flatten()
    cos = (a @ b) / (a.norm() * b.norm() + 1e-30)
    max_abs = (triton_out - torch_out).abs().max().item()
    ok = cos.item() > 0.9999 and max_abs < 1e-1
    tag = "OK " if ok else "FAIL"
    print(f"[{tag}] sq={sq:>5} sk={sk:>5} win={win:>4} "
          f"cos={cos.item():.8f} max_abs={max_abs:.3e}")
    return ok

if __name__ == "__main__":
    all_ok = True
    for c in CASES:
        all_ok &= run_case(*c)
    print("\nALL PASS" if all_ok else "\nFAILURES")
    sys.exit(0 if all_ok else 1)
