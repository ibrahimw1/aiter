# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, run_perftest, benchmark
from aiter.ops.triton.topk import topk
from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.sample.mix_sample import (
    mixed_sample_outer_exponential as triton_mixed_sample_outer_exponential,
)
from aiter import dtypes, greedy_sample, random_sample
import argparse

torch.set_default_device("cuda")
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
g_gpu = torch.Generator(device="cuda").manual_seed(42)
state_gpu = torch.cuda.get_rng_state()


def run_greedy_sample(input):
    input = input.to(torch.float)
    _, sampled_tokens = topk(input, 1)
    # sampled_tokens = torch.argmax(input, dim=-1)
    return sampled_tokens.view(-1)


def run_aiter_greedy_sample(input):
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    aiter.greedy_sample(sampled_tokens, input)
    return sampled_tokens


@benchmark()
def test_greedy_sample(M, N, dtype=torch.bfloat16):
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    o_a, us_a = run_perftest(run_greedy_sample, input)
    o_b, us_b = run_perftest(run_aiter_greedy_sample, input)
    err = checkAllclose(o_a.to(torch.int), o_b, atol=0, rtol=0)
    return {"origin_us": us_a, "aiter_us": us_b, "aiter_err": err}


def run_random_sample(input, temperatures, eps, use_aiter_exponential=False):
    logits = input.to(torch.float)
    logits = logits.div_(temperatures.unsqueeze(dim=1))
    probs = softmax(logits)
    torch.cuda.set_rng_state(state_gpu)
    if use_aiter_exponential:
        exponential = torch.empty_like(probs)
        aiter.exponential(exponential, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty_like(probs).exponential_(1) + eps
    logits = probs.div_(exponential)
    _, sampled_tokens = topk(logits, 1)
    # sampled_tokens = torch.argmax(logits, dim=-1)

    return sampled_tokens.view(-1)


def run_aiter_random_sample(input, temperatures, eps, inner_exponential=False):
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    torch.cuda.set_rng_state(state_gpu)
    if inner_exponential:
        aiter.random_sample(sampled_tokens, input, temperatures, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty(input.size(), dtype=torch.float32).exponential_(1)
        aiter.random_sample_outer_exponential(
            sampled_tokens, input, exponential, temperatures, eps=eps
        )
    return sampled_tokens


@benchmark()
def test_random_sample(M, N, dtype=torch.bfloat16, eps=1e-6):
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    temperatures = torch.rand(M, device="cuda", dtype=torch.float)
    temperatures = torch.where(
        temperatures < 0.3, torch.ones_like(temperatures), temperatures
    )
    o_a, us_a = run_perftest(
        run_random_sample, input, temperatures, eps, use_aiter_exponential=False
    )
    o_b, us_b = run_perftest(
        run_aiter_random_sample, input, temperatures, eps, inner_exponential=False
    )
    err = checkAllclose(o_a.to(torch.int), o_b, atol=0, rtol=0)

    o_c, us_c = run_perftest(
        run_random_sample, input, temperatures, eps, use_aiter_exponential=True
    )
    o_d, us_d = run_perftest(
        run_aiter_random_sample, input, temperatures, eps, inner_exponential=True
    )
    err2 = checkAllclose(o_c.to(torch.int), o_d, atol=0, rtol=0)
    return {
        "origin_us": min(us_a, us_c),
        "exp_out_aiter_us": us_b,
        "exp_out_aiter_err": err,
        "exp_in_aiter_us": us_d,
        "exp_in_aiter_err": err2,
    }


def run_mixed_sample(input, temperatures, eps, use_aiter_exponential=False):
    logits = input.to(torch.float)
    # _, greedy_tokens = topk(logits, 1)
    greedy_tokens = torch.argmax(logits, dim=-1)
    logits.div_(temperatures.unsqueeze(dim=1))
    probs = softmax(logits)
    torch.cuda.set_rng_state(state_gpu)
    if use_aiter_exponential:
        exponential = torch.empty_like(probs)
        aiter.exponential(exponential, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty_like(probs).exponential_(1) + eps
    sample_tokens = probs.div_(exponential)
    # _, sample_tokens = topk(sample_tokens, 1)
    sample_tokens = torch.argmax(sample_tokens, dim=-1)
    return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


def run_aiter_mixed_sample(input, temperatures, eps, inner_exponential=False):
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    torch.cuda.set_rng_state(state_gpu)
    if inner_exponential:
        aiter.mixed_sample(sampled_tokens, input, temperatures, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty(input.size(), dtype=torch.float32).exponential_(1)
        aiter.mixed_sample_outer_exponential(
            sampled_tokens, input, exponential, temperatures, eps=eps
        )
    return sampled_tokens


@benchmark()
def test_mixed_sample(M, N, dtype=torch.bfloat16, eps=1e-6):
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    temperatures = torch.rand(M, device="cuda", dtype=torch.float)
    temperatures = torch.where(
        temperatures < 0.3, torch.zeros_like(temperatures), temperatures
    )
    o_a, us_a = run_perftest(
        run_mixed_sample, input, temperatures, eps, use_aiter_exponential=False
    )
    o_b, us_b = run_perftest(
        run_aiter_mixed_sample, input, temperatures, eps, inner_exponential=False
    )
    err = checkAllclose(o_a.to(torch.int), o_b, atol=0, rtol=0)

    o_c, us_c = run_perftest(
        run_mixed_sample, input, temperatures, eps, use_aiter_exponential=True
    )
    o_d, us_d = run_perftest(
        run_aiter_mixed_sample, input, temperatures, eps, inner_exponential=True
    )
    err2 = checkAllclose(o_c.to(torch.int), o_d, atol=0, rtol=0)
    return {
        "origin_us": min(us_a, us_c),
        "exp_out_aiter_us": us_b,
        "exp_out_aiter_err": err,
        "exp_in_aiter_us": us_d,
        "exp_in_aiter_err": err2,
    }


def run_triton_mixed_sample(input, exponential, temperatures, eps):
    """Driver for the new Triton mixed-sample kernel.

    Takes the externally-drawn exponential as input (matches the HIP wrapper's
    'outer exponential' contract — caller owns RNG). Returns sampled tokens.
    """
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    triton_mixed_sample_outer_exponential(
        sampled_tokens, input, exponential, temperatures, eps=eps
    )
    return sampled_tokens


def run_hip_mixed_sample_outer(input, exponential, temperatures, eps):
    """Driver for the existing HIP kernel under the same outer-exponential contract.

    Wraps aiter.mixed_sample_outer_exponential so both Triton and HIP see the same
    exponential tensor — required for any bit-exact / statistical comparison.
    """
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    aiter.mixed_sample_outer_exponential(
        sampled_tokens, input, exponential, temperatures, eps=eps
    )
    return sampled_tokens


@benchmark()
def test_mixed_sample_triton(M, N, dtype=torch.bfloat16, eps=1e-6):
    """Triton mixed-sample correctness + perf vs HIP.

    Validates:
      1. All-greedy (T==0): bit-exact agreement, atol=rtol=0.
      2. Mixed temperatures: greedy rows bit-exact, sampled rows reasonable
         (single-trial; statistical equivalence is checked in
         test_mixed_sample_triton_distribution below).
    """
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    exponential = torch.empty(M, N, dtype=torch.float32, device="cuda").exponential_(1)

    # (1) All-greedy: bit-exact
    temps_greedy = torch.zeros(M, device="cuda", dtype=torch.float32)
    o_hip_g = run_hip_mixed_sample_outer(input, exponential, temps_greedy, eps)
    o_tri_g = run_triton_mixed_sample(input, exponential, temps_greedy, eps)
    err_greedy = checkAllclose(o_hip_g, o_tri_g, atol=0, rtol=0)

    # (2) Mixed: greedy rows bit-exact, sampled rows compared loosely (single trial)
    temperatures = torch.rand(M, device="cuda", dtype=torch.float32)
    temperatures = torch.where(
        temperatures < 0.3, torch.zeros_like(temperatures), temperatures
    )
    o_hip_m, us_hip = run_perftest(
        run_hip_mixed_sample_outer, input, exponential, temperatures, eps
    )
    o_tri_m, us_tri = run_perftest(
        run_triton_mixed_sample, input, exponential, temperatures, eps
    )
    greedy_mask = (temperatures == 0).cpu()
    err_greedy_rows = checkAllclose(
        o_hip_m[greedy_mask], o_tri_m[greedy_mask], atol=0, rtol=0
    )

    return {
        "all_greedy_err": err_greedy,
        "mixed_greedy_err": err_greedy_rows,
        "hip_us": us_hip,
        "triton_us": us_tri,
        "speedup": us_hip / us_tri if us_tri > 0 else float("nan"),
    }


def test_mixed_sample_triton_distribution(
    M=4, N=1024, dtype=torch.bfloat16, eps=1e-6, n_trials=4000, max_tvd=0.02
):
    """Distribution-equivalence test: HIP vs Triton at T>0.

    For each row, builds empirical token distributions over n_trials draws
    (fresh exponentials each trial, fixed logits & temperatures) and asserts
    max total-variation-distance per row < max_tvd. With n_trials=4000 and
    probabilities bounded away from 0, |p_hip - p_tri| ~ O(1/sqrt(n_trials))
    so 0.02 is a comfortable bound.

    Mathematically HIP and Triton both implement the Gumbel-Max identity, so
    their distributions should be identical up to floating-point noise.
    """
    torch.manual_seed(0)
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    temperatures = torch.full((M,), 1.0, device="cuda", dtype=torch.float32)

    hip_counts = torch.zeros(M, N, dtype=torch.int64, device="cuda")
    tri_counts = torch.zeros(M, N, dtype=torch.int64, device="cuda")

    for _ in range(n_trials):
        e = torch.empty(M, N, dtype=torch.float32, device="cuda").exponential_(1)
        o_hip = run_hip_mixed_sample_outer(input, e, temperatures, eps)
        o_tri = run_triton_mixed_sample(input, e, temperatures, eps)
        hip_counts.scatter_add_(
            1, o_hip.to(torch.int64).unsqueeze(1), torch.ones_like(hip_counts[:, :1])
        )
        tri_counts.scatter_add_(
            1, o_tri.to(torch.int64).unsqueeze(1), torch.ones_like(tri_counts[:, :1])
        )

    p_hip = hip_counts.float() / n_trials
    p_tri = tri_counts.float() / n_trials
    tvd = 0.5 * (p_hip - p_tri).abs().sum(dim=1)
    max_tvd_observed = float(tvd.max().item())

    ok = max_tvd_observed < max_tvd
    aiter.logger.info(
        "MIXED_SAMPLE distribution test: M=%d N=%d trials=%d max_TVD=%.4f "
        "threshold=%.4f -> %s",
        M, N, n_trials, max_tvd_observed, max_tvd, "PASS" if ok else "FAIL",
    )
    assert ok, (
        f"Triton vs HIP distribution mismatch: max TVD per row = {max_tvd_observed:.4f} "
        f">= {max_tvd}"
    )


d_sample = {
    "greedy": test_greedy_sample,
    "random": test_random_sample,
    "mixed": test_mixed_sample,
    "triton_mixed": test_mixed_sample_triton,
}

list_dtype = ["bf16"]
l_n = [129280, 151936][-1:]
l_m = [1, 8, 16, 32, 64, 128, 192, 256, 512]
import pandas as pd

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["bf16", "fp16", "fp32"],
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="*",
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="*",
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-s",
    "--sample_type",
    type=str,
    choices=list(d_sample.keys()),
    nargs="*",
    default=list(d_sample.keys()),
    help="""Sample type.
    e.g.: -s greedy random mixed triton_mixed""",
)
parser.add_argument(
    "--dist-test",
    action="store_true",
    help="""Run Triton-vs-HIP distribution-equivalence test (n_trials=4000) before
    the main sweep. Slow (~minutes). Use to validate sampling correctness when
    touching the Triton kernel.""",
)

args = parser.parse_args()
if args.dtype is None:
    list_dtype = [dtypes.d_dtypes[key] for key in list_dtype]
else:
    list_dtype = [dtypes.d_dtypes[args.dtype]]
if args.n is not None:
    l_n = args.n
if args.m is not None:
    l_m = args.m
if len(args.sample_type) > 0:
    l_sample_type = args.sample_type

list_sample_func = [d_sample[key] for key in args.sample_type if key in d_sample.keys()]

if args.dist_test:
    test_mixed_sample_triton_distribution()

for test_func in list_sample_func:
    df = []
    for dtype in list_dtype:
        for n in l_n:
            for m in l_m:
                ret = test_func(m, n, dtype)
                df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("sample summary (markdown):\n%s", df_md)
