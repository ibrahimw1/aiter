# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import warnings

import torch
import torch.nn.functional as F
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.embedding.gather import (
    _embedding_gather_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Per-shape validation state (item 6 in fork/triton review). Each (V, D, dtype)
# tuple is validated at most once per process; subsequent calls skip the
# F.embedding cross-check. Shapes where Triton output diverges from F.embedding
# beyond bf16 rounding fall back permanently to F.embedding for that shape.
_VALIDATED_SHAPES: set[tuple[int, int, torch.dtype]] = set()
_FALLBACK_SHAPES: set[tuple[int, int, torch.dtype]] = set()


def _gather_fake_tensor(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """torch.library fake tensor for ``gather``. Returns a same-shape /
    same-dtype empty tensor without launching the kernel — lets
    ``torch.compile`` trace the op without materializing the result."""
    return torch.empty(
        *indices.shape, weight.shape[1], dtype=weight.dtype, device=weight.device
    )


def _validate_against_reference(
    out_tri: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor
) -> bool:
    """One-shot bit-exact compare vs F.embedding. Returns True if outputs
    agree exactly (gather is a pure load/store so anything less is a bug).
    Called only when ATOM_VALIDATE_TRITON_EMBEDDING=1 and the shape has not
    been validated yet."""
    out_ref = F.embedding(indices, weight)
    return torch.equal(out_tri, out_ref)


@torch_compile_guard(gen_fake=_gather_fake_tensor)
def gather(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Triton drop-in for `F.embedding(indices, weight)` (no padding_idx,
    no max_norm, no scale_grad).

    Replaces aten::indexSelectSmallIndex on the embedding lookup hot path.
    Caller MUST guarantee 0 <= indices[n] < weight.shape[0]; out-of-range
    rows are NOT clamped (matches F.embedding's undefined-behaviour contract).

    Defensive fallback: when ATOM_VALIDATE_TRITON_EMBEDDING=1, the first call
    per (V, D, dtype) shape compares against F.embedding bit-exactly. On
    divergence, that shape is added to a per-process fallback set and all
    subsequent calls for it go to F.embedding. Adds one extra call per
    unique shape per process — only enabled when explicitly requested.

    Args:
        indices: integer tensor of token IDs (any shape).
        weight:  (V, D) embedding matrix.

    Returns:
        out: tensor with shape ``indices.shape + (D,)`` and same dtype as
        ``weight``.
    """
    assert weight.dim() == 2, f"expected (V, D) weight, got {tuple(weight.shape)}"
    V, D = weight.shape
    shape_key = (V, D, weight.dtype)

    # Per-shape fallback (set at first divergence; sticky for process life).
    if shape_key in _FALLBACK_SHAPES:
        return F.embedding(indices, weight)

    out_lead_shape = indices.shape
    indices_flat = indices.contiguous().view(-1)
    # AITER Triton kernels read garbage from non-contiguous view tensors.
    # Same precedent as gpt-oss MHA/RMSNorm fixes; just hardcode the guard.
    weight = weight.contiguous()

    N = indices_flat.numel()

    # 1024 columns/block: with D up to ~16k (largest LLM hidden) this is at most
    # 16 col-blocks per row; under that we fit in a single block. 1024*4B fp32 =
    # 4KB per program, well within LDS budget — leaves room for the pipeliner.
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    out = torch.empty(N, D, dtype=weight.dtype, device=weight.device)

    _LOGGER.info(f"EMBEDDING_GATHER: N={N} V={V} D={D} dtype={weight.dtype}")

    grid = (N, triton.cdiv(D, BLOCK_D))
    _embedding_gather_kernel[grid](
        indices_flat,
        weight,
        out,
        weight.stride(0),
        out.stride(0),
        D,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    out = out.view(*out_lead_shape, D)

    # One-shot validation: only fires when the env is set, only the first
    # time we see a (V, D, dtype) triple. Compares bit-exact against
    # F.embedding; on divergence flips the shape to permanent fallback.
    if (
        os.environ.get("ATOM_VALIDATE_TRITON_EMBEDDING", "0") == "1"
        and shape_key not in _VALIDATED_SHAPES
    ):
        if _validate_against_reference(out, indices, weight):
            _VALIDATED_SHAPES.add(shape_key)
        else:
            _FALLBACK_SHAPES.add(shape_key)
            warnings.warn(
                f"aiter.ops.triton.embedding.gather: Triton output diverges "
                f"from F.embedding at shape (V={V}, D={D}, dtype={weight.dtype}). "
                f"Falling back permanently for this shape.",
                RuntimeWarning,
                stacklevel=2,
            )
            return F.embedding(indices, weight)
    return out
