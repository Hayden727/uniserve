"""Dense segment operators for the shared exact grid composition.

Grid bindings lend output and LSE buffers to each dense rectangle. Neither
provider registers global state or changes ordinary dense Attention.
"""

from __future__ import annotations

import math

import torch

from uniserve.nn.attention.grid import AttentionState


def _validate(q, k, v, scale, out):
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.shape != k.shape
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
        or min(q.shape) < 1
        or min(k.shape) < 1
        or not math.isfinite(scale)
        or scale <= 0
        or any(
            t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)
        )
    ):
        raise ValueError(
            "dense segments require BF16 B,Q,H,D / B,K,H,D projections"
        )
    if out is not None and (
        out.output.shape != q.shape or out.output.device != q.device
    ):
        raise ValueError("segment destination must match its query")


class TorchSegments:
    """Bounded CPU reference implementation with FP32 scores and softmax.

    This explicit reference provider refuses large score domains. It is not
    a CUDA fallback and cannot accidentally materialize full-grid scores.
    """

    name = "torch"

    def __call__(self, q, k, v, *, scale, out=None):
        _validate(q, k, v, scale, out)
        if q.device.type != "cpu":
            raise ValueError("TorchSegments is a CPU reference provider")
        if q.shape[0] * q.shape[1] * q.shape[2] * k.shape[1] > 2**20:
            raise ValueError(
                "CPU reference segments exceed the bounded score domain"
            )
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
        lse = scores.logsumexp(-1).contiguous()
        output = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())
        if out is None:
            return AttentionState(output.to(torch.bfloat16).contiguous(), lse)
        out.output.copy_(output)
        out.lse.copy_(lse)
        return out


class FlashAttention4Segments:
    """Fused native FA4 segments, preserving its output and LSE convention.

    Each call is a dense rectangle, never the full masked token square.
    Output/LSE destinations are borrowed when supplied. Missing FA4 or
    unsupported CUDA shapes raise directly.
    """

    name = "flash_attn_4"

    def __init__(self):
        from flash_attn.cute.interface import _flash_attn_fwd

        self._forward = _flash_attn_fwd

    def __call__(self, q, k, v, *, scale, out=None):
        _validate(q, k, v, scale, out)
        if q.device.type != "cuda":
            raise ValueError(
                "FlashAttention4Segments requires CUDA projections"
            )
        # Pinned FA4's public flash_attn_func forwards these same defaults to
        # _flash_attn_fwd. This entry additionally accepts borrowed out/lse.
        # Preserve native softcap=0, num_splits=1, pack_gqa=None semantics.
        output, lse, *_ = self._forward(
            q,
            k,
            v,
            softmax_scale=scale,
            softcap=0.0,
            return_lse=True,
            out=None if out is None else out.output,
            lse=None if out is None else out.lse,
        )
        return AttentionState(output.contiguous(), lse.contiguous())


class Sm100Segments:
    """Dense segments using the fixed SM100 BF16 exp2 rounding recipe."""

    name = "flash_attn_4_sm100"

    def __init__(self):
        self.plans = {}

    def __call__(self, q, k, v, *, scale, out=None):
        from uniserve_kernels.attention.dense_sm100 import DenseAttention

        _validate(q, k, v, scale, out)
        if out is None:
            out = AttentionState(
                torch.empty_like(q, memory_format=torch.contiguous_format),
                torch.empty(
                    (q.shape[0], q.shape[2], q.shape[1]),
                    dtype=torch.float32,
                    device=q.device,
                ),
            )
        tensors = (q, k, v, out.output, out.lse)
        signature = tuple(
            (t.shape, t.stride(), t.dtype, t.device) for t in tensors
        )
        plan = self.plans.get(signature)
        if plan is None:
            plan = DenseAttention(*tensors, scale=scale)
            self.plans[signature] = plan
        plan(*tensors, scale=scale)
        return out


def resolve(name, device):
    """Select fused CUDA segments or the bounded explicit CPU reference."""
    if name == "auto":
        name = "flash_attn_4" if device.type == "cuda" else "torch"
    if name == "flash_attn_4":
        return FlashAttention4Segments()
    if name == "flash_attn_4_sm100":
        return Sm100Segments()
    if name == "torch":
        return TorchSegments()
    raise ValueError(f"unknown grid attention backend {name!r}")
