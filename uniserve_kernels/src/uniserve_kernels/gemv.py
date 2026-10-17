"""FP32 GEMV with serial strided accumulation and a specified lane tree.

Sixteen lanes each accumulate k=lane, lane+16, ... with one rounded FP32
FMA per term. Lane partials combine at distances8,4,2,1. Output tiling may
change scheduling, but must preserve these arithmetic dependencies.
"""

import torch

from uniserve_kernels.triton import tl, triton

if triton is not None:

    @triton.jit
    def _gemv(
        x,
        weight,
        bias,
        out,
        n: tl.constexpr,
        k: tl.constexpr,
        sx: tl.constexpr,
        sw0: tl.constexpr,
        sw1: tl.constexpr,
        sb: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        rows = tl.program_id(0) * 8 + tl.arange(0, 8)
        lane = tl.arange(0, 16)
        accumulator = tl.full((8, 16), 0.0, tl.float32)
        for iteration in range(tl.cdiv(k, 16)):
            column = iteration * 16 + lane
            valid = column < k
            a = tl.load(x + column * sx, mask=valid, other=0)
            b = tl.load(
                weight
                + rows[:, None].to(tl.int64) * sw0
                + column[None, :] * sw1,
                mask=(rows[:, None] < n) & valid[None, :],
                other=0,
            )
            accumulator = tl.where(
                valid[None, :], tl.fma(b, a[None, :], accumulator), accumulator
            )
        for stage in tl.static_range(4, 0, -1):
            distance = 1 << (stage - 1)
            partner = tl.gather(
                accumulator,
                tl.broadcast_to(
                    tl.minimum(lane + distance, 15)[None, :], (8, 16)
                ),
                axis=1,
            )
            accumulator = tl.where(
                lane[None, :] < distance, accumulator + partner, accumulator
            )
        if has_bias:
            accumulator = (
                accumulator
                + tl.load(bias + rows * sb, mask=rows < n, other=0)[:, None]
            )
        destination = tl.broadcast_to(out + rows[:, None], (8, 16))
        tl.store(
            destination,
            accumulator,
            mask=(rows[:, None] < n) & (lane[None, :] == 0),
        )


def strided16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write one dense FP32 row using a fixed FMA/reduction order.

    Operands are borrowed CUDA views; output is contiguous. No workspace,
    model-specific dimensions, or retained execution state is required.
    """
    if triton is None or not x.is_cuda:
        raise ValueError("strided16 GEMV requires CUDA and Triton")
    if weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError("strided16 GEMV requires a nonempty weight[N,K]")
    n, k = weight.shape
    if x.shape != (1, k) or out.shape != (1, n):
        raise ValueError("strided16 GEMV requires one input/output row")
    values = (x, weight, out) if bias is None else (x, weight, out, bias)
    if (
        any(
            value.dtype != torch.float32 or value.device != x.device
            for value in values
        )
        or not out.is_contiguous()
    ):
        raise ValueError(
            "strided16 GEMV requires FP32 operands on one device "
            "and contiguous output"
        )
    if bias is not None and bias.shape != (n,):
        raise ValueError("strided16 GEMV bias must match output channels")
    _gemv[(triton.cdiv(n, 8),)](
        x,
        weight,
        bias,
        out,
        n,
        k,
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        0 if bias is None else bias.stride(0),
        bias is not None,
        num_warps=4,
        enable_fp_fusion=True,
    )
    return out
