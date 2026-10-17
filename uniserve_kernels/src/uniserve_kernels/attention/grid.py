"""Signed softmax-state merge for rectangular row/column/reference unions."""

from __future__ import annotations

import torch

from uniserve_kernels.triton import tl, triton

if triton is not None:

    @triton.jit
    def _merge(
        out,
        union_lse,
        oa_ptr,
        ob_ptr,
        oc_ptr,
        od_ptr,
        la_ptr,
        lb_ptr,
        lc_ptr,
        ld_ptr,
        column_restore,
        destination,
        tokens: tl.constexpr,
        row: tl.constexpr,
        column: tl.constexpr,
        cell: tl.constexpr,
        heads: tl.constexpr,
        width: tl.constexpr,
        scatter: tl.constexpr,
    ):
        # The native domain has 798,720 tokens. Head/width offsets must not
        # overflow int32, including when all 40 heads run on one rank.
        j = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        jc = tl.load(column_restore + j).to(tl.int64)
        d = tl.arange(0, width)
        ja = tl.maximum(j - row, 0)
        dest = tl.load(destination + j).to(tl.int64) if scatter else j

        oa = tl.load(oa_ptr + (ja * heads + h) * width + d).to(tl.float32)
        ob = tl.load(ob_ptr + (jc * heads + h) * width + d).to(tl.float32)
        oc = tl.load(oc_ptr + (j * heads + h) * width + d).to(tl.float32)
        od = tl.load(od_ptr + (ja * heads + h) * width + d).to(tl.float32)
        la = tl.load(la_ptr + ((ja // row) * heads + h) * row + ja % row)
        lb = tl.load(
            lb_ptr + ((jc // column) * heads + h) * column + jc % column
        )
        lc = tl.load(lc_ptr + h * tokens + j)
        ld = tl.load(ld_ptr + ((ja // cell) * heads + h) * cell + ja % cell)

        # Reference queries have no A or D1. Excluding their placeholder
        # LSE from the maximum is essential: a large placeholder otherwise
        # underflows the valid B'/C weights even with a zero A coefficient.
        nonreference = (j >= row).to(tl.float32)
        la = tl.where(j >= row, la, -float("inf"))
        ld = tl.where(j >= row, ld, -float("inf"))
        maximum = tl.maximum(tl.maximum(la, lb), lc)
        wa = nonreference * tl.exp(la - maximum)
        wb = tl.exp(lb - maximum)
        wc = tl.exp(lc - maximum)
        wd = -nonreference * tl.exp(ld - maximum)
        denominator = wa + wb + wc + wd
        numerator = wa * oa + wb * ob + wc * oc + wd * od
        tl.store(
            out + (dest * heads + h) * width + d,
            (numerator / denominator).to(tl.bfloat16),
        )
        tl.store(union_lse + h * tokens + dest, maximum + tl.log(denominator))


def merge_grid_states(
    outputs: tuple[torch.Tensor, ...],
    lses: tuple[torch.Tensor, ...],
    column_restore: torch.Tensor,
    destination: torch.Tensor | None,
    *,
    row_tokens: int,
    column_tokens: int,
    cell_tokens: int,
    out: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge ordered A,B′,C,D1 states; return model-order output and LSE.

    Inputs use dense-segment output B,Q,H,D and natural-log LSE B,H,Q.
    ``column_restore`` addresses B' for each canonical camera-major token.
    Optional ``destination`` maps canonical tokens to original model order.
    CUDA executes the native FP32 expression in a fused kernel; CPU evaluates
    the same ordered expression. No token-pair mask or scores are allocated.
    """
    if len(outputs) != 4 or len(lses) != 4 or outputs[2].ndim != 4:
        raise ValueError("grid merge requires four ordered output/LSE states")
    _, tokens, heads, width = outputs[2].shape
    if (
        min(heads, width) < 1
        or any(
            type(n) is not int or n < 1
            for n in (row_tokens, column_tokens, cell_tokens)
        )
        or row_tokens % cell_tokens
        or column_tokens % cell_tokens
        or column_tokens // cell_tokens < 2
        or tokens != row_tokens * (column_tokens // cell_tokens)
    ):
        raise ValueError("grid merge segment extents must form a rectangle")
    views, columns = column_tokens // cell_tokens, row_tokens // cell_tokens
    expected = (
        (views - 1, row_tokens),
        (columns, column_tokens),
        (1, tokens),
        ((views - 1) * columns, cell_tokens),
    )
    device = outputs[2].device
    for output, lse, (batch, rows) in zip(outputs, lses, expected, strict=True):
        if (
            output.shape != (batch, rows, heads, width)
            or lse.shape != (batch, heads, rows)
            or output.dtype != torch.bfloat16
            or lse.dtype != torch.float32
            or any(
                t.device != device or not t.is_contiguous()
                for t in (output, lse)
            )
        ):
            raise ValueError(
                "grid merge requires matching contiguous BF16/FP32 states"
            )
    for indices in (column_restore, destination):
        if indices is not None and (
            indices.shape != (tokens,)
            or indices.dtype != torch.int64
            or indices.device != device
            or not indices.is_contiguous()
        ):
            raise ValueError(
                "grid merge indices must cover the complete sequence"
            )
    for tensor, shape, dtype in (
        (out, (tokens, heads, width), torch.bfloat16),
        (lse_out, (heads, tokens), torch.float32),
    ):
        if tensor is not None and (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "grid merge destinations must match output and LSE"
            )
    output = (
        out
        if out is not None
        else torch.empty(
            (tokens, heads, width), device=device, dtype=torch.bfloat16
        )
    )
    lse = (
        lse_out
        if lse_out is not None
        else torch.empty((heads, tokens), device=device, dtype=torch.float32)
    )
    if device.type == "cuda":
        if triton is None:
            raise RuntimeError("CUDA grid merge requires Triton")
        if width & (width - 1) or width > 256:
            raise ValueError(
                "CUDA grid merge requires power-of-two head width at most 256"
            )
        _merge[(tokens, heads)](
            output,
            lse,
            *outputs,
            *lses,
            column_restore,
            column_restore if destination is None else destination,
            tokens=tokens,
            row=row_tokens,
            column=column_tokens,
            cell=cell_tokens,
            heads=heads,
            width=width,
            scatter=destination is not None,
            num_warps=1,
        )
        return output, lse
    if device.type != "cpu":
        raise ValueError("grid merge supports CPU or CUDA tensors")

    index = torch.arange(tokens, device=device)
    nonreference = index >= row_tokens
    local = (index - row_tokens).clamp_min(0)
    a, b, c, d = (state.flatten(0, 1).float() for state in outputs)
    la, lb, lc, ld = (
        state.transpose(1, 2).reshape(-1, heads) for state in lses
    )
    a, d, la, ld = a[local], d[local], la[local], ld[local]
    b, lb = b[column_restore], lb[column_restore]
    la = la.masked_fill(~nonreference[:, None], -torch.inf)
    ld = ld.masked_fill(~nonreference[:, None], -torch.inf)
    maximum = torch.maximum(torch.maximum(la, lb), lc)
    active = nonreference[:, None].float()
    wa = active * torch.exp(la - maximum)
    wb = torch.exp(lb - maximum)
    wc = torch.exp(lc - maximum)
    wd = -active * torch.exp(ld - maximum)
    denominator = wa + wb + wc + wd
    numerator = (
        wa[..., None] * a
        + wb[..., None] * b
        + wc[..., None] * c
        + wd[..., None] * d
    )
    merged = (numerator / denominator[..., None]).to(torch.bfloat16)
    normalizer = (maximum + denominator.log()).T.contiguous()
    if destination is None:
        output.copy_(merged)
        lse.copy_(normalizer)
    else:
        output.index_copy_(0, destination, merged)
        lse.index_copy_(1, destination, normalizer)
    return output, lse
