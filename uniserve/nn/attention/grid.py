"""Exact rectangular row/column/reference attention over borrowed resources.

The complete sequence is cell-contiguous in the caller's order. Camera zero
is the reference row; every query sees its column, its camera row, and that
reference row. Clean conditioning slots are deliberately not part of this
attention contract. This inference composition runs after sequence-to-head
exchange and before the inverse exchange. It creates no groups or caches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch
from torch import nn

from uniserve.distributed.mesh import Communicator
from uniserve.distributed.tokens import HeadExchange, TokenShard
from uniserve.nn import _binding
from uniserve.tensors import BufferConfig


@dataclass(frozen=True, slots=True)
class AttentionState:
    """One dense segment's BF16 output and natural-log FP32 normalizer.

    Output is contiguous ``[batch, queries, heads, width]``; LSE is
    contiguous ``[batch, heads, queries]``. The record borrows both tensors.
    Segment providers must not return base-2 LSE or an unnormalized sum.
    """

    output: torch.Tensor
    lse: torch.Tensor

    def __post_init__(self):
        if (
            self.output.ndim != 4
            or min(self.output.shape) < 1
            or self.output.dtype != torch.bfloat16
            or not self.output.is_contiguous()
            or self.lse.shape
            != (
                self.output.shape[0],
                self.output.shape[2],
                self.output.shape[1],
            )
            or self.lse.dtype != torch.float32
            or self.lse.device != self.output.device
            or not self.lse.is_contiguous()
        ):
            raise ValueError(
                "attention states require contiguous BF16 B,Q,H,D output "
                "and FP32 B,H,Q natural-log LSE"
            )


class SegmentOperator(Protocol):
    """Borrow dense BF16 B,Q,H,D / B,K,H,D projections and return their state.

    The provider owns its execution resources. Every segment is noncausal,
    with no padding or mask. Q/K/V already have positional transforms.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
        out: AttentionState | None = None,
    ) -> AttentionState: ...


@dataclass(frozen=True, slots=True)
class SegmentShape:
    batch: int
    queries: int
    keys: int


@dataclass(frozen=True, slots=True, eq=False)
class GridLayout:
    """Complete rectangular cell layout with caller-owned device indices.

    Build with ``from_cells`` before capture. ``canonical`` maps camera-major
    tokens to original tokens, or is None for identity. Column indices gather
    from camera-major tokens; ``column_restore`` maps those tokens to their
    positions in the column gather. No index tensor or request is cached.
    Prefix tokens, padding, and cells without a camera row are unsupported.
    """

    views: int
    columns: int
    cell_tokens: int
    canonical: torch.Tensor | None
    column_queries: torch.Tensor
    column_keys: torch.Tensor
    column_restore: torch.Tensor

    def __post_init__(self):
        if (
            type(self.views) is not int
            or self.views < 2
            or any(
                type(n) is not int or n < 1
                for n in (self.columns, self.cell_tokens)
            )
        ):
            raise ValueError(
                "grid attention requires two views and "
                "positive columns/cell tokens"
            )
        indices = (
            (self.canonical, self.num_tokens),
            (self.column_queries, self.num_tokens),
            (self.column_keys, self.num_tokens - self.row_tokens),
            (self.column_restore, self.num_tokens),
        )
        if any(
            value is not None
            and (
                value.shape != (count,)
                or value.dtype != torch.int64
                or value.device != self.column_queries.device
                or not value.is_contiguous()
            )
            for value, count in indices
        ):
            raise ValueError(
                "grid indices must cover their complete token domains"
            )

    @classmethod
    def from_cells(
        cls,
        cell_camera: torch.Tensor,
        cell_column: torch.Tensor,
        cell_tokens: int,
        *,
        device: torch.device | str,
    ) -> GridLayout:
        """Validate a rectangular bijection and prepare its token indices.

        Reads cell tables on the CPU and places the resulting indices on the
        caller's device. Cell labels must be contiguous from zero; camera
        zero is the reference. Original cells may have any bijective order.
        """
        if (
            type(cell_tokens) is not int
            or cell_tokens < 1
            or cell_camera.ndim != 1
            or not cell_camera.numel()
            or cell_column.shape != cell_camera.shape
            or any(
                t.dtype not in (torch.int32, torch.int64)
                for t in (cell_camera, cell_column)
            )
        ):
            raise ValueError(
                "grid cells require paired nonempty integer tables "
                "and positive token count"
            )
        camera, column = (
            t.detach().to(device="cpu", dtype=torch.int64)
            for t in (cell_camera, cell_column)
        )
        if bool((camera < 0).any()) or bool((column < 0).any()):
            raise ValueError(
                "grid cells must have nonnegative camera and column labels"
            )
        views, columns = int(camera.max()) + 1, int(column.max()) + 1
        count = camera.numel()
        slot = camera * columns + column
        if (
            views < 2
            or views * columns != count
            or not torch.equal(slot.sort().values, torch.arange(count))
        ):
            raise ValueError(
                "cell tables must be a rectangular bijection "
                "with at least two views"
            )
        positions = torch.arange(cell_tokens)
        cell_order = slot.argsort()
        canonical = None
        if not torch.equal(cell_order, torch.arange(count)):
            canonical = (
                (cell_order[:, None] * cell_tokens + positions)
                .flatten()
                .to(device)
            )
        raster = torch.arange(count * cell_tokens).view(
            views, columns, cell_tokens
        )
        queries = raster.transpose(0, 1).reshape(-1)
        keys = raster[1:].transpose(0, 1).reshape(-1)
        return cls(
            views,
            columns,
            cell_tokens,
            canonical,
            queries.to(device),
            keys.to(device),
            queries.argsort().to(device),
        )

    @property
    def num_tokens(self) -> int:
        return self.views * self.columns * self.cell_tokens

    @property
    def row_tokens(self) -> int:
        return self.columns * self.cell_tokens

    @property
    def column_tokens(self) -> int:
        return self.views * self.cell_tokens

    @property
    def segments(self) -> Mapping[str, SegmentShape]:
        """Dense rectangles whose states enter the signed merge."""
        return MappingProxyType(
            {
                "row": SegmentShape(
                    self.views - 1, self.row_tokens, self.row_tokens
                ),
                "column": SegmentShape(
                    self.columns,
                    self.column_tokens,
                    (self.views - 1) * self.cell_tokens,
                ),
                "reference": SegmentShape(1, self.num_tokens, self.row_tokens),
                "cell": SegmentShape(
                    (self.views - 1) * self.columns,
                    self.cell_tokens,
                    self.cell_tokens,
                ),
            }
        )

    def buffer_configs(
        self, heads: int, head_dim: int
    ) -> Mapping[str, BufferConfig]:
        """Exact numerical buffer extents for runtime preparation.

        These are composition buffers, not provider-internal scratch.
        Runtime bindings lend them across serialized calls; returned values
        are copied to caller-owned output before another call reuses them.
        """
        if any(type(n) is not int or n < 1 for n in (heads, head_dim)):
            raise ValueError(
                "grid buffer head dimensions must be positive integers"
            )
        result = {}
        for name, shape in self.segments.items():
            result[f"{name}.output"] = BufferConfig(
                (shape.batch, shape.queries, heads, head_dim), torch.bfloat16
            )
            result[f"{name}.lse"] = BufferConfig(
                (shape.batch, heads, shape.queries), torch.float32
            )
        for name, rows in (
            ("column.query", self.num_tokens),
            ("column.key", self.num_tokens - self.row_tokens),
            ("column.value", self.num_tokens - self.row_tokens),
            ("output", self.num_tokens),
        ):
            result[name] = BufferConfig((rows, heads, head_dim), torch.bfloat16)
        result["lse"] = BufferConfig((heads, self.num_tokens), torch.float32)
        if self.canonical is not None:
            for name in ("query", "key", "value"):
                result[f"canonical.{name}"] = BufferConfig(
                    (self.num_tokens, heads, head_dim), torch.bfloat16
                )
        return MappingProxyType(result)


def merge_states(
    layout: GridLayout,
    *,
    row: AttentionState,
    column: AttentionState,
    reference: AttentionState,
    cell: AttentionState,
    out: AttentionState | None = None,
) -> AttentionState:
    """Merge A+B′+C−D1; reference queries use B′+C, in original token order.

    Multiplication and addition/subtraction follow the native FP32 order.
    A/D1 placeholders for reference queries are excluded from the maximum.
    Return output and union LSE in the original model order, not canonical
    order. Every state and index tensor remains borrowed and unchanged.
    """
    heads, width = reference.output.shape[2:]
    if out is not None and out.output.shape != (
        1,
        layout.num_tokens,
        heads,
        width,
    ):
        raise ValueError(
            "merged grid destination must cover the complete sequence"
        )
    states = {
        "row": row,
        "column": column,
        "reference": reference,
        "cell": cell,
    }
    for name, state in states.items():
        shape = layout.segments[name]
        if (
            state.output.shape != (shape.batch, shape.queries, heads, width)
            or state.output.device != layout.column_queries.device
        ):
            raise ValueError(
                f"{name} attention state does not match its grid segment"
            )
    from uniserve_kernels.attention.grid import merge_grid_states

    output, lse = merge_grid_states(
        tuple(
            states[name].output
            for name in ("row", "column", "reference", "cell")
        ),
        tuple(
            states[name].lse for name in ("row", "column", "reference", "cell")
        ),
        layout.column_restore,
        layout.canonical,
        row_tokens=layout.row_tokens,
        column_tokens=layout.column_tokens,
        cell_tokens=layout.cell_tokens,
        out=None if out is None else out.output[0],
        lse_out=None if out is None else out.lse[0],
    )
    return AttentionState(output.unsqueeze(0), lse.unsqueeze(0))


class GridAttention(nn.Module):
    """Exact grid attention on local token projections and a global layout.

    Head counts describe the architecture. ``parallelize_`` binds Ulysses:
    inputs then cover this rank's contiguous token interval with TP-local
    heads. Positional transforms precede the exchange. Outputs restore the
    same interval and heads; the complete grid is never inferred from a shard.
    An ExecutionContext supplies the operator and reusable workspace. Explicit
    operators remain available for standalone numerical calls.
    """

    def __init__(
        self, num_heads: int, head_dim: int, *, scale: float | None = None
    ):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (num_heads, head_dim)):
            raise ValueError("grid head dimensions must be positive integers")
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("attention scale must be finite and positive")
        self.num_heads = self.local_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5 if scale is None else scale
        self.exchange = HeadExchange(Communicator())

    def forward(
        self,
        q,
        k,
        v,
        layout: GridLayout,
        *,
        out: torch.Tensor | None = None,
        operator: SegmentOperator | None = None,
    ):
        partition = TokenShard(layout.num_tokens, self.exchange.group)
        if any(
            value.ndim != 3
            or value.shape != q.shape
            or value.shape
            != (
                partition.count,
                self.local_heads * self.exchange.group.size,
                self.head_dim,
            )
            or value.dtype != torch.bfloat16
            or value.device != layout.column_queries.device
            for value in (q, k, v)
        ):
            raise ValueError(
                "grid projections must cover the complete BF16 token sequence "
                "or its bound local interval and head partition"
            )
        if out is not None and (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
        ):
            raise ValueError(
                "grid output must match its local query representation"
            )
        if torch.is_grad_enabled() and any(
            value.requires_grad for value in (q, k, v)
        ):
            raise RuntimeError("grid attention is inference-only")
        binding = _binding.grid.get().get(id(self))
        buffers = storage = None
        if operator is None:
            if binding is None:
                # Standalone calls own their temporary resources. Runtime
                # capture requires a binding warmed at the numerical shape.
                from uniserve.runtime.backends.attention.grid import resolve
                from uniserve.runtime.bindings import capturing

                if capturing(q.device):
                    raise RuntimeError(
                        "grid capture requires an ExecutionContext"
                    )
                operator = resolve("auto", q.device)
            else:
                operator, buffers, storage = binding.prepare(layout, q)

        destination = torch.empty_like(q) if out is None else out
        if self.exchange.group.size > 1:
            q, k, v = (
                value[: layout.num_tokens]
                for value in self.exchange.heads(
                    tuple(partition.pad(t) for t in (q, k, v)), storage=storage
                )
            )
        result = self._compute(q, k, v, layout, operator, buffers)
        if self.exchange.group.size > 1:
            physical = partition.capacity * self.exchange.group.size
            if physical != layout.num_tokens:
                padded = result.new_zeros((physical, *result.shape[1:]))
                padded[: layout.num_tokens].copy_(result)
                result = padded
            result = self.exchange.tokens(result, storage=storage)[
                : partition.count
            ]
        destination.copy_(result)
        return destination

    def _compute(self, q, k, v, layout, operator, buffers):
        heads, width = q.shape[1:]

        def gather(tensor, indices, name):
            if buffers is None:
                return tensor.index_select(0, indices)
            return torch.index_select(tensor, 0, indices, out=buffers[name])

        def state(name):
            if buffers is None:
                return None
            return AttentionState(
                buffers[f"{name}.output"], buffers[f"{name}.lse"]
            )

        query, key, value = (
            (
                tensor
                if layout.canonical is None
                else gather(tensor, layout.canonical, f"canonical.{name}")
            )
            for name, tensor in zip(
                ("query", "key", "value"), (q, k, v), strict=True
            )
        )
        nonreference = tuple(
            tensor[layout.row_tokens :] for tensor in (query, key, value)
        )
        cell = operator(
            *(
                tensor.reshape(
                    (layout.views - 1) * layout.columns,
                    layout.cell_tokens,
                    heads,
                    width,
                )
                for tensor in nonreference
            ),
            scale=self.scale,
            out=state("cell"),
        )
        row = operator(
            *(
                tensor.reshape(
                    layout.views - 1, layout.row_tokens, heads, width
                )
                for tensor in nonreference
            ),
            scale=self.scale,
            out=state("row"),
        )
        reference = operator(
            query.unsqueeze(0),
            key[: layout.row_tokens].unsqueeze(0),
            value[: layout.row_tokens].unsqueeze(0),
            scale=self.scale,
            out=state("reference"),
        )
        column = operator(
            gather(query, layout.column_queries, "column.query").reshape(
                layout.columns, layout.column_tokens, heads, width
            ),
            gather(key, layout.column_keys, "column.key").reshape(
                layout.columns, -1, heads, width
            ),
            gather(value, layout.column_keys, "column.value").reshape(
                layout.columns, -1, heads, width
            ),
            scale=self.scale,
            out=state("column"),
        )
        # Segment states retain no projections. Release canonical copies
        # before allocating the merged output, as in the native composition.
        del query, key, value, nonreference
        return merge_states(
            layout,
            row=row,
            column=column,
            reference=reference,
            cell=cell,
            out=None
            if buffers is None
            else AttentionState(
                buffers["output"].unsqueeze(0), buffers["lse"].unsqueeze(0)
            ),
        ).output[0]
