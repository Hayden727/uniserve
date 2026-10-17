"""Video tile attention with BF16 pooled scores and gated compression.

Packing follows a time, height, width raster. Each spatial tile stores its
valid tokens as a contiguous prefix, followed by zeros. The compression
branch sums in FP32 and rounds each mean to BF16 before its BF16 score,
softmax and value operations. Fine attention uses the existing public
``BlockAttention`` binding; unsupported tile sizes have no dense fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.distributed.mesh import Communicator
from uniserve.distributed.tokens import HeadExchange, TokenShard
from uniserve.nn import _binding

from .inputs import BlockInput, Pattern
from .layer import BlockAttention


def _dimensions(grid, tile_shape):
    if any(
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(type(value) is not int or value < 1 for value in shape)
        for shape in (grid, tile_shape)
    ):
        raise ValueError(
            "video grid and tiles require positive integer triples"
        )


@dataclass(frozen=True, slots=True, eq=False)
class TileLayout:
    """Immutable description borrowing one grid's numerical index tensors.

    ``permutation`` lists the real raster tokens in tile order. ``slots``
    gives their positions in the padded sequence; ``restore`` gathers that
    sequence back into raster order. These three tensors are int64 vectors
    of ``num_tokens`` entries. ``valid_counts`` is one int32 count per tile.
    Callers keep these tensors unchanged while the layout is in use.
    Construction and device placement happen before captured execution;
    neither a layer nor this record caches request or execution resources.
    """

    grid: tuple[int, int, int]
    tile_shape: tuple[int, int, int]
    permutation: torch.Tensor
    slots: torch.Tensor
    restore: torch.Tensor
    valid_counts: torch.Tensor

    def __post_init__(self):
        _dimensions(self.grid, self.tile_shape)
        if any(
            value.shape != (self.num_tokens,)
            or value.dtype != torch.int64
            or value.device != self.valid_counts.device
            for value in (self.permutation, self.slots, self.restore)
        ) or (
            self.valid_counts.shape != (self.num_tiles,)
            or self.valid_counts.dtype != torch.int32
        ):
            raise ValueError(
                "tile indices and valid counts must match the grid"
            )

    @classmethod
    def from_grid(
        cls,
        grid: tuple[int, int, int],
        tile_shape: tuple[int, int, int],
        *,
        device: torch.device | str,
    ) -> TileLayout:
        """Build exact valid-prefix tables on the caller's chosen device."""
        _dimensions(grid, tile_shape)
        raster = torch.arange(math.prod(grid), dtype=torch.int64).view(grid)
        parts, slots, counts = [], [], []
        tile = math.prod(tile_shape)
        for t in range(0, grid[0], tile_shape[0]):
            for h in range(0, grid[1], tile_shape[1]):
                for w in range(0, grid[2], tile_shape[2]):
                    part = raster[
                        t : t + tile_shape[0],
                        h : h + tile_shape[1],
                        w : w + tile_shape[2],
                    ].flatten()
                    start = len(counts) * tile
                    slots.append(torch.arange(start, start + part.numel()))
                    parts.append(part)
                    counts.append(part.numel())
        permutation, positions = torch.cat(parts), torch.cat(slots)
        restore = positions[permutation.argsort()]
        return cls(
            grid,
            tile_shape,
            permutation.to(device),
            positions.to(device),
            restore.to(device),
            torch.tensor(counts, dtype=torch.int32, device=device),
        )

    @property
    def tile(self) -> int:
        return math.prod(self.tile_shape)

    @property
    def num_tokens(self) -> int:
        return math.prod(self.grid)

    @property
    def num_tiles(self) -> int:
        return math.prod(
            (extent + tile - 1) // tile
            for extent, tile in zip(self.grid, self.tile_shape, strict=True)
        )

    @property
    def padded_tokens(self) -> int:
        return self.num_tiles * self.tile

    def keep(
        self, sparsity: float, *, allowed_count: float | None = None
    ) -> int:
        """Resolve the native tile budget from real tokens or mask density.

        Unmasked selection keeps ``ceil((1-sparsity)*num_tokens/tile)``;
        even zero sparsity need not select every padded tile. For a
        structural mask, the caller supplies its architecture's permitted
        key-tile budget, computed before capture. The budget is at least one;
        live per-query counts may be lower after intersection with the mask.
        """
        if not math.isfinite(sparsity) or not 0 <= sparsity < 1:
            raise ValueError("tile sparsity must be finite and in [0, 1)")
        if allowed_count is None:
            domain = self.num_tokens / self.tile
        else:
            if (
                not math.isfinite(allowed_count)
                or not 0 <= allowed_count <= self.num_tiles
            ):
                raise ValueError("allowed tile count must lie within the grid")
            domain = allowed_count
        return max(1, math.ceil((1 - sparsity) * domain))

    def pack(self, rows: torch.Tensor) -> torch.Tensor:
        """Pack leading real token rows; preserve every trailing dimension."""
        if rows.ndim < 1 or rows.shape[0] != self.num_tokens:
            raise ValueError("tile packing requires the grid's real token rows")
        if rows.device != self.permutation.device:
            raise ValueError(
                "tile packing requires indices on the rows' device"
            )
        result = rows.new_zeros((self.padded_tokens, *rows.shape[1:]))
        result[self.slots] = rows[self.permutation]
        return result

    def unpack(self, rows: torch.Tensor) -> torch.Tensor:
        """Restore padded tile rows to the original real-token raster."""
        if rows.ndim < 1 or rows.shape[0] != self.padded_tokens:
            raise ValueError("tile unpacking requires every padded token row")
        if rows.device != self.restore.device:
            raise ValueError(
                "tile unpacking requires indices on the rows' device"
            )
        return rows[self.restore]


def _rows(rows, layout):
    if (
        rows.ndim != 3
        or rows.shape[0] != layout.padded_tokens
        or min(rows.shape[1:]) < 1
        or rows.dtype != torch.bfloat16
        or rows.device != layout.valid_counts.device
    ):
        raise ValueError(
            "tile projections require padded BF16 [rows, heads, width]"
        )


def _mask(allowed, scores_shape, device):
    if allowed is not None and (
        allowed.shape != scores_shape[-2:]
        or allowed.dtype != torch.bool
        or allowed.device != device
    ):
        raise ValueError(
            "tile mask must be bool [query tiles, key tiles] "
            "on the projection device"
        )


def pool(rows: torch.Tensor, layout: TileLayout) -> torch.Tensor:
    """Return BF16 ``[tiles, heads, width]`` means of valid token prefixes."""
    _rows(rows, layout)
    values = rows.reshape(layout.num_tiles, layout.tile, *rows.shape[1:])
    live = (
        torch.arange(layout.tile, device=rows.device)
        < layout.valid_counts[:, None]
    )
    total = values.float().masked_fill(~live[:, :, None, None], 0).sum(1)
    return (total / layout.valid_counts[:, None, None]).to(torch.bfloat16)


def coarse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: TileLayout,
    *,
    scale: float | None = None,
    allowed: torch.Tensor | None = None,
    pooled: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 scores ``[heads, tiles, tiles]`` and compressed tile rows.

    Q/K already carry their positional transforms. Pooled means, matmul
    outputs, scaled scores, softmax and compressed values each round to
    BF16. ``allowed`` masks scores before both softmax and selection. A
    query tile with no permitted keys has zero compressed output. ``pooled``
    can supply already-rounded THD means from a fused preparation provider.
    """
    for rows in (q, k, v):
        _rows(rows, layout)
        if rows.shape != q.shape or rows.device != q.device:
            raise ValueError("tile Q/K/V must share shape and device")
    scale = 1.0 / (q.shape[-1] ** 0.5) if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("tile attention requires a positive finite scale")
    if pooled is None:
        pooled = tuple(pool(rows, layout) for rows in (q, k, v))
    elif len(pooled) != 3 or any(
        value.shape != (layout.num_tiles, *q.shape[1:])
        or value.dtype != q.dtype
        or value.device != q.device
        for value in pooled
    ):
        raise ValueError("pooled Q/K/V must be matching BF16 tile means")
    means = tuple(value.transpose(0, 1) for value in pooled)
    scores = torch.matmul(means[0], means[1].transpose(-1, -2)) * scale
    _mask(allowed, scores.shape, q.device)
    if allowed is None:
        probabilities = scores.softmax(-1)
    else:
        scores = scores.masked_fill(~allowed, -torch.inf)
        live = allowed.any(-1)[None, :, None]
        # Avoid all-masked softmax NaNs without a host read or changing any
        # permitted row's arithmetic.
        probabilities = scores.masked_fill(~live, 0).softmax(-1)
        probabilities = probabilities.masked_fill(~live, 0)
    compressed = torch.matmul(probabilities, means[2])
    return scores, compressed.transpose(0, 1)


def select_tiles(
    scores: torch.Tensor,
    keep: int,
    *,
    allowed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select finite BF16 scores, breaking ties by earlier key-tile index.

    Return int32 indices ``[heads, query tiles, keep]`` and live counts
    ``[heads, query tiles]``. Selected indices are ascending, independent of
    score order. Unread slots after each count hold zero. The optional
    structural mask constrains both scoring and final selection, including
    rows with fewer permitted keys than ``keep``. Scores are assumed finite
    on permitted pairs; forbidden pairs may already contain negative infinity.
    """
    if (
        scores.ndim != 3
        or scores.dtype != torch.bfloat16
        or min(scores.shape) < 1
    ):
        raise ValueError(
            "tile selection requires BF16 "
            "[heads, query tiles, key tiles] scores"
        )
    if type(keep) is not int or not 1 <= keep <= scores.shape[-1]:
        raise ValueError("tile selection count must lie within the key tiles")
    _mask(allowed, scores.shape, scores.device)
    ranked = (
        scores if allowed is None else scores.masked_fill(~allowed, -torch.inf)
    )
    chosen = ranked.argsort(dim=-1, descending=True, stable=True)[..., :keep]
    if allowed is None:
        present = torch.ones_like(chosen, dtype=torch.bool)
    else:
        present = allowed.expand_as(scores).gather(-1, chosen)
    counts = present.sum(-1, dtype=torch.int32)
    # Put unread slots after every selected key, then give them an in-range
    # value so even diagnostic consumers never observe an invalid index.
    indices = chosen.masked_fill(~present, scores.shape[-1]).sort(-1).values
    indices = indices.masked_fill(indices == scores.shape[-1], 0).to(
        torch.int32
    )
    return indices, counts


def add_compressed(
    fine: torch.Tensor,
    compressed: torch.Tensor,
    gate: torch.Tensor,
    layout: TileLayout,
) -> torch.Tensor:
    """Add raw gated tile values, rounding product and sum separately to BF16.

    ``fine`` and ``gate`` are padded token projections; ``compressed`` is
    ``[tiles, heads, width]``. Inputs are borrowed and remain unchanged.
    """
    for rows in (fine, gate):
        _rows(rows, layout)
    if (
        gate.shape != fine.shape
        or compressed.shape != (layout.num_tiles, *fine.shape[1:])
        or compressed.dtype != fine.dtype
        or compressed.device != fine.device
    ):
        raise ValueError(
            "compressed tiles and gate must match the fine projection"
        )
    tiled_gate = gate.reshape(layout.num_tiles, layout.tile, *gate.shape[1:])
    product = compressed[:, None] * tiled_gate
    return fine + product.reshape_as(fine)


class TileAttention(nn.Module):
    """Compose tiled fine attention with BF16 pooled attention for one sample.

    The numerical input is four unpadded ``[tokens, heads, width]`` BF16
    projections. With ``rotary``, Q/K are normalized but unrotated and the
    provider fuses interleaved RoPE, packing and stored-BF16 means. Without
    it Q/K already carry their positional transforms and use the ordinary
    packing/pooling composition. The learned gate is raw. Calls borrow a
    global layout and tile mask.
    Optional static head dimensions allow ``parallelize_`` to bind Ulysses:
    inputs and outputs then cover the rank's contiguous local token interval,
    with all TP-local heads. The runtime owns exchange storage and fine
    attention providers; this module owns no request or metadata cache.
    """

    def __init__(
        self,
        attention: BlockAttention,
        *,
        num_heads: int | None = None,
        head_dim: int | None = None,
    ):
        super().__init__()
        if (num_heads is None) != (head_dim is None) or any(
            type(value) is not int or value < 1
            for value in (num_heads, head_dim)
            if value is not None
        ):
            raise ValueError(
                "tile head dimensions require a positive integer pair"
            )
        self.attention = attention
        self.num_heads = self.local_heads = num_heads
        self.head_dim = head_dim
        self.exchange = HeadExchange(Communicator())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        layout: TileLayout,
        *,
        keep: int,
        allowed: torch.Tensor | None = None,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        partition = TokenShard(layout.num_tokens, self.exchange.group)
        if any(
            rows.ndim != 3
            or rows.shape != q.shape
            or rows.shape[0] != partition.count
            or min(rows.shape[1:]) < 1
            or rows.dtype != torch.bfloat16
            or rows.device != layout.valid_counts.device
            for rows in (q, k, v, gate)
        ):
            raise ValueError(
                "tile Q/K/V/gate require matching BF16 local token projections"
            )
        if self.num_heads is not None and q.shape[1:] != (
            self.local_heads * self.exchange.group.size,
            self.head_dim,
        ):
            raise ValueError(
                "tile projections disagree with the bound head partition"
            )
        if out is not None and (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
        ):
            raise ValueError(
                "tile output must match its local query representation"
            )
        if rotary is not None and (
            not isinstance(rotary, tuple)
            or len(rotary) != 2
            or any(
                value.shape != (layout.num_tokens, q.shape[2])
                or value.dtype != torch.float32
                or value.device != q.device
                for value in rotary
            )
        ):
            raise ValueError(
                "tile rotary tables must be full-grid FP32 interleaved lanes"
            )

        storage = None
        if self.exchange.group.size > 1:
            if self.num_heads is None:
                raise ValueError(
                    "Ulysses tile attention requires static head dimensions"
                )
            binding = _binding.vsa.get().get(id(self))
            if binding is not None:
                storage = binding.prepare_exchange(self, layout, q)
            elif q.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "bind and prepare tile exchange before capture"
                )
            # A single feature-packed payload transports all four projections
            # in one collective. Gate values have no positional transform.
            shape = (partition.capacity, q.shape[1], 4 * q.shape[2])
            payload = (
                q.new_empty(shape)
                if storage is None
                else storage.view("projection_input", shape, q)
            )
            payload[partition.count :].zero_()
            for target, source in zip(
                payload.chunk(4, dim=-1), (q, k, v, gate), strict=True
            ):
                target[: partition.count].copy_(source)
            received = self.exchange.heads((payload,), storage=storage)[0]
            q, k, v, gate = received[: layout.num_tokens].chunk(4, dim=-1)

        result = self._compute(
            q, k, v, gate, layout, keep, allowed, rotary=rotary
        )
        if self.exchange.group.size > 1:
            physical = partition.capacity * self.exchange.group.size
            if physical != layout.num_tokens:
                padded = result.new_zeros((physical, *result.shape[1:]))
                padded[: layout.num_tokens].copy_(result)
                result = padded
            result = self.exchange.tokens(result, storage=storage)[
                : partition.count
            ]
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _prepare(self, q, k, v, gate, layout, rotary):
        """Borrow preparation from the bound or transient provider."""
        binding = _binding.vsa.get().get(id(self))
        if binding is not None:
            return binding.prepare_tiles(self, q, k, v, gate, layout, rotary)
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "bind and prepare VSA tile shapes before capture"
            )
        from uniserve.runtime.backends.attention import vsa

        provider = vsa.resolve(
            "auto",
            device=q.device,
            tile=layout.tile,
            recipe=self.attention.recipe,
        )
        return provider.prepare_tiles(q, k, v, gate, layout, rotary)

    def _compute(self, q, k, v, gate, layout, keep, allowed, *, rotary=None):
        pooled = None
        if rotary is None:
            q, k, v, gate = (layout.pack(rows) for rows in (q, k, v, gate))
        else:
            q, k, v, gate, *pooled = self._prepare(
                q, k, v, gate, layout, rotary
            )
        _rows(gate, layout)
        scores, compressed = coarse_attention(
            q,
            k,
            v,
            layout,
            scale=self.attention.scale,
            allowed=allowed,
            pooled=pooled,
        )
        indices, counts = select_tiles(scores, keep, allowed=allowed)
        batch = BlockInput(
            Pattern(((keep,) * layout.num_tiles,), 0, 0, layout.tile),
            indices,
            counts,
            layout.valid_counts,
            0,
        )
        fine = self.attention(q, k, v, batch)
        return layout.unpack(add_compressed(fine, compressed, gate, layout))
