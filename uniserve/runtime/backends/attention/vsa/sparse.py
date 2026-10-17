"""Compact logical blocks into combined partial and full FA4 key lists."""

import torch

from uniserve.nn.attention.vsa.inputs import Recipe
from uniserve.tensors import BufferConfig


def _dimensions(tile, recipe):
    if not isinstance(recipe, Recipe):
        raise TypeError("sparse maps require a typed numerical recipe")
    recipe.validate_tile(tile)
    return 64 if recipe is Recipe.FA4_128X64 else 128


def buffer_configs(
    heads, queries, selected, keys, *, tile=256, recipe=Recipe.FA4_128X64
):
    """Describe caller-owned maps; no allocation depends on device counts."""
    key_tile = _dimensions(tile, recipe)
    children = tile // key_tile
    fields = {
        "valid": BufferConfig((keys * children,), torch.int32),
    }
    for kind in ("mask", "full"):
        fields[f"{kind}_indices"] = BufferConfig(
            (1, heads, queries, selected * children), torch.int32
        )
        fields[f"{kind}_counts"] = BufferConfig(
            (1, heads, queries), torch.int32
        )
    return fields


def split_blocks(
    indices, counts, valid, *, out, tile=256, recipe=Recipe.FA4_128X64
):
    """Write combined native partial/full lists for device-side traversal.

    Logical query blocks keep their original extent. Each selected key splits
    into physical children, excluding empty children and splitting partial from
    complete children. Each list is ascending; FA4 visits partial keys first,
    then complete keys, both backwards. FA4_128X64 partitions this traversal
    inside its CTA, retaining raw accumulators until their final merge.
    FA4_128X128 keeps one accumulator per query. Counts and indices stay live.
    Only each count's prefix is read, including when unused indices are invalid.
    """
    key_tile = _dimensions(tile, recipe)
    if (
        indices.ndim != 3
        or min(indices.shape) < 1
        or counts.shape != indices.shape[:2]
        or valid.ndim != 1
        or not valid.numel()
        or any(
            t.dtype != torch.int32 or t.device != indices.device
            for t in (indices, counts, valid)
        )
    ):
        raise ValueError("sparse maps require matching int32 metadata")
    heads, queries, selected = indices.shape
    fields = buffer_configs(
        heads, queries, selected, valid.numel(), tile=tile, recipe=recipe
    )
    if any(
        out[name].shape != field.shape
        or out[name].dtype != field.dtype
        or out[name].device != indices.device
        or not out[name].is_contiguous()
        for name, field in fields.items()
    ):
        raise ValueError("sparse map destinations must match their capacities")

    factor = tile // key_tile
    child = torch.arange(factor, device=indices.device, dtype=torch.int32)
    out["valid"].copy_(
        (valid[:, None] - child * key_tile).clamp(0, key_tile).flatten()
    )
    live = torch.arange(selected, device=indices.device) < counts[..., None]
    safe = indices.masked_fill(~live, 0)
    expanded = (safe[..., None] * factor + child).flatten(-2)
    sizes = out["valid"][expanded.long()]
    live = live.repeat_interleave(factor, dim=-1)
    sentinel = valid.numel() * factor
    for kind, present in (
        ("mask", live & (sizes > 0) & (sizes < key_tile)),
        ("full", live & (sizes == key_tile)),
    ):
        children = expanded.masked_fill(~present, sentinel).sort(-1).values
        out[f"{kind}_indices"].copy_(
            children.masked_fill(children == sentinel, 0)
        )
        out[f"{kind}_counts"].copy_(present.sum(-1, dtype=torch.int32))
