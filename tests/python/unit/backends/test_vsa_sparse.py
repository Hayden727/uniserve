"""Combined sparse domains preserve token membership and validity bounds."""

import pytest
import torch

from uniserve.nn.attention.vsa import BlockAttention, Recipe
from uniserve.runtime.backends.attention.vsa import resolve
from uniserve.runtime.backends.attention.vsa.sparse import (
    buffer_configs,
    split_blocks,
)

pytestmark = pytest.mark.unit


def _buffers(heads, queries, selected, keys, *, tile=256):
    return {
        name: torch.empty(field.shape, dtype=field.dtype)
        for name, field in buffer_configs(
            heads, queries, selected, keys, tile=tile
        ).items()
    }


@pytest.mark.parametrize(
    ("tile", "valid_sizes", "traversals", "child_sizes"),
    [
        (
            256,
            [256, 129, 128, 1, 0],
            ([12, 6, 5, 4, 3, 2, 1, 0], [9, 8], [6, 9, 8, 5, 4], []),
            [
                64,
                64,
                64,
                64,
                64,
                64,
                1,
                0,
                64,
                64,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ),
        (
            128,
            [128, 65, 64, 1, 0],
            ([6, 3, 2, 1, 0], [4], [3, 4, 2], []),
            [64, 64, 64, 1, 64, 0, 1, 0, 0, 0],
        ),
    ],
)
def test_sparse_maps_preserve_partial_full_order_and_ignore_unused_indices(
    tile, valid_sizes, traversals, child_sizes
):
    valid = torch.tensor(valid_sizes, dtype=torch.int32)
    indices = torch.tensor(
        [[[0, 1, 3], [2, -999, 999], [1, 2, 999], [999, -999, 999]]],
        dtype=torch.int32,
    )
    counts = torch.tensor([[3, 1, 2, 0]], dtype=torch.int32)
    output = _buffers(1, 4, 3, 5, tile=tile)
    split_blocks(indices, counts, valid, out=output, tile=tile)
    # Native order is partial children descending, then full descending.
    for query, reference in enumerate(traversals):
        actual = []
        for kind in ("mask", "full"):
            count = output[f"{kind}_counts"][0, 0, query].item()
            actual.extend(
                output[f"{kind}_indices"][0, 0, query, :count].flip(0).tolist()
            )
        assert actual == reference
    assert output["valid"].tolist() == child_sizes


def test_hoisted_recipe_keeps_all_valid_children_in_one_query_accumulator():
    recipe = Recipe.FA4_128X128
    indices = torch.tensor([[[0, 1], [1, -99], [-99, -99]]], dtype=torch.int32)
    counts = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    valid = torch.tensor([240, 120, 0], dtype=torch.int32)
    output = {
        name: torch.empty(field.shape, dtype=field.dtype)
        for name, field in buffer_configs(1, 3, 2, 3, recipe=recipe).items()
    }
    split_blocks(indices, counts, valid, out=output, recipe=recipe)
    assert output["valid"].tolist() == [128, 112, 120, 0, 0, 0]
    for query, expected in enumerate(([2, 1, 0], [2], [])):
        actual = []
        for kind in ("mask", "full"):
            count = output[f"{kind}_counts"][0, 0, query].item()
            actual.extend(
                output[f"{kind}_indices"][0, 0, query, :count].flip(0).tolist()
            )
        assert actual == expected


def test_recipe_rejects_untyped_or_incompatible_logical_tiles():
    with pytest.raises(TypeError, match="typed numerical recipe"):
        BlockAttention(128**-0.5, recipe="fa4_128x128")
    with pytest.raises(ValueError, match="requires logical tiles"):
        resolve("auto", device="cpu", tile=128, recipe=Recipe.FA4_128X128)
