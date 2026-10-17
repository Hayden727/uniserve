"""Video tile packing, BF16 compression and selected key domains."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from uniserve.nn.attention.vsa import BlockAttention
from uniserve.nn.attention.vsa.tiles import (
    TileAttention,
    TileLayout,
    add_compressed,
    coarse_attention,
    pool,
    select_tiles,
)

pytestmark = pytest.mark.unit


def test_boundary_tiles_pack_valid_prefixes_and_restore_raster_order():
    layout = TileLayout.from_grid((2, 3, 3), (2, 2, 2), device="cpu")
    # Each coordinate carries an odd raster label; zero belongs exclusively
    # to padding. The single selected column is a strided view.
    rows = torch.arange(1, 37).reshape(18, 2)[:, :1]
    expected = torch.tensor(
        [
            1,
            3,
            7,
            9,
            19,
            21,
            25,
            27,
            5,
            11,
            23,
            29,
            0,
            0,
            0,
            0,
            13,
            15,
            31,
            33,
            0,
            0,
            0,
            0,
            17,
            35,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )[:, None]

    packed = layout.pack(rows)

    assert torch.equal(packed, expected)
    assert torch.equal(layout.unpack(packed), rows)
    assert layout.valid_counts.tolist() == [8, 4, 4, 2]
    assert layout.valid_counts.dtype == torch.int32
    with pytest.raises(FrozenInstanceError):
        layout.grid = (1, 1, 18)


@pytest.mark.parametrize(
    "tile_shape,tiles,smallest,keep,full_keep",
    [((4, 8, 8), 336, 96, 30, 293), ((4, 4, 8), 672, 32, 59, 585)],
)
def test_full_video_selection_budget_uses_real_tokens(
    tile_shape, tiles, smallest, keep, full_keep
):
    layout = TileLayout.from_grid((48, 30, 52), tile_shape, device="cpu")

    assert layout.num_tokens == 74880
    assert layout.num_tiles == tiles
    assert layout.padded_tokens == 86016
    assert layout.valid_counts.sum().item() == 74880
    assert layout.valid_counts.min().item() == smallest
    assert layout.keep(0.9) == keep
    assert layout.keep(0.0) == full_keep


def test_a_single_boundary_token_and_masked_budget():
    layout = TileLayout.from_grid((1, 1, 1), (4, 4, 8), device="cpu")
    value = torch.tensor([[7.0]])
    packed = layout.pack(value)
    assert packed.shape == (128, 1)
    assert torch.equal(packed[:1], value)
    assert torch.count_nonzero(packed[1:]) == 0
    assert layout.keep(0.9) == 1

    larger = TileLayout.from_grid((48, 30, 52), (4, 4, 8), device="cpu")
    # Structural masks use the supplied mean permitted key count, not the
    # full grid. This host value may be fractional for unequal rows.
    assert larger.keep(0.9, allowed_count=15.5) == 2
    assert larger.keep(0.9, allowed_count=0) == 1


def test_compression_rounds_means_scores_probabilities_and_values_to_bf16():
    layout = TileLayout.from_grid((1, 1, 6), (1, 1, 2), device="cpu")
    q, k, v = (
        torch.tensor(values, dtype=torch.bfloat16).reshape(6, 1, 1)
        for values in (
            [1, 1.0078125, 2, 2.015625, -1, -1.0078125],
            [1, 1.0078125, -2, -2.015625, 3, 3.015625],
            [0.1, 0.2, 1, 1.0078125, 100, 101],
        )
    )
    # Means halfway between BF16 numbers round to even before scoring.
    assert pool(q, layout).flatten().tolist() == [1, 2, -1]
    assert pool(v, layout).flatten().tolist() == [0.150390625, 1, 100.5]

    scores, compressed = coarse_attention(q, k, v, layout)

    assert scores.dtype == compressed.dtype == torch.bfloat16
    assert scores.tolist() == [[[1, -2, 3], [2, -4, 6], [-1, 2, -3]]]
    # Independent scalar softmax, each probability rounded to BF16 before
    # the weighted sum. FP32 softmax/value multiplication differs here.
    assert compressed.flatten().tolist() == [88, 98.5, 1.59375]


def test_partial_tile_pooling_excludes_padding_from_sum_and_divisor():
    layout = TileLayout.from_grid((1, 1, 3), (1, 1, 2), device="cpu")
    packed = torch.tensor([2, 4, 10, 1000], dtype=torch.bfloat16).view(4, 1, 1)
    assert pool(packed, layout).flatten().tolist() == [3, 10]


def test_scores_round_matmul_before_head_scaling():
    layout = TileLayout.from_grid((1, 1, 1), (1, 1, 1), device="cpu")
    rows = torch.tensor([[[1, 1.0625]]], dtype=torch.bfloat16)

    scores, compressed = coarse_attention(rows, rows, rows, layout)

    # The dot product 2.12890625 first rounds to BF16 2.125, then scales by
    # 1/sqrt(2). Scaling the FP32 dot directly would yield BF16 1.5078125.
    assert scores.item() == 1.5
    assert torch.equal(compressed, rows)


def test_supplied_rounded_means_preserve_compressed_attention():
    layout = TileLayout.from_grid((1, 1, 3), (1, 1, 2), device="cpu")
    rows = torch.tensor([2, 4, 10, 999], dtype=torch.bfloat16).view(4, 1, 1)
    means = torch.tensor([3, 10], dtype=torch.bfloat16).view(2, 1, 1)
    expected = coarse_attention(rows, rows, rows, layout)
    actual = coarse_attention(
        rows, rows, rows, layout, pooled=(means, means, means)
    )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(ValueError, match="tile means"):
        coarse_attention(
            rows, rows, rows, layout, pooled=(means.float(), means, means)
        )


def test_rotary_tables_cover_the_global_grid_with_fp32_lanes():
    layout = TileLayout.from_grid((1, 1, 3), (1, 1, 128), device="cpu")
    rows = torch.zeros(3, 1, 128, dtype=torch.bfloat16)
    layer = TileAttention(BlockAttention(128**-0.5))
    for rotary in (
        (torch.ones(2, 128), torch.zeros(2, 128)),
        (torch.ones(3, 128, dtype=torch.bfloat16), torch.zeros(3, 128)),
        (torch.ones(3, 128),),
    ):
        with pytest.raises(ValueError, match="rotary tables"):
            layer(rows, rows, rows, rows, layout, keep=1, rotary=rotary)


def test_structural_mask_limits_compression_and_selection():
    layout = TileLayout.from_grid((1, 1, 6), (1, 1, 2), device="cpu")
    q = torch.ones(6, 2, 1, dtype=torch.bfloat16)
    v = torch.tensor([2, 2, 10, 10, 100, 100], dtype=torch.bfloat16)
    v = v[:, None, None].expand_as(q)
    allowed = torch.tensor(
        [[True, False, False], [False, True, False], [False] * 3]
    )

    scores, compressed = coarse_attention(q, q, v, layout, allowed=allowed)
    indices, counts = select_tiles(scores, 2, allowed=allowed)

    assert compressed[:, 0, 0].tolist() == [2, 10, 0]
    assert torch.equal(compressed[:, 0], compressed[:, 1])
    assert counts.tolist() == [[1, 1, 0], [1, 1, 0]]
    assert indices[:, :2, 0].tolist() == [[0, 1], [0, 1]]
    assert torch.isneginf(scores[:, 2]).all()


def test_selection_keeps_earlier_ties_and_publishes_indices_in_key_order():
    scores = torch.tensor(
        [[[2, 2, 2, 1], [1, 3, 2, 4]], [[4, 1, 2, 3], [0, -0.0, 0, -0.0]]],
        dtype=torch.bfloat16,
    )
    indices, counts = select_tiles(scores, 2)

    assert indices.dtype == counts.dtype == torch.int32
    assert indices.tolist() == [[[0, 1], [1, 3]], [[0, 3], [0, 1]]]
    assert counts.tolist() == [[2, 2], [2, 2]]


def test_compression_gate_is_raw_and_product_rounds_before_addition():
    layout = TileLayout.from_grid((1, 1, 2), (1, 1, 2), device="cpu")
    fine = torch.full((2, 1, 1), -9, dtype=torch.bfloat16)
    compressed = torch.full((1, 1, 1), 1.0078125, dtype=torch.bfloat16)
    gate = torch.tensor([1.0078125, -2], dtype=torch.bfloat16).view(2, 1, 1)

    output = add_compressed(fine, compressed, gate, layout)

    # A fused multiply/add would produce -7.96875 in the first row.
    assert output.flatten().tolist() == [-8, -11]
    assert fine.flatten().tolist() == [-9, -9]


def test_full_attention_refuses_a_tile_without_a_provider():
    layout = TileLayout.from_grid((1, 1, 1), (4, 8, 8), device="cpu")
    rows = torch.ones(1, 1, 128, dtype=torch.bfloat16)
    module = TileAttention(BlockAttention(128**-0.5))
    with pytest.raises(RuntimeError, match="serves 256-row tiles"):
        module(rows, rows, rows, rows, layout, keep=1)


@pytest.mark.parametrize(
    "grid,tile",
    [((0, 2, 2), (1, 1, 1)), ((1, 2), (1, 1, 1)), ((1, 2, 2), (1, 0, 1))],
)
def test_tile_dimensions_must_be_positive_triples(grid, tile):
    with pytest.raises(ValueError, match="positive integer triples"):
        TileLayout.from_grid(grid, tile, device="cpu")


def test_incompatible_projection_and_mask_shapes_are_rejected():
    layout = TileLayout.from_grid((1, 1, 3), (1, 1, 2), device="cpu")
    rows = torch.ones(4, 2, 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16"):
        coarse_attention(rows.float(), rows, rows, layout)
    with pytest.raises(ValueError, match="mask"):
        coarse_attention(rows, rows, rows, layout, allowed=torch.ones(2, 2))
    with pytest.raises(ValueError, match="real token"):
        layout.pack(rows)
    with pytest.raises(ValueError, match="sparsity"):
        layout.keep(1.0)


def test_static_heads_reject_a_mismatched_projection_before_provider_dispatch():
    layout = TileLayout.from_grid((1, 1, 1), (1, 4, 32), device="cpu")
    module = TileAttention(BlockAttention(128**-0.5), num_heads=8, head_dim=128)
    rows = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="bound head partition"):
        module(rows, rows, rows, rows, layout, keep=1)


@pytest.mark.parametrize(
    "heads,width", ((8, None), (None, 128), (0, 128), (8, True))
)
def test_static_head_dimensions_are_declared_together(heads, width):
    with pytest.raises(ValueError, match="positive integer pair"):
        TileAttention(
            BlockAttention(128**-0.5), num_heads=heads, head_dim=width
        )
