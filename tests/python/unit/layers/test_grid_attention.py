"""Exact grid visibility through dense segments and signed softmax states."""

import pytest
import torch

from uniserve.nn.attention.grid import (
    AttentionState,
    GridAttention,
    GridLayout,
    merge_states,
)
from uniserve.runtime.backends.attention.grid import TorchSegments

pytestmark = pytest.mark.unit


def _cells(views, columns, order):
    camera = torch.arange(views).repeat_interleave(columns)
    column = torch.arange(columns).repeat(views)
    if order == "time_major":
        permutation = (
            torch.arange(views * columns).reshape(views, columns).T.flatten()
        )
        camera, column = camera[permutation], column[permutation]
    return camera, column


def _oracle(q, k, v, camera, column, cell_tokens):
    # Evaluate the native predicate independently in the original token order.
    cameras = camera.repeat_interleave(cell_tokens)
    columns = column.repeat_interleave(cell_tokens)
    allowed = (
        (columns[:, None] == columns[None, :])
        | (cameras[:, None] == cameras[None, :])
        | (cameras[None, :] == 0)
    )
    logits = (
        torch.einsum("qhd,khd->hqk", q.float(), k.float()) * q.shape[-1] ** -0.5
    )
    probabilities = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v.float()).to(q.dtype)


@pytest.mark.parametrize("order", ["view_major", "time_major"])
@pytest.mark.parametrize("heads", [1, 5])
def test_rectangular_attention_matches_independent_dense_union(order, heads):
    camera, column = _cells(3, 4, order)
    # Neither the original order nor camera-major is required from a caller.
    permutation = torch.tensor([7, 2, 11, 4, 0, 9, 3, 1, 8, 5, 10, 6])
    camera, column = camera[permutation], column[permutation]
    layout = GridLayout.from_cells(camera, column, 3, device="cpu")
    generator = torch.Generator().manual_seed(27)
    q, k, v = [
        torch.randn(36, heads, 8, generator=generator).bfloat16()
        for _ in range(3)
    ]
    result = GridAttention(heads, 8)(q, k, v, layout, operator=TorchSegments())
    expected = _oracle(q, k, v, camera, column, 3)
    # Four partial outputs round to BF16 before the FP32 signed merge.
    # A single dense softmax has a different rounding path.
    torch.testing.assert_close(
        result, expected, rtol=0, atol=4 * torch.finfo(torch.bfloat16).eps
    )


def test_reference_row_and_overlap_subtraction_have_the_exact_key_union():
    camera, column = _cells(2, 2, "time_major")
    layout = GridLayout.from_cells(camera, column, 1, device="cpu")
    q = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    # Model order: ref/head, other/head, ref/tail, other/tail.
    v = torch.tensor([0, 6, 12, 30], dtype=torch.bfloat16)[
        :, None, None
    ].expand_as(q)
    result = GridAttention(1, 8)(q, q, v, layout, operator=TorchSegments())
    expected = torch.tensor([6, 12, 14, 12], dtype=torch.bfloat16)[
        :, None, None
    ].expand_as(q)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_reference_queries_ignore_large_row_and_cell_placeholder_lse():
    camera, column = _cells(2, 1, "view_major")
    layout = GridLayout.from_cells(camera, column, 1, device="cpu")

    def state(output, lse):
        return AttentionState(
            torch.tensor(output, dtype=torch.bfloat16).reshape(-1, 1, 1, 1),
            torch.tensor(lse, dtype=torch.float32).reshape(-1, 1, 1),
        )

    row = state([9], [10000])
    cell = state([9], [10000])
    reference = AttentionState(
        torch.tensor([2, 2], dtype=torch.bfloat16).reshape(1, 2, 1, 1),
        torch.zeros(1, 1, 2),
    )
    col = AttentionState(
        torch.tensor([4, 9], dtype=torch.bfloat16).reshape(1, 2, 1, 1),
        torch.tensor([0, 10000], dtype=torch.float32).reshape(1, 1, 2),
    )
    result = merge_states(
        layout, row=row, column=col, reference=reference, cell=cell
    )
    # Camera zero sees only the two equally weighted valid states. A huge
    # placeholder from camera one must not set its normalization maximum.
    assert result.output[0, 0, 0, 0] == 3
    assert result.lse[0, 0, 0] == pytest.approx(0.6931471805599453)
    assert torch.isfinite(result.output).all()


def test_union_lse_restores_time_major_order():
    camera, column = _cells(2, 2, "time_major")
    layout = GridLayout.from_cells(camera, column, 1, device="cpu")

    def state(batch, queries, keys):
        # Zero logits and unit values: output is one; LSE is log(key count).
        return AttentionState(
            torch.ones(batch, queries, 1, 1, dtype=torch.bfloat16),
            torch.full((batch, 1, queries), float(torch.tensor(keys).log())),
        )

    result = merge_states(
        layout,
        row=state(1, 2, 2),
        column=state(2, 2, 1),
        reference=state(1, 4, 2),
        cell=state(2, 1, 1),
    )
    torch.testing.assert_close(
        result.output, torch.ones_like(result.output), rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.lse.flatten(), torch.tensor([3.0, 4.0, 3.0, 4.0]).log()
    )


def test_full_grid_rejects_a_divisible_local_shard_before_attention():
    camera, column = _cells(32, 16, "time_major")
    layout = GridLayout.from_cells(camera, column, 1560, device="cpu")
    assert layout.num_tokens == 798720
    # This is one SP8 rank's token count and is also divisible by 512.
    q = torch.zeros(99840, 1, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="complete.*sequence"):
        GridAttention(1, 8)(q, q, q, layout, operator=TorchSegments())


def test_bound_grid_calls_preserve_outputs_across_workspace_reuse():
    from uniserve.runtime import ExecutionContext

    camera, column = _cells(2, 2, "time_major")
    layout = GridLayout.from_cells(camera, column, 1, device="cpu")
    q = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    v = torch.arange(4, dtype=torch.bfloat16)[:, None, None].expand_as(q)
    layer = GridAttention(1, 8)
    with ExecutionContext(layer, grid="torch") as context:
        context.prepare(None)
        first = layer(q, q, v, layout)
        expected = _oracle(q, q, v, camera, column, 1)
        destination = torch.empty_like(q)
        actual = layer(q, q, v + 3, layout, out=destination)
        assert actual is destination
        torch.testing.assert_close(
            first, expected, rtol=0, atol=4 * torch.finfo(torch.bfloat16).eps
        )
        torch.testing.assert_close(
            actual,
            _oracle(q, q, v + 3, camera, column, 1),
            rtol=0,
            atol=4 * torch.finfo(torch.bfloat16).eps,
        )
    # Returned values own their storage independently of the retired context.
    torch.testing.assert_close(
        first, expected, rtol=0, atol=4 * torch.finfo(torch.bfloat16).eps
    )


@pytest.mark.parametrize(
    "camera,column,cell_tokens",
    [
        ([0, 0, 1, 1], [0, 1, 0, 0], 2),
        ([0, 0, 2, 2], [0, 1, 0, 1], 2),
        ([0, -1, 1, 1], [0, 1, 0, 1], 2),
        ([0, 0], [0, 1], 2),
        ([0, 1], [0, 0], 0),
    ],
)
def test_grid_rejects_nonrectangular_or_unsupported_cells(
    camera, column, cell_tokens
):
    with pytest.raises(ValueError):
        GridLayout.from_cells(
            torch.tensor(camera),
            torch.tensor(column),
            cell_tokens,
            device="cpu",
        )
