"""Declaring a single-row recipe preserves ordinary projection behavior."""

import pytest
import torch
import torch.nn.functional as F

from uniserve.nn import Linear
from uniserve.runtime import ExecutionContext

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "rows,dtype", [(2, torch.float32), (7, torch.float32), (1, torch.float64)]
)
@torch.inference_mode()
def test_other_matrix_shapes_and_dtypes_keep_the_selected_dense_arithmetic(
    rows, dtype
):
    layer = Linear(13, 19, dtype=dtype, gemv_reduction="strided16")
    x = torch.arange(rows * 13, dtype=dtype).reshape(rows, 13) / 7
    expected = F.linear(x, layer.weight, layer.bias)
    output = torch.empty(rows, 38, dtype=dtype)[:, ::2]
    with ExecutionContext(layer, matmul="torch") as context:
        context.prepare(None)
        assert layer(x, out=output) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_unknown_reduction_is_rejected_at_construction():
    with pytest.raises(ValueError, match="GEMV reduction"):
        Linear(13, 19, gemv_reduction="unrecognized")
