"""Explicit convolution recipes preserve public convolution semantics."""

import itertools

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn.convolution import Conv3d
from uniserve.runtime import ExecutionContext

pytestmark = pytest.mark.unit


def _convolution(input, weight, bias, stride, padding, dilation):
    """Small independent scalar oracle; integer operands are exactly summed."""
    shape = tuple(
        (n + 2 * p - d * (k - 1) - 1) // s + 1
        for n, k, s, p, d in zip(
            input.shape[2:],
            weight.shape[2:],
            stride,
            padding,
            dilation,
            strict=True,
        )
    )
    output = torch.empty(
        input.shape[0], weight.shape[0], *shape, dtype=input.dtype
    )
    for batch, channel, *position in itertools.product(
        range(input.shape[0]),
        range(weight.shape[0]),
        *(range(n) for n in shape),
    ):
        total = float(bias[channel]) if bias is not None else 0.0
        for feature, *kernel in itertools.product(
            range(weight.shape[1]), *(range(n) for n in weight.shape[2:])
        ):
            source = tuple(
                a * s - p + k * d
                for a, s, p, k, d in zip(
                    position, stride, padding, kernel, dilation, strict=True
                )
            )
            if all(
                0 <= a < n for a, n in zip(source, input.shape[2:], strict=True)
            ):
                total += float(input[(batch, feature, *source)]) * float(
                    weight[(channel, feature, *kernel)]
                )
        output[(batch, channel, *position)] = total
    return output


@pytest.mark.parametrize("unbatched,bias", [(False, True), (True, False)])
@torch.inference_mode()
def test_unfold_convolution_with_stride_padding_dilation_and_strided_input(
    unbatched, bias
):
    layer = Conv3d(
        2,
        3,
        (2, 2, 3),
        stride=(2, 1, 2),
        padding=(1, 0, 2),
        dilation=(1, 2, 1),
        bias=bias,
        dtype=torch.float64,
        backend="torch_unfold",
    )
    values = torch.arange(2 * 2 * 4 * 5 * 12, dtype=torch.float64).reshape(
        2, 2, 4, 5, 12
    )
    values = (values.remainder(7) - 3)[..., ::2]
    layer.weight.copy_(
        torch.arange(layer.weight.numel()).reshape_as(layer.weight).remainder(5)
        - 2
    )
    if bias:
        layer.bias.copy_(torch.tensor([2, -1, 3]))
    expected = _convolution(
        values,
        layer.weight,
        layer.bias,
        layer.stride,
        layer.padding,
        layer.dilation,
    )
    if unbatched:
        values, expected = values[0], expected[0]
    torch.testing.assert_close(layer(values), expected, rtol=0, atol=0)
    # An explicit recipe remains valid when a caller supplies an unrelated
    # strict cuDNN table, and ordinary checkpoint parameter names are kept.
    with ExecutionContext(layer, convolution=()) as context:
        context.prepare(None)
        torch.testing.assert_close(layer(values), expected, rtol=0, atol=0)
    ordinary = torch.nn.Conv3d(2, 3, (2, 2, 3), bias=bias, dtype=torch.float64)
    ordinary.load_state_dict(layer.state_dict(), strict=True)
    assert torch.equal(ordinary.weight, layer.weight)


@pytest.mark.parametrize(
    "kwargs",
    [{"groups": 2}, {"padding": "same"}, {"padding_mode": "reflect"}],
)
def test_unfold_convolution_rejects_unsupported_contracts(kwargs):
    with pytest.raises(ValueError, match="ungrouped numeric zero padding"):
        Conv3d(2, 4, 3, backend="torch_unfold", **kwargs)


@torch.inference_mode()
def test_automatic_convolution_retains_torch_semantics():
    layer = Conv3d(2, 4, 3, groups=2, padding="same", dtype=torch.float64)
    values = torch.arange(2 * 4 * 5 * 6, dtype=torch.float64).reshape(
        1, 2, 4, 5, 6
    )
    torch.testing.assert_close(
        layer(values),
        F.conv3d(values, layer.weight, layer.bias, padding="same", groups=2),
        rtol=0,
        atol=0,
    )
