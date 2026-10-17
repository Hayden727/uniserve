"""Ordered RMS statistics and activation rounding for short text sequences."""

import pytest
import torch
from uniserve_kernels.norm.mean import mean_square_ascending

from uniserve.nn.norm import RMSNorm

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA RMS needs a device"
    ),
]


def _native_variance(values, lanes):
    """FP32 oracle for native vector loads and ascending warp additions.

    Each lane sums four independent streams, combines those in order, and
    merges warps before adjacent ascending lane groups. The expected lane
    counts below come from Torch2.9 Reduce.cuh's contiguous-row launch.
    """
    rows, width = values.shape
    streams = torch.zeros((rows, lanes, 4), dtype=torch.float32)
    square = values.float().square()
    for start in range(0, width, lanes * 4):
        chunk = square[:, start : start + lanes * 4]
        padded = torch.zeros((rows, lanes * 4), dtype=torch.float32)
        padded[:, : chunk.shape[1]] = chunk
        streams = streams + padded.reshape(rows, lanes, 4)
    partial = ((streams[..., 0] + streams[..., 1]) + streams[..., 2]) + streams[
        ..., 3
    ]
    while partial.shape[1] > 32:
        half = partial.shape[1] // 2
        partial = partial[:, :half] + partial[:, half:]
    for offset in (1, 2, 4, 8, 16):
        partial = partial[:, :-offset] + partial[:, offset:]
    return partial * (1.0 / width)


@pytest.mark.parametrize(
    ("rows", "width", "lanes"),
    (
        (1, 256, 64),
        (9, 256, 64),
        (9, 384, 64),
        (2, 512, 128),
        (3, 1024, 256),
        (225, 256, 32),
        (1, 4096, 512),
        (9, 5120, 64),
    ),
)
def test_ordered_rms_preserves_native_short_row_tree(rows, width, lanes):
    generator = torch.Generator(device="cpu").manual_seed(852)
    values = torch.randn((rows, width), generator=generator).to(torch.bfloat16)
    weights = (torch.randn((width,), generator=generator) * 0.2 + 1).to(
        torch.bfloat16
    )
    variance = _native_variance(values, lanes)
    module = RMSNorm(
        width, round_before_scale=True, reduction_order="ascending"
    ).to("cuda", torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(weights)
        inputs = values.cuda()
        actual_variance = mean_square_ascending(inputs.float())
        torch.testing.assert_close(
            actual_variance.cpu(), variance, rtol=0, atol=0
        )
        # Keep rsqrt on the same GPU; this test isolates the reduction tree
        # and the two separately rounded activation/scale operations.
        normalized = (
            inputs.float() * torch.rsqrt(variance.cuda() + module.eps)
        ).to(inputs.dtype)
        expected = normalized * weights.cuda()
        output = torch.empty_like(expected)
        assert module(inputs, out=output) is output
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_ordered_rms_rejects_unimplemented_block_y_tree():
    with pytest.raises(ValueError, match="block-Y"):
        mean_square_ascending(torch.ones((2, 8064), device="cuda"))
