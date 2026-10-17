"""FP32 mean-square with a specified lane reduction order.

Four strided accumulators per lane combine from left to right; adjacent
lane groups combine at distances 1, 2, 4, 8 and 16. This preserves the
contiguous-row reduction used by PyTorch 2.9. Short sequences use additional
warps per row, reduced in descending order before the final warp. Later
PyTorch versions reverse the lane
order, which can change activation-precision rounding after RMSNorm.
"""

import torch

from uniserve_kernels.triton import tl, triton

if triton is not None:

    @triton.jit
    def _mean_square(
        x, out, width: tl.constexpr, lanes: tl.constexpr, stages: tl.constexpr
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, lanes)
        a = tl.full((lanes,), 0, tl.float32)
        b = tl.full((lanes,), 0, tl.float32)
        c = tl.full((lanes,), 0, tl.float32)
        d = tl.full((lanes,), 0, tl.float32)
        for start in range(0, width, 4 * lanes):
            offset = row.to(tl.int64) * width + start + lane * 4
            valid = start + lane * 4 < width
            v0 = tl.load(x + offset, mask=valid, other=0).to(tl.float32)
            v1 = tl.load(x + offset + 1, mask=valid, other=0).to(tl.float32)
            v2 = tl.load(x + offset + 2, mask=valid, other=0).to(tl.float32)
            v3 = tl.load(x + offset + 3, mask=valid, other=0).to(tl.float32)
            a = a + v0 * v0
            b = b + v1 * v1
            c = c + v2 * v2
            d = d + v3 * v3
        value = ((a + b) + c) + d
        for shift in tl.static_range(stages, 4, -1):
            other = tl.gather(
                value, tl.minimum(lane + (1 << shift), lanes - 1), axis=0
            )
            value = value + other
        for offset in tl.static_range(5):
            other = tl.gather(
                value, tl.minimum(lane + (1 << offset), lanes - 1), axis=0
            )
            value = value + other
        mean = value * (1.0 / width)
        tl.store(out + row, tl.sum(tl.where(lane == 0, mean, 0), axis=0))


def mean_square_ascending(x: torch.Tensor) -> torch.Tensor:
    """Reduce contiguous CUDA rows without fusing products into additions.

    The width is a multiple of 128 in 256..8064. Short sequences can use
    multiple warps per row; their lane count is also capped by the number
    of four-element input vectors. Shapes that split a row along the
    reference block's Y dimension require a different summation tree.
    The output retains the reduced axis and uses FP32.
    """
    if triton is None or not x.is_cuda:
        raise ValueError("ascending mean-square requires CUDA and Triton")
    width = x.shape[-1] if x.ndim else 0
    if (
        x.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        or not 256 <= width <= 8064
        or width % 128
        or x.numel() // width < 1
    ):
        raise ValueError(
            "ascending mean-square requires contiguous floating rows with "
            "width a multiple of 128 in 256..8064"
        )
    rows = x.numel() // width
    block_y = min(1 << (rows.bit_length() - 1), 16)
    # PyTorch 2.9 Reduce.cuh::set_block_dimension first caps block Y at
    # 512 / 32, then expands block X only as far as the input vector count.
    # For [9, 256] this is 64 lanes, while [1, 256] also needs only 64.
    vectors = width // 4
    lanes = min(512 // block_y, 1 << (vectors.bit_length() - 1))
    values_per_thread = (width + lanes - 1) // lanes
    if block_y > 1 and values_per_thread >= min(block_y * 16, 256):
        raise ValueError(
            "ascending mean-square does not support reference block-Y "
            f"reduction for {rows} rows of width {width}"
        )
    out = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    _mean_square[(out.numel(),)](
        x,
        out,
        width,
        lanes,
        lanes.bit_length() - 2,
        num_warps=max(4, lanes // 32),
        enable_fp_fusion=False,
    )
    return out
