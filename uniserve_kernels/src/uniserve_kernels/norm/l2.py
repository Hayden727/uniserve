"""Channel L2 norms with ascending warp reductions and activation rounding."""

import torch

from uniserve_kernels.triton import tl, triton

if triton is not None:

    @triton.jit
    def _channel_l2(x, out, width: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, 32)
        a = tl.full((32,), 0, tl.float32)
        b = tl.full((32,), 0, tl.float32)
        c = tl.full((32,), 0, tl.float32)
        d = tl.full((32,), 0, tl.float32)
        for start in range(0, width, 128):
            # Short rows use scalar, strided loads. Longer rows use four
            # adjacent values per lane, each with its own accumulator.
            if width > 128:
                offset = start + lane * 4
                step: tl.constexpr = 1
            else:
                offset = start + lane
                step: tl.constexpr = 32
            base = x + row.to(tl.int64) * width
            v0 = tl.load(base + offset, mask=offset < width, other=0).to(
                tl.float32
            )
            v1 = tl.load(
                base + offset + step, mask=offset + step < width, other=0
            ).to(tl.float32)
            v2 = tl.load(
                base + offset + 2 * step,
                mask=offset + 2 * step < width,
                other=0,
            ).to(tl.float32)
            v3 = tl.load(
                base + offset + 3 * step,
                mask=offset + 3 * step < width,
                other=0,
            ).to(tl.float32)
            a = a + v0 * v0
            b = b + v1 * v1
            c = c + v2 * v2
            d = d + v3 * v3
        value = ((a + b) + c) + d
        for shift in tl.static_range(5):
            other = tl.gather(
                value, tl.minimum(lane + (1 << shift), 31), axis=0
            )
            value = value + other
        norm = tl.sqrt_rn(value)
        tl.store(out + row, tl.sum(tl.where(lane == 0, norm, 0), axis=0))


def channel_l2_ascending(x: torch.Tensor) -> torch.Tensor:
    """Return activation-dtype L2 norms for dense channels-last video/images.

    The channel axis is one. At least sixteen pixels and channel widths in
    32..512 (multiples of 32) preserve the reference one-warp-per-pixel tree.
    The output retains the channel axis, so division keeps BF16 norm rounding.
    """
    if triton is None or not x.is_cuda:
        raise ValueError("ascending channel L2 requires CUDA and Triton")
    layout = torch.channels_last if x.ndim == 4 else torch.channels_last_3d
    if (
        x.ndim not in (4, 5)
        or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not x.is_contiguous(memory_format=layout)
        or not 32 <= x.shape[1] <= 512
        or x.shape[1] % 32
        or x.numel() // x.shape[1] < 16
    ):
        raise ValueError(
            "ascending channel L2 requires dense channels-last floating "
            "pixels with 32..512 channels in multiples of 32 and at least "
            "16 pixels"
        )
    out = torch.empty(
        (x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype
    )
    _channel_l2[(out.numel(),)](
        x, out, x.shape[1], num_warps=4, enable_fp_fusion=False
    )
    return out
