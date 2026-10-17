"""CUDA scale arithmetic for the compiled NVFP4 training-forward recipe."""

import torch

from uniserve_kernels.triton import tl, triton

if triton is not None:

    @triton.jit
    def _global_scale(maximum, out):
        value = tl.load(maximum)
        # The compiled recipe lowers aten.reciprocal to ordinary Triton
        # division, then multiplies in FP32. Eager ATen reciprocal and a
        # single correctly rounded 2688/value division can both differ.
        inverse = tl.full((), 1.0, tl.float32) / value
        scale = inverse * tl.full((), 2688.0, tl.float32)
        tl.store(out, scale)


def global_scale(maximum: torch.Tensor) -> torch.Tensor:
    """Scale a precomputed FP32 maximum using native compiled arithmetic.

    The caller computes float/abs/nan_to_num/max/clamp before this operation.
    This only fixes the GPU reciprocal primitive; it neither recomputes the
    statistic nor changes the zero-operand clamp or output representation.
    """
    if (
        triton is None
        or not maximum.is_cuda
        or maximum.dtype != torch.float32
        or maximum.shape != ()
    ):
        raise ValueError(
            "NVFP4 QAT scale arithmetic requires a CUDA FP32 scalar"
        )
    out = torch.empty_like(maximum)
    _global_scale[(1,)](maximum, out, num_warps=4)
    return out
