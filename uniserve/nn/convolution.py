"""Convolution recipes and borrowed runtime operators."""

from typing import Literal

import torch
from torch import nn

from . import _binding

Conv3dBackend = Literal["auto", "torch_unfold"]


def conv3d_unfold(input, weight, bias, *, stride, padding, dilation):
    """Torch's ungrouped volume-to-columns/GEMM convolution.

    ATen owns temporary column storage. Bias initializes the GEMM output
    before accumulation; adding it after a rounded convolution is different
    in low precision. Calling the operator directly avoids global cuDNN
    flags and preserves the same arithmetic inside CUDA graph capture.
    """
    return torch.ops.aten.slow_conv_dilated3d.default(
        input, weight, weight.shape[2:], bias, stride, padding, dilation
    )


class Conv2d(nn.Conv2d):
    def _conv_forward(self, input, weight, bias):
        operator = _binding.convolution.get().get(id(self))
        if operator is None:
            return super()._conv_forward(input, weight, bias)
        return operator(input, weight, bias)


class Conv3d(nn.Conv3d):
    """Conv3d with an optional explicit Torch unfold/GEMM recipe.

    ``auto`` retains Torch dispatch or the caller's bound convolution plan.
    ``torch_unfold`` selects ATen's slow_conv_dilated3d on CPU or CUDA and
    takes precedence over runtime cuDNN plans. It requires ungrouped numeric
    zero padding and accepts batched or unbatched inputs of arbitrary shape.
    Ordinary parameter names and checkpoint loading are unchanged.
    """

    def __init__(self, *args, backend: Conv3dBackend = "auto", **kwargs):
        if backend not in ("auto", "torch_unfold"):
            raise ValueError(f"unknown Conv3d backend: {backend}")
        super().__init__(*args, **kwargs)
        if backend == "torch_unfold" and (
            self.groups != 1
            or self.padding_mode != "zeros"
            or isinstance(self.padding, str)
        ):
            raise ValueError(
                "torch_unfold requires ungrouped numeric zero padding"
            )
        self.backend = backend

    def _conv_forward(self, input, weight, bias):
        operator = _binding.convolution.get().get(id(self))
        if operator is not None:
            return operator(input, weight, bias)
        if self.backend == "torch_unfold":
            return conv3d_unfold(
                input,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        return super()._conv_forward(input, weight, bias)
