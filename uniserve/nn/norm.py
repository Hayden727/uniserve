"""Root-mean-square normalization as an ordinary numerical module."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from . import functional


class RMSNorm(nn.Module):
    """Normalize the final feature axis with FP32 variance accumulation.

    With ``elementwise_affine`` (the default) the normalized rows multiply a
    learned ``weight`` Parameter. Without it the rows are only normalized;
    ``weight`` is then a non-persistent unit buffer, so both forms share one
    normalization primitive and multiplying by one leaves every value exact.
    ``round_before_scale`` retains a checkpoint's explicit activation-precision
    rounding between normalization and the learned scale multiplication.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        elementwise_affine: bool = True,
        round_before_scale: bool = False,
        reduction_order: Literal["default", "ascending"] = "default",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or eps <= 0:
            raise ValueError(
                "RMS normalization requires positive width and epsilon"
            )
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            # The unit scale is a derived constant, not a checkpoint value:
            # it stays a real FP32 tensor under meta construction, and loading
            # moves it with its module.
            actual = torch.device("cpu" if device is None else device)
            if actual.type == "meta":
                actual = torch.device("cpu")
            self.register_buffer(
                "weight",
                torch.ones(hidden_size, device=actual, dtype=torch.float32),
                persistent=False,
            )
        self.eps = eps
        self.round_before_scale = round_before_scale
        if reduction_order not in {"default", "ascending"}:
            raise ValueError("unknown RMS reduction order")
        if reduction_order != "default" and not round_before_scale:
            raise ValueError("ordered RMS requires rounding before scale")
        self.reduction_order = reduction_order

    def forward(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self.round_before_scale:
            return functional.rms_norm(x, self.weight, self.eps, out=out)
        # Some checkpoints normalize in FP32, round to activation precision,
        # and only then apply the learned scale. A one-round fused RMS kernel
        # does not implement that contract.
        value = x.float()
        if self.reduction_order == "ascending":
            from uniserve_kernels.norm.mean import mean_square_ascending

            variance = mean_square_ascending(value)
        else:
            variance = value.square().mean(-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.eps)
        result = normalized.to(x.dtype) * self.weight
        if out is None:
            return result
        if (
            out.shape != result.shape
            or out.dtype != result.dtype
            or out.device != result.device
        ):
            raise ValueError("RMS output must match shape, dtype and device")
        return out.copy_(result)


class LayerNorm(nn.LayerNorm):
    """Layer normalization with FP32 statistics and affine arithmetic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(x.dtype)
