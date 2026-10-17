"""Immutable-weight NVFP4 projection with native training-forward arithmetic."""

import torch
from torch import nn

from uniserve.quantization import QuantizedTensor
from uniserve.quantization.nvfp4_qat import encode_weight

from . import _binding
from .functional._linear import apply_linear
from .linear import Linear


class NVFP4QATLinear(Linear):
    """Dynamic activation scaling, FP32 GEMM alpha, then BF16 bias addition.

    The layer owns immutable encoded weights and their FP32 global scale.
    Runtime binding owns activation encoding, scratch and prepared operators.
    This is the QAT forward computation; inference retains no backward state.
    """

    def __init__(self, weight: QuantizedTensor, global_scale, *, bias):
        super().__init__(
            weight.shape[1],
            weight.shape[0],
            bias=bias is not None,
            device="meta",
            dtype=torch.bfloat16,
        )
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = (
            None
            if bias is None
            else nn.Parameter(bias.detach(), requires_grad=False)
        )
        self.register_buffer("global_scale", global_scale)

    @classmethod
    def from_linear(cls, source: Linear | nn.Linear) -> "NVFP4QATLinear":
        """Encode one complete local projection after loading and placement."""
        if type(source) not in (Linear, nn.Linear) or isinstance(
            source.weight, QuantizedTensor
        ):
            raise ValueError(
                "NVFP4 QAT conversion requires an unencoded local Linear"
            )
        if source.bias is not None and (
            source.bias.dtype != torch.bfloat16
            or source.bias.device != source.weight.device
        ):
            raise ValueError("NVFP4 QAT bias must be BF16 beside its weight")
        weight, scale = encode_weight(source.weight)
        return cls(weight, scale, bias=source.bias)

    def forward(
        self, x, *, round_before_bias=False, output_dtype=None, out=None
    ):
        """Project BF16 activations with a separate BF16 bias store."""
        if output_dtype not in (None, torch.bfloat16):
            raise ValueError("NVFP4 QAT projection produces BF16 output")
        operator = _binding.matmul.get().get(id(self))
        if operator is None:
            from uniserve.runtime.backends.matmul.nvfp4_qat import Backend

            provider = Backend(global_scale=self.global_scale)
            options = {
                "input_dtype": x.dtype,
                "input_quantizer": None,
                "max_rows": x.numel() // x.shape[-1],
                "output_dtype": torch.bfloat16,
            }
            requirements = provider.workspace_buffers(self.weight, **options)
            workspace = {
                name: torch.empty(
                    config.shape, dtype=config.dtype, device=x.device
                )
                for name, config in requirements.items()
            }
            operator = provider.prepare(
                self.weight, **options, workspace=workspace
            )
        # The recipe always rounds before bias; the ordinary dense hint cannot
        # turn that required rounding boundary off.
        return apply_linear(
            x,
            self.weight,
            self.bias,
            output_dtype=torch.bfloat16,
            out=out,
            operator=operator,
        )
