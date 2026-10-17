"""Local NVFP4 projection with an explicit deployment epilogue."""

import torch
from torch import nn

from uniserve.quantization import QuantizedTensor
from uniserve.quantization.nvfp4 import (
    NVFP4Epilogue,
    NVFP4Scales,
    encode_weight,
)

from . import _binding
from .functional._linear import apply_linear
from .linear import Linear


class NVFP4Linear(Linear):
    """An encoded local projection with distinct encoding and output scales.

    Convert only after strict BF16 checkpoint loading and final placement.
    ``weight`` remains an ordinary encoded PyTorch parameter; ``global_scale``
    and ``alpha`` are persistent numerical buffers. No request state, backend
    handle, workspace or device-selection policy is retained by this layer.
    """

    def __init__(
        self,
        weight: QuantizedTensor,
        scales: NVFP4Scales,
        *,
        bias: torch.Tensor | None,
        epilogue: NVFP4Epilogue,
    ):
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
        self.register_buffer("global_scale", scales.global_scale)
        self.register_buffer("alpha", scales.alpha)
        self.epilogue = epilogue

    @classmethod
    def from_linear(
        cls,
        source: Linear | nn.Linear,
        *,
        epilogue: NVFP4Epilogue = NVFP4Epilogue(),
    ) -> "NVFP4Linear":
        """Encode one loaded local projection without changing its source.

        Parallel Linear subclasses have collective semantics and must not be
        replaced by a local layer. BF16 source weights are encoded once on
        their existing CUDA device, with no CPU/device transfer or load hook.
        """
        if type(source) not in (Linear, nn.Linear) or isinstance(
            source.weight, QuantizedTensor
        ):
            raise ValueError(
                "NVFP4 conversion requires an unencoded local Linear"
            )
        if source.bias is not None and source.bias.dtype != torch.bfloat16:
            raise ValueError("NVFP4 conversion requires a BF16 bias")
        weight, scales = encode_weight(source.weight)
        return cls(weight, scales, bias=source.bias, epilogue=epilogue)

    def forward(
        self, x, *, round_before_bias=False, output_dtype=None, out=None
    ):
        """Project with the declared FP4 epilogue, always producing BF16.

        ``round_before_bias`` belongs to the dense GEMM contract; an FP4
        projection's two row-count branches have their own explicit rounding.
        """
        if round_before_bias:
            raise ValueError(
                "NVFP4 uses its explicit epilogue, not round_before_bias"
            )
        if output_dtype not in (None, torch.bfloat16):
            raise ValueError("NVFP4 deployment projection produces BF16 output")
        operator = _binding.matmul.get().get(id(self))
        if operator is None:
            # Standalone numerical calls use the same provider and allocation
            # contract as MatmulBinding, without storing runtime state here.
            from uniserve.runtime.backends.matmul.nvfp4 import Backend

            provider = Backend(alpha=self.alpha, epilogue=self.epilogue)
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
        return apply_linear(
            x,
            self.weight,
            self.bias,
            output_dtype=torch.bfloat16,
            out=out,
            operator=operator,
        )
