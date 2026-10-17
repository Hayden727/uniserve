"""FP32 operand scales for the native NVFP4 training-forward recipe.

Inference weights are immutable, so their encoding is computed once after
loading. Activations retain the native per-projection dynamic global scale.
Neither scale passes through BF16 and zero operands use the native clamp.
"""

import torch

from .nvfp4 import quantize
from .quantizer import Quantizer
from .tensor import QuantizedTensor, ScaleLayout


def global_scale(value: torch.Tensor) -> torch.Tensor:
    """Return the operand's FP32 scale with compiled-recipe CUDA rounding.

    CPU evaluates the mathematical expression; CUDA preserves the native
    Triton reciprocal followed by FP32 multiplication, which is not bitwise
    interchangeable with eager ATen reciprocal or Tensor/Tensor division.
    """
    if value.numel() == 0 or value.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("NVFP4 QAT scales require a nonempty 16-bit operand")
    maximum = value.float().abs().nan_to_num().amax().clamp(min=1e-12)
    if value.is_cuda:
        from uniserve_kernels.nvfp4_qat import global_scale as cuda_scale

        return cuda_scale(maximum)
    return (448.0 * 6.0) / maximum


def encode_weight(weight: torch.Tensor) -> tuple[QuantizedTensor, torch.Tensor]:
    """Encode a loaded BF16 weight, retaining its exact FP32 global scale."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        raise ValueError("NVFP4 QAT requires placed CUDA BF16 weights")
    scale = global_scale(weight.detach())
    values, blocks = quantize(weight.detach(), scale)
    encoded = Quantizer("nvfp4").from_tensors(
        {
            "values": values,
            "block_scale": blocks.reshape(-1),
            "tensor_scale": 1.0 / scale,
        },
        shape=tuple(weight.shape),
        dtype=weight.dtype,
        scale_layout=ScaleLayout.SWIZZLED_128X4,
    )
    return encoded, scale
