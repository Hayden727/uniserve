"""NVFP4 deployment scales with separately rounded GEMM epilogues.

These explicit numerical rules differ from generic NVFP4: weights encode
with the FP32 global scale ``2688 / amax``, but the output multiplier is
the reciprocal of that scale rounded to BF16. Activations encode with
global scale one. FlashInfer 0.6.18 provides the packed E2M1 encoder;
the layout and explicit zero row padding match its native deployment use.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .quantizer import Quantizer
from .tensor import QuantizedTensor, ScaleLayout


@dataclass(frozen=True, slots=True)
class NVFP4Epilogue:
    """Select bias fusion and whether this projection applies its alpha.

    A caller disabling ``apply_alpha`` owns its numerical consequence: an
    ensuing RMSNorm consumes the unscaled output, or a residual gate must
    multiply by the projection's alpha before its separately rounded tail.
    """

    fuse_bias: bool = True
    apply_alpha: bool = True

    def __post_init__(self):
        if (
            type(self.fuse_bias) is not bool
            or type(self.apply_alpha) is not bool
        ):
            raise ValueError("NVFP4 epilogue choices must be boolean")


@dataclass(frozen=True, slots=True)
class NVFP4Scales:
    """Borrowed scalar tensors for encoding and output scaling."""

    global_scale: torch.Tensor
    stored_global_scale: torch.Tensor
    alpha: torch.Tensor


def weight_scales(weight: torch.Tensor) -> NVFP4Scales:
    """Compute the native BF16 weight statistic without reciprocal rewrites.

    ``abs`` and ``nan_to_num`` run in the source BF16 representation before
    the scalar maximum converts to FP32. Zero weights intentionally retain
    the native infinite encoder scale and zero alpha; no epsilon is added.
    This helper does not encode or select a device.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.numel():
        raise ValueError(
            "NVFP4 deployment weights require a nonempty BF16 matrix"
        )
    maximum = weight.detach().abs().nan_to_num().amax().to(torch.float32)
    # Keep scalar / Tensor: Tensor.__rdiv__ uses reciprocal then multiply.
    # A Tensor/Tensor division or a decode-scale round trip can differ by an
    # FP32 ULP before block-scale quantization.
    global_scale = (448 * 6) / maximum
    stored_global_scale = global_scale.to(torch.bfloat16)
    # Multiplication by FP32 one promotes the stored BF16 scalar before
    # taking its reciprocal, exactly as the activation-global-one recipe.
    alpha = 1.0 / (torch.ones_like(global_scale) * stored_global_scale)
    return NVFP4Scales(global_scale, stored_global_scale, alpha)


def bias_over_alpha(bias: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Round the FP32 bias/alpha quotient once to the BF16 epilogue bias."""
    return (bias.detach().float() / alpha).to(torch.bfloat16)


def fold_gate(gate: torch.Tensor, alpha: float) -> torch.Tensor:
    """Multiply a residual gate by a load-time scalar before its next store.

    Resolve ``float(layer.alpha)`` outside capture. Keeping the value as a
    Python scalar matches the native gate's dtype promotion and leaves the
    multiply's BF16 rounding separate from the subsequent residual tail.
    """
    return gate if alpha == 1.0 else gate * alpha


def quantize(
    matrix: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a matrix with an explicit FP32 global scale and 128x4 layout.

    The caller supplies a scalar on the input device. No statistic is
    recomputed and no inverse-scale round trip is introduced. Return packed
    values at logical row count and scales retaining padded physical shape.
    Both returned tensors belong to the caller's execution allocation scope.
    """
    if (
        matrix.ndim != 2
        or matrix.dtype not in (torch.bfloat16, torch.float16)
        or matrix.shape[1] % 32
        or not matrix.is_cuda
        or global_scale.shape != ()
        or global_scale.dtype != torch.float32
        or global_scale.device != matrix.device
    ):
        raise ValueError(
            "NVFP4 encoding requires a CUDA 16-bit K32 matrix and FP32 scalar"
        )
    import flashinfer

    rows = matrix.shape[0]
    padded = F.pad(matrix.contiguous(), (0, 0, 0, (-rows) % 128))
    values, scales = flashinfer.nvfp4_quantize(
        padded,
        global_scale,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
    )
    return values.narrow(0, 0, rows), scales


def encode_weight(weight: torch.Tensor) -> tuple[QuantizedTensor, NVFP4Scales]:
    """Encode an already-loaded BF16 weight once, before runtime capture.

    The ordinary QuantizedTensor retains its true decode scale. The distinct
    BF16-derived GEMM alpha is returned separately, so dequantization and
    existing generic quantization never silently adopt another recipe.
    """
    scales = weight_scales(weight)
    values, blocks = quantize(weight.detach(), scales.global_scale)
    encoded = Quantizer("nvfp4").from_tensors(
        {
            "values": values,
            "block_scale": blocks.reshape(-1),
            "tensor_scale": 1.0 / scales.global_scale,
        },
        shape=tuple(weight.shape),
        dtype=weight.dtype,
        scale_layout=ScaleLayout.SWIZZLED_128X4,
    )
    return encoded, scales
