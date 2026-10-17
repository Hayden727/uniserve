"""Explicit NVFP4 projection epilogues using the deployed native kernels.

FlashInfer 0.6.18 supplies encoding and CUTLASS FP4 GEMM. PyTorch supplies
the block-scaled ``_scaled_mm`` bias epilogue for matrices with M > 32.
Weights and their numerical constants are borrowed from the shared layer;
prepared scratch belongs to its execution context.
"""

import torch

from uniserve.quantization import QuantizedTensor, ScaleLayout
from uniserve.quantization.nvfp4 import (
    NVFP4Epilogue,
    bias_over_alpha,
    quantize,
)
from uniserve.tensors import BufferConfig

from .base import Backend as BaseBackend
from .base import Operator as BaseOperator


class _Operator(BaseOperator):
    def __init__(self, *args, alpha, epilogue, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha, self.epilogue = alpha, epilogue
        self.workspace["one"].fill_(1.0)

    def __call__(self, x, bias, *, out):
        import flashinfer

        x = self._input(x, out)
        if x.shape[0] < 1:
            raise ValueError("NVFP4 projection requires at least one row")
        if isinstance(x, QuantizedTensor):
            raise ValueError(
                "NVFP4 deployment projection expects unencoded input"
            )
        if bias is not None and (
            bias.shape != (self.weight.shape[0],)
            or bias.dtype != torch.bfloat16
            or bias.device != x.device
        ):
            raise ValueError("NVFP4 bias must be BF16 on the projection device")
        # Native camera inputs may be strided or FP32. Their conversion is
        # part of this recipe, before activation quantization at global one.
        matrix = (
            x if x.dtype in (torch.float16, torch.bfloat16) else x.bfloat16()
        )
        values, scales = quantize(matrix, self.workspace["one"])
        fields = self.weight.buffers()
        rows, width = self.weight.shape
        weight_scales = fields["block_scale"].view(
            (rows + 127) // 128 * 128, (width // 16 + 3) // 4 * 4
        )
        fused = self.epilogue.fuse_bias and bias is not None
        if fused:
            folded = self.workspace["bias_over_alpha"]
            folded.copy_(bias_over_alpha(bias, self.alpha))
        if fused and x.shape[0] > 32:
            result = torch._scaled_mm(
                values.view(torch.float4_e2m1fn_x2),
                fields["values"].view(torch.float4_e2m1fn_x2).T,
                scales.view(torch.float8_e4m3fn),
                weight_scales.view(torch.float8_e4m3fn),
                bias=folded,
                out_dtype=torch.bfloat16,
            )
            # The BF16 GEMM store precedes the separate alpha multiply.
            # Moving alpha into the GEMM changes this rounding boundary.
            out.copy_(result)
            if self.epilogue.apply_alpha:
                out.mul_(self.alpha)
            return out

        alpha = (
            self.alpha if self.epilogue.apply_alpha else self.workspace["one"]
        )
        target = (
            out
            if out.is_contiguous()
            else torch.empty(out.shape, dtype=out.dtype, device=out.device)
        )
        flashinfer.mm_fp4(
            values,
            fields["values"].T,
            scales,
            weight_scales.T,
            alpha,
            out_dtype=torch.bfloat16,
            out=target,
            backend="cutlass",
        )
        if bias is not None:
            # Small fused calls use CUTLASS's alpha before the BF16 store,
            # followed by a distinct bias add. Unscaled calls use bias/alpha.
            target.add_(
                folded if fused and not self.epilogue.apply_alpha else bias
            )
        return out if target is out else out.copy_(target)


class Backend(BaseBackend):
    """Prepare the explicit NVFP4 recipe independently of dense GEMM policy."""

    name = "nvfp4"

    def __init__(self, *, alpha: torch.Tensor, epilogue: NVFP4Epilogue):
        self.alpha, self.epilogue = alpha, epilogue

    def workspace_buffers(
        self, weight, *, input_dtype, input_quantizer, max_rows, output_dtype
    ):
        if (
            not isinstance(weight, QuantizedTensor)
            or weight.quantizer.format != "nvfp4"
            or weight.scale_layout is not ScaleLayout.SWIZZLED_128X4
            or weight.ndim != 2
            or weight.shape[1] % 32
            or weight.dtype != torch.bfloat16
            or not weight.is_cuda
            or input_quantizer is not None
            or output_dtype != torch.bfloat16
            or max_rows < 1
            or self.alpha.shape != ()
            or self.alpha.dtype != torch.float32
            or self.alpha.device != weight.device
        ):
            raise ValueError(
                "NVFP4 deployment requires a CUDA BF16 encoded weight "
                "and FP32 alpha"
            )
        return {
            "one": BufferConfig((), torch.float32),
            "bias_over_alpha": BufferConfig((weight.shape[0],), torch.bfloat16),
        }

    def prepare(
        self,
        weight,
        *,
        input_dtype,
        input_quantizer,
        max_rows,
        output_dtype,
        workspace,
    ):
        return _Operator(
            weight,
            input_dtype=input_dtype,
            input_quantizer=input_quantizer,
            max_rows=max_rows,
            output_dtype=output_dtype,
            workspace=workspace,
            alpha=self.alpha,
            epilogue=self.epilogue,
        )
