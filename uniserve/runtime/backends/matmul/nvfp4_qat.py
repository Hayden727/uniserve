"""CUTLASS NVFP4 QAT forward with dynamic FP32 operand-global scaling."""

import torch

from uniserve.quantization import QuantizedTensor, ScaleLayout
from uniserve.quantization.nvfp4 import quantize
from uniserve.quantization.nvfp4_qat import global_scale
from uniserve.tensors import BufferConfig

from .base import Backend as BaseBackend
from .base import Operator as BaseOperator


class _Operator(BaseOperator):
    def __init__(self, *args, weight_global_scale, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_global_scale = weight_global_scale

    def __call__(self, x, bias, *, out):
        import flashinfer

        x = self._input(x, out)
        if (
            isinstance(x, QuantizedTensor)
            or x.dtype != torch.bfloat16
            or x.shape[0] < 1
        ):
            raise ValueError("NVFP4 QAT requires nonempty BF16 activations")
        if bias is not None and (
            bias.shape != (self.weight.shape[0],)
            or bias.dtype != torch.bfloat16
            or bias.device != x.device
        ):
            raise ValueError(
                "NVFP4 QAT bias must be BF16 on the projection device"
            )
        matrix = x.contiguous()
        activation_scale = self.workspace["activation_scale"]
        alpha = self.workspace["alpha"]
        activation_scale.copy_(global_scale(matrix))
        alpha.copy_(1.0 / (activation_scale * self.weight_global_scale))
        values, scales = quantize(matrix, activation_scale)
        fields = self.weight.buffers()
        rows, width = self.weight.shape
        weight_scales = fields["block_scale"].view(
            (rows + 127) // 128 * 128, (width // 16 + 3) // 4 * 4
        )
        target = (
            out
            if out.is_contiguous()
            else torch.empty_like(out, memory_format=torch.contiguous_format)
        )
        flashinfer.mm_fp4(
            values,
            fields["values"].T,
            scales,
            weight_scales.T,
            alpha,
            out_dtype=torch.bfloat16,
            out=target,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
        )
        if bias is not None:
            target.add_(bias)
        return out if target is out else out.copy_(target)


class Backend(BaseBackend):
    """Bind the declared QAT arithmetic independently of dense GEMM policy."""

    name = "nvfp4_qat"

    def __init__(self, *, global_scale):
        self.global_scale = global_scale

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
            or input_dtype != torch.bfloat16
            or input_quantizer is not None
            or output_dtype != torch.bfloat16
            or max_rows < 1
            or self.global_scale.shape != ()
            or self.global_scale.dtype != torch.float32
            or self.global_scale.device != weight.device
        ):
            raise ValueError(
                "NVFP4 QAT requires CUDA BF16 operands and an FP32 weight scale"
            )
        return {
            "activation_scale": BufferConfig((), torch.float32),
            "alpha": BufferConfig((), torch.float32),
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
            weight_global_scale=self.global_scale,
        )
