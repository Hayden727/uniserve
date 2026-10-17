"""A declared single-row arithmetic recipe over an ordinary GEMM provider."""

import torch

from uniserve.nn.gemv import GemvReduction, validate_reduction
from uniserve.quantization import QuantizedTensor

from .base import Backend as _Backend
from .base import Operator as _Operator


class _GemvOperator(_Operator):
    def __init__(self, weight, *, ordinary, **options):
        super().__init__(weight, **options)
        self.ordinary = ordinary

    def __call__(self, x, bias, *, out):
        # Validate capacity and representation even on the one-row branch.
        prepared = self._input(x, out)
        if prepared.shape[0] == 1 and all(
            value.dtype == torch.float32
            and not isinstance(value, QuantizedTensor)
            for value in (prepared, self.weight, out)
        ):
            from uniserve_kernels.gemv import strided16

            return strided16(prepared, self.weight, bias, out=out)
        return self.ordinary(prepared, bias, out=out)


class Backend(_Backend):
    """Preserve the selected provider's resources and all other GEMM calls."""

    def __init__(self, ordinary: _Backend, reduction: GemvReduction):
        validate_reduction(reduction)
        if reduction != "strided16":
            raise ValueError("ordered GEMV backend requires an explicit recipe")
        self.ordinary = ordinary
        self.name = f"{ordinary.name}+{reduction}"

    def workspace_buffers(self, weight, **options):
        return self.ordinary.workspace_buffers(weight, **options)

    def prepare(self, weight, **options):
        return _GemvOperator(
            weight, ordinary=self.ordinary.prepare(weight, **options), **options
        )
