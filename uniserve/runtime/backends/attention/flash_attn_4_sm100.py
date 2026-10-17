"""Dense SM100 attention with a fixed BF16 softmax rounding recipe."""

from uniserve.nn.attention.inputs import DenseInput

from . import Backend as BaseBackend
from . import Operator


class _DenseOperator(Operator):
    builds_launch_plan = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plans = {}

    def __call__(self, q, k, v, batch, *, scale, out):
        from uniserve_kernels.attention.dense_sm100 import DenseAttention

        if (
            not isinstance(batch, DenseInput)
            or batch.causal
            or batch.mask is not None
        ):
            raise ValueError(
                "SM100 dense FA4 requires unmasked noncausal attention"
            )
        if self.cache is not None:
            raise ValueError("SM100 dense FA4 does not consume a prefix cache")
        self._validate(q, k, v, batch, out)
        packed = q.ndim == 3
        query, key, value = (
            (
                tensor.unsqueeze(0) if packed else tensor.transpose(1, 2)
            ).contiguous()
            for tensor in (q, k, v)
        )
        destination = out.unsqueeze(0) if packed else out.transpose(1, 2)
        signature = tuple(
            (tensor.shape, tensor.stride(), tensor.dtype, tensor.device)
            for tensor in (query, key, value, destination)
        )
        plan = self.plans.get(signature)
        if plan is None:
            plan = DenseAttention(
                query, key, value, destination, None, scale=scale
            )
            self.plans[signature] = plan
        plan(query, key, value, destination, None, scale=scale)
        return out


class Backend(BaseBackend):
    name = "flash_attn_4_sm100"
    operator_class = _DenseOperator

    def workspace_buffers(self, **kwargs):
        return {}
