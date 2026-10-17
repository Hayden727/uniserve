"""PyTorch Flash and memory-efficient SDPA with explicit kernel selection.

The portable torch provider follows PyTorch's global/default kernel policy.
This provider fixes the fused pair for reproducible numerical execution
across PyTorch/cuDNN releases. Unsupported inputs fail in SDPA; the policy
does not silently select a math or cuDNN implementation with other rounding.
"""

from torch.nn.attention import SDPBackend, sdpa_kernel

from .torch import Backend as TorchBackend
from .torch import _TorchOperator


class _FusedOperator(_TorchOperator):
    def __call__(self, q, k, v, batch, *, scale, out):
        # Kernel selection belongs to this bound invocation. Restore the
        # caller's policy after both eager execution and graph capture.
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        ):
            return super().__call__(q, k, v, batch, scale=scale, out=out)


class Backend(TorchBackend):
    name = "torch_fused"
    operator_class = _FusedOperator
