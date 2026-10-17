"""Optional Inductor fusion of Wan's per-block elementwise chains.

Wan blocks spend a large share of each step in memory-bound chains around the
GEMMs and attention: FP32 LayerNorm plus AdaLN modulation, FP32 RMSNorm plus
interleaved RoPE on Q/K, and gated residual updates. The eager blocks keep the
checkpoint's exact operation order (the parity recipe). With
``TransformerConfig.compile_elementwise`` the blocks call these functions
instead; static argument layouts share compiled programs across blocks. Each
layout has its own Dynamo code cache. Inductor keeps intermediates in FP32
registers, so results differ from
eager by BF16 rounding only (protocol Tier E, never Tier P). The functions hold
no request state and launch only Triton kernels, so runtime CUDA graph capture
records them like any other kernel after warmup.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import struct
from types import FunctionType

import torch
from torch.nn import functional as F


def _rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved-pair RoPE in FP32. x: (B, S, H, D); cos/sin: (S, D)."""
    real, imaginary = x.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-imaginary, real), dim=-1).flatten(-2)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def layer_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    fused: bool,
) -> torch.Tensor:
    """Affine-free FP32 LayerNorm, then ``shift + norm * (1 + scale)``.

    x: (B, S, W); scale/shift: (B, 1, W). Returns x's dtype.
    """
    value = F.layer_norm(x.float(), x.shape[-1:], None, None, eps).to(x.dtype)
    if fused:
        return torch.addcmul(shift, value, 1 + scale)
    return value * (1 + scale) + shift


def layer_norm_affine(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """FP32 LayerNorm with learned affine. x: (B, S, W). Returns x's dtype."""
    return F.layer_norm(
        x.float(), x.shape[-1:], weight.float(), bias.float(), eps
    ).to(x.dtype)


def rms_scale(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """FP32 RMSNorm rounded before learned scale. x: (B, S, W)."""
    value = x.float()
    normalized = value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + eps
    )
    return normalized.to(x.dtype) * weight


def rms_rotate(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    heads: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Full-width RMSNorm and RoPE: (B, S, W) -> (B, S, H, D)."""
    normalized = rms_scale(x, weight, eps).unflatten(-1, (heads, -1))
    return _rotate(normalized.float(), cos, sin).to(x.dtype)


def gated_residual(
    hidden: torch.Tensor, gate: torch.Tensor, value: torch.Tensor, fused: bool
) -> torch.Tensor:
    """``hidden + gate * value``. hidden/value: (B, S, W); gate: (B, 1, W)."""
    if fused:
        return torch.addcmul(hidden, gate, value)
    return hidden + gate * value


def bias_silu(product: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """BF16 GEMM product -> FP32 bias and SiLU -> one BF16 store."""
    value = product.float() + bias.float()
    return (value * torch.sigmoid(value)).to(product.dtype)


def plucker_modulate(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    camera_product: torch.Tensor,
    camera_bias: torch.Tensor,
    scale_product: torch.Tensor,
    scale_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Join AdaLN and camera branches before the projection-input BF16 store.

    Hidden/camera products are (B,S,W); modulation is (B,1,W), and the
    unbiased scale product is (B,W). Each GEMM product has already stored BF16.
    """
    normalized = F.layer_norm(
        hidden.float(), hidden.shape[-1:], None, None, eps
    )
    modulated = torch.addcmul(shift.float(), normalized, 1 + scale.float())
    camera = camera_product.float() + camera_bias.float()
    scaled = scale_product.float() + scale_bias.float()
    return (modulated + (camera + scaled.unsqueeze(1))).to(hidden.dtype)


def bias_rms(
    product: torch.Tensor, bias: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """BF16 unbiased (B,S,W) product -> FP32 bias/RMS/scale -> BF16."""
    value = product.float() + bias.float()
    normalized = value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(product.dtype)


_specializations = itertools.count()


def _signature(arguments):
    """Describe borrowed inputs without retaining their values or addresses."""
    tensors, storages = {}, {}
    result = []
    for value in arguments:
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise TypeError("Wan fusion requires strided tensor inputs")
            # Duplicate Python arguments and distinct views of shared storage
            # are separate alias contracts. Only their relative group numbers
            # enter the cache; the actual identities live for this call only.
            tensor = tensors.setdefault(id(value), len(tensors))
            address = (value.device, value.untyped_storage().data_ptr())
            storage = storages.setdefault(address, len(storages))
            result.append(
                (
                    type(value),
                    value.device,
                    value.dtype,
                    tuple(value.shape),
                    value.stride(),
                    value.storage_offset(),
                    value.requires_grad,
                    torch.is_inference(value),
                    value.is_conj(),
                    value.is_neg(),
                    tensor,
                    storage,
                )
            )
        elif type(value) in (bool, int, float, str, type(None)):
            # Keep bool/int distinct and preserve float signed zero and bits.
            scalar = struct.pack("!d", value) if type(value) is float else value
            result.append((type(value), scalar))
        else:
            raise TypeError(
                "Wan fusion accepts only tensors and numerical scalars"
            )
    return (
        tuple(result),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        tuple(
            (
                torch.is_autocast_enabled(device),
                torch.get_autocast_dtype(device),
            )
            for device in ("cpu", "cuda")
        ),
    )


@functools.cache
def compiled(function):
    """Share static programs by numerical layout with independent guard caches.

    The runtime warms all admitted layouts before recording any CUDA graphs.
    Each layout keeps ``dynamic=False, fullgraph=True`` and the original
    operation body. Distinct code objects are necessary: creating multiple
    torch.compile wrappers around one function still shares its Dynamo limit.
    The cache owns only callables and immutable metadata, never request tensors.
    """
    parameters = inspect.signature(function)

    @functools.cache
    def specialize(signature):
        name = f"{function.__name__}__static_{next(_specializations)}"
        code = function.__code__.replace(co_name=name, co_qualname=name)
        variant = FunctionType(
            code,
            function.__globals__,
            name,
            function.__defaults__,
            function.__closure__,
        )
        variant.__kwdefaults__ = function.__kwdefaults__
        variant.__module__ = function.__module__
        # Do not install __wrapped__: compiler unwrapping would recover the
        # original shared code object and collapse these independent caches.
        return torch.compile(variant, dynamic=False, fullgraph=True)

    def dispatch(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        return specialize(_signature(bound.arguments.values()))(*args, **kwargs)

    return dispatch
