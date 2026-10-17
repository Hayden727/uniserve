"""Wan projection tails with explicit accumulation and store boundaries.

GEMMs supply BF16 products without bias. Residuals, bias and normalization
statistics accumulate in FP32; callers decide where the result stores BF16.
"""

import torch
from torch.nn import functional as F


def projection_residual(hidden, gate, product, bias):
    """FP32 bias/gate/residual followed by one store in hidden's dtype."""
    value = hidden.float() + gate.float() * (product.float() + bias.float())
    return value.to(hidden.dtype)


def projection_residual_affine(
    hidden, gate, product, bias, weight, offset, eps
):
    """Normalize the unrounded main-attention residual, then store BF16."""
    value = hidden.float() + gate.float() * (product.float() + bias.float())
    return F.layer_norm(
        value, value.shape[-1:], weight.float(), offset.float(), eps
    ).to(hidden.dtype)


def attention_residual_statistics(
    hidden, gate, main_product, main_bias, text_product, text_bias
):
    """Store residual and its pre-rounding FP32 statistics across a call.

    The following modulation consumes the stored residual and these statistics.
    Recomputing them from the rounded residual changes the native contract.
    """
    main = hidden.float() + gate.float() * (
        main_product.float() + main_bias.float()
    )
    value = main + (text_product.float() + text_bias.float())
    variance, mean = torch.var_mean(value, dim=-1, correction=0, keepdim=True)
    return mean, variance, value.to(hidden.dtype)


def modulate_from_statistics(rounded, mean, variance, scale, shift, eps):
    """Reload BF16 residual with its FP32 statistics, then AdaLN and store."""
    value = (rounded.float() - mean) * torch.rsqrt(variance + eps)
    return (shift.float() + value * (scale.float() + 1)).to(rounded.dtype)


def projection_gelu(product, bias):
    """Apply bias and tanh GELU in FP32 before the activation store."""
    return F.gelu(product.float() + bias.float(), approximate="tanh").to(
        product.dtype
    )
