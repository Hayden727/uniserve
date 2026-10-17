"""Projection tails retain information until their declared BF16 stores."""

import pytest
import torch

from uniserve_models.wan import residual

pytestmark = pytest.mark.unit


def test_projection_bias_survives_residual_cancellation_and_normalization():
    hidden = torch.full((1, 1, 2), -256, dtype=torch.bfloat16)
    product = -hidden
    gate = torch.ones_like(hidden)
    bias = torch.tensor([1, 2], dtype=torch.bfloat16)
    result = residual.projection_residual(hidden, gate, product, bias)
    assert torch.equal(result, torch.tensor([[[1, 2]]], dtype=hidden.dtype))
    normalized = residual.projection_residual_affine(
        hidden, gate, product, bias, torch.ones(2), torch.zeros(2), 0.0
    )
    assert torch.equal(
        normalized, torch.tensor([[[-1, 1]]], dtype=hidden.dtype)
    )


def test_modulation_uses_unrounded_statistics_and_stored_residual():
    product = torch.tensor([[[256, 258]]], dtype=torch.bfloat16)
    zero = torch.zeros_like(product)
    bias = torch.ones(2, dtype=product.dtype)
    mean, variance, stored = residual.attention_residual_statistics(
        zero, zero, zero, bias * 0, product, bias
    )
    # Exact pre-store values [257,259] have mean258 and variance1;
    # the BF16 store rounds them to [256,260].
    assert torch.equal(mean, torch.tensor([[[258.0]]]))
    assert torch.equal(variance, torch.ones(1, 1, 1))
    assert torch.equal(
        stored, torch.tensor([[[256, 260]]], dtype=product.dtype)
    )
    actual = residual.modulate_from_statistics(
        stored, mean, variance, zero, zero, 0.0
    )
    assert torch.equal(actual, torch.tensor([[[-2, 2]]], dtype=product.dtype))
