"""Wan Inductor elementwise fusion: eager equivalence and compiled tolerance."""

import pytest
import torch

from uniserve.nn.norm import LayerNorm, RMSNorm
from uniserve_models.wan import fusion
from uniserve_models.wan.config import TransformerConfig
from uniserve_models.wan.transformer import rotate

pytestmark = pytest.mark.unit

WIDTH, HEADS, TOKENS = 64, 4, 24


def _inputs(device="cpu"):
    generator = torch.Generator().manual_seed(7)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).to(device)

    hidden = randn(1, TOKENS, WIDTH, scale=3.0).bfloat16()
    scale = randn(1, 1, WIDTH, scale=0.2).bfloat16()
    shift = randn(1, 1, WIDTH, scale=0.2).bfloat16()
    angles = randn(TOKENS, WIDTH // HEADS)
    return hidden, scale, shift, angles.cos(), angles.sin()


@pytest.mark.parametrize("fused", [True, False])
def test_layer_norm_modulate_matches_eager_block_order(fused):
    hidden, scale, shift, _, _ = _inputs()
    value = LayerNorm(WIDTH, eps=1e-6, elementwise_affine=False)(hidden)
    expected = (
        torch.addcmul(shift, value, 1 + scale)
        if fused
        else value * (1 + scale) + shift
    )
    actual = fusion.layer_norm_modulate(hidden, scale, shift, 1e-6, fused)
    assert actual.shape == (1, TOKENS, WIDTH)
    assert torch.equal(actual, expected)


def test_rms_rotate_matches_norm_then_rotate():
    hidden, _, _, cos, sin = _inputs()
    norm = RMSNorm(WIDTH, 1e-6, round_before_scale=True)
    norm.weight.data = torch.linspace(0.5, 1.5, WIDTH).bfloat16()
    expected = rotate(norm(hidden).unflatten(-1, (HEADS, -1)), (cos, sin))
    actual = fusion.rms_rotate(hidden, norm.weight, norm.eps, HEADS, cos, sin)
    assert actual.shape == (1, TOKENS, HEADS, WIDTH // HEADS)
    assert torch.equal(actual, expected)


def test_affine_layer_norm_matches_module():
    hidden, _, _, _, _ = _inputs()
    norm = LayerNorm(WIDTH, eps=1e-6, elementwise_affine=True).bfloat16()
    norm.weight.data.uniform_(0.5, 1.5)
    norm.bias.data.uniform_(-0.1, 0.1)
    actual = fusion.layer_norm_affine(hidden, norm.weight, norm.bias, norm.eps)
    assert torch.equal(actual, norm(hidden))


def test_fusion_requires_default_rms_reduction():
    with pytest.raises(ValueError, match="default RMS"):
        TransformerConfig(
            compile_elementwise=True, rms_reduction_order="ascending"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_chains_stay_within_bf16_rounding_and_replay_in_graphs():
    hidden, scale, shift, cos, sin = _inputs("cuda")
    weight = torch.linspace(0.5, 1.5, WIDTH, device="cuda").bfloat16()
    eager = fusion.rms_rotate(hidden, weight, 1e-6, HEADS, cos, sin)
    compiled = fusion.compiled(fusion.rms_rotate)
    # Warm outside capture, as the runtime does before graph capture.
    first = compiled(hidden, weight, 1e-6, HEADS, cos, sin)
    error = (first.float() - eager.float()).abs().max()
    assert (
        error <= 2 * torch.finfo(torch.bfloat16).eps * eager.float().abs().max()
    )

    static = hidden.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(static, weight, 1e-6, HEADS, cos, sin)
    changed = hidden * 0.5
    static.copy_(changed)
    graph.replay()
    assert torch.equal(output, compiled(changed, weight, 1e-6, HEADS, cos, sin))
    modulated = fusion.compiled(fusion.layer_norm_modulate)(
        hidden, scale, shift, 1e-6, True
    )
    assert modulated.shape == (1, TOKENS, WIDTH)
