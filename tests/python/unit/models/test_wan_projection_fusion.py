"""Projection epilogues preserve the declared BF16 store boundaries."""

import json
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from uniserve.model import TextSize
from uniserve.nn.linear import Linear
from uniserve.runtime import ExecutionContext
from uniserve_models.wan import fusion
from uniserve_models.wan.config import Config, TransformerConfig, read_config
from uniserve_models.wan.ucpe import CameraConfig, UcpeConfig, bias_project

pytestmark = pytest.mark.unit


def test_projection_recipe_round_trip_and_default(tmp_path):
    assert TransformerConfig().fusion_recipe == "elementwise"
    config = Config(
        transformer=TransformerConfig(
            compile_elementwise=True, fusion_recipe="plucker_prefix"
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["WanCameraPipeline"], **asdict(config)})
    )
    assert read_config(tmp_path, None, sources={}) == config
    with pytest.raises(ValueError, match="does not apply to UCPE"):
        Config(transformer=config.transformer, ucpe=UcpeConfig())


@pytest.mark.parametrize(
    "change",
    [
        {"compile_elementwise": False},
        {"use_scale_embedding": False},
        {"fused_modulation": False},
        {"fusion_recipe": "unknown"},
    ],
)
def test_projection_recipe_rejects_incompatible_math(change):
    values = {
        "compile_elementwise": True,
        "fusion_recipe": "plucker_prefix",
        **change,
    }
    with pytest.raises(ValueError):
        TransformerConfig(**values)


@torch.inference_mode()
@pytest.mark.parametrize("round_before_bias", [False, True])
def test_unbiased_projection_retains_runtime_binding_and_bias_parameter(
    round_before_bias,
):
    layer = Linear(2, 1, dtype=torch.bfloat16)
    layer.weight.copy_(torch.tensor([[256, 1]], dtype=torch.bfloat16))
    layer.bias.fill_(-256)
    x = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    out = torch.empty(1, 1, 1, dtype=torch.bfloat16)
    with ExecutionContext(layer, matmul="torch") as context:
        context.prepare(TextSize(1, 1))
        result = layer(
            x, include_bias=False, round_before_bias=round_before_bias, out=out
        )
        assert result is out
        assert torch.equal(result, torch.full_like(out, 256))
        assert torch.equal(
            layer.bias, torch.tensor([-256], dtype=torch.bfloat16)
        )
        assert torch.equal(layer(x), torch.ones_like(out))


def test_combined_camera_store_preserves_cancellation():
    hidden = torch.tensor([[[-1, 1]]], dtype=torch.bfloat16)
    zeros = torch.zeros_like(hidden)
    camera = torch.full_like(hidden, 256)
    bias = torch.ones(2, dtype=torch.bfloat16)
    scale = torch.full((1, 2), -256, dtype=torch.bfloat16)
    actual = fusion.plucker_modulate(
        hidden, zeros, zeros, camera, bias, scale, bias * 0, 0.0
    )
    assert actual.shape == hidden.shape
    assert torch.equal(actual, torch.tensor([[[0, 2]]], dtype=torch.bfloat16))
    assert not torch.equal(
        actual, hidden + ((camera + bias) + scale.unsqueeze(1))
    )


def test_scale_bias_silu_keeps_fp32_until_activation_store():
    product = torch.tensor([[-3.21875]], dtype=torch.bfloat16)
    bias = torch.tensor([-0.004302978515625], dtype=torch.bfloat16)
    value = product.float() + bias.float()
    expected = (value / (1 + torch.exp(-value))).to(product.dtype)
    actual = fusion.bias_silu(product, bias)
    assert actual.shape == product.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, F.silu(product + bias))


def test_q_bias_rms_keeps_bias_and_weight_inside_final_store():
    product = torch.tensor([[[256, 1]]], dtype=torch.bfloat16)
    bias = torch.tensor([1, 0], dtype=torch.bfloat16)
    weight = torch.ones(2, dtype=torch.bfloat16)
    value = product.double() + bias.double()
    expected = (
        value / (value.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    ).to(product.dtype)
    actual = fusion.bias_rms(product, bias, weight, 1e-6)
    assert actual.shape == product.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(
        actual, fusion.rms_scale(product + bias, weight, 1e-6)
    )


def test_ucpe_prefix_recipe_round_trip(tmp_path):
    config = Config(
        transformer=TransformerConfig(
            compile_elementwise=True,
            fusion_recipe="ucpe_prefix",
            use_scale_embedding=False,
        ),
        ucpe=UcpeConfig(camera=CameraConfig(projection="elementwise")),
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["WanCameraPipeline"], **asdict(config)})
    )
    assert read_config(tmp_path, None, sources={}) == config
    with pytest.raises(ValueError, match="requires a UCPE architecture"):
        Config(transformer=config.transformer)
    with pytest.raises(ValueError, match="dense elementwise orbit cameras"):
        Config(transformer=config.transformer, ucpe=UcpeConfig())


def test_camera_bias_and_projection_share_final_bf16_store():
    product = torch.tensor([[[256, -256, 1, 0]]], dtype=torch.bfloat16)
    bias = torch.tensor([1, 0, 0, 0], dtype=torch.bfloat16)
    matrix = torch.eye(4, dtype=torch.bfloat16).reshape(1, 1, 4, 4)
    matrix[..., 0, 1] = 1
    actual = bias_project(product, bias, matrix, 1)
    assert actual.shape == (1, 1, 1, 4)
    assert actual.dtype == product.dtype
    assert torch.equal(
        actual, torch.tensor([[[[1, -256, 1, 0]]]], dtype=torch.bfloat16)
    )
