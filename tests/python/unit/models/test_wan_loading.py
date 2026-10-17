"""The public loader places every Wan parameter and rejects partial models."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    from uniserve_models.wan.config import TransformerConfig
    from uniserve_models.wan.transformer import PluckerTransformer

    config = TransformerConfig(
        num_layers=1,
        num_heads=2,
        head_dim=8,
        ffn_dim=24,
        text_dim=12,
        freq_dim=8,
        scale_bands=2,
    )
    torch.manual_seed(15)
    source = PluckerTransformer(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(std=0.1)
    values = source.state_dict()
    save_file(values, tmp_path / "model.safetensors")
    return tmp_path, config, values


def test_strict_loading_preserves_all_parameters(checkpoint):
    from uniserve_models.wan.loading import load_transformer

    path, config, expected = checkpoint
    loaded = load_transformer(path, config, device="cpu")
    actual = loaded.model.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def test_catalog_loads_wan_denoiser(checkpoint):
    from uniserve.model import Denoiser
    from uniserve_models import loading

    path, config, expected = checkpoint
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WanCameraPipeline"],
                "transformer": asdict(config),
            }
        )
    )
    resolved = loading.read_config(path)
    loaded = loading.load_model(resolved, device="cpu")
    assert isinstance(loaded.model.transformer, Denoiser)
    assert loaded.model.transformer.num_steps == 4
    actual = loaded.model.transformer.network.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("defect", ["missing", "extra", "shape"])
def test_loading_rejects_camera_checkpoint_defects(checkpoint, defect):
    from uniserve_models.wan.loading import load_transformer

    path, config, values = checkpoint
    if defect == "missing":
        values.pop("blocks.0.cam_encoder.weight")
    elif defect == "extra":
        values["unmapped_camera_gate"] = torch.ones(1)
    else:
        values["blocks.0.cam_encoder.weight"] = torch.ones(1, 6)
    save_file(values, path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError)):
        load_transformer(path, config, device="cpu")
