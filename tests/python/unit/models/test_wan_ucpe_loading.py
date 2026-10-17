"""Strict UCPE loading preserves the complete Quality/Fast student state."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from uniserve_models.wan.config import Config, TransformerConfig
from uniserve_models.wan.loading import load_ucpe_transformer
from uniserve_models.wan.ucpe import CameraConfig, UcpeConfig, UcpeTransformer

pytestmark = pytest.mark.unit


@pytest.fixture
def ucpe_checkpoint(tmp_path):
    config = TransformerConfig(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        ffn_dim=24,
        text_dim=12,
        freq_dim=8,
        use_scale_embedding=False,
    )
    camera = CameraConfig(1, 8, compress_gate=True)
    torch.manual_seed(471)
    source = UcpeTransformer(config, camera, main_compress_gate=True).to(
        torch.bfloat16
    )
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(std=0.1)
    values = source.state_dict()
    save_file(values, tmp_path / "model.safetensors")
    return tmp_path, config, camera, values


def test_ucpe_loading_preserves_camera_and_gate_parameters(ucpe_checkpoint):
    path, config, camera, expected = ucpe_checkpoint
    result = load_ucpe_transformer(
        path, config, camera, main_compress_gate=True, device="cpu"
    )
    actual = result.model.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    report = result.reports[0]
    assert report.loaded == expected.keys()
    assert not (report.skipped or report.missing or report.unexpected)
    assert not report.incomplete


def test_catalog_loads_ucpe_camera_capability_with_complete_state(
    ucpe_checkpoint,
):
    from uniserve.model.camera import CameraDenoiser
    from uniserve_models import loading

    path, transformer, camera, expected = ucpe_checkpoint
    config = Config(
        transformer, ucpe=UcpeConfig(camera, main_compress_gate=True)
    )
    (path / "config.json").write_text(
        json.dumps({"architectures": ["WanCameraPipeline"], **asdict(config)})
    )
    resolved = loading.read_config(path)
    loaded = loading.load_model(resolved, device="cpu")
    denoiser = loaded.model.transformer
    assert isinstance(denoiser, CameraDenoiser)
    assert denoiser.config == config
    actual = denoiser.network.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "name",
    [
        "cam_self_attn.q_proj.weight",
        "cam_self_attn.k_proj.weight",
        "cam_self_attn.v_proj.weight",
        "cam_self_attn.out_proj.weight",
        "cam_self_attn.out_proj.bias",
        "cam_self_attn.to_gate_compress.weight",
        "to_gate_compress.weight",
    ],
)
def test_ucpe_loading_rejects_missing_camera_or_gate(ucpe_checkpoint, name):
    path, config, camera, values = ucpe_checkpoint
    values.pop("blocks.0." + name)
    save_file(values, path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError), match="missing"):
        load_ucpe_transformer(
            path, config, camera, main_compress_gate=True, device="cpu"
        )


@pytest.mark.parametrize("name", ["q_proj", "out_proj"])
def test_ucpe_loading_rejects_wrong_camera_shape(ucpe_checkpoint, name):
    path, config, camera, values = ucpe_checkpoint
    values[f"blocks.0.cam_self_attn.{name}.weight"] = torch.ones(1, 1)
    save_file(values, path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError)):
        load_ucpe_transformer(
            path, config, camera, main_compress_gate=True, device="cpu"
        )


def test_ucpe_loading_rejects_plucker_parameters(ucpe_checkpoint):
    path, config, camera, values = ucpe_checkpoint
    values["blocks.0.cam_encoder.weight"] = torch.ones(16, 6)
    save_file(values, path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError), match="unexpected"):
        load_ucpe_transformer(
            path, config, camera, main_compress_gate=True, device="cpu"
        )


def test_ucpe_loading_rejects_incomplete_export(ucpe_checkpoint):
    path, config, camera, _ = ucpe_checkpoint
    (path / ".incomplete").touch()
    with pytest.raises(ValueError, match="incomplete"):
        load_ucpe_transformer(
            path, config, camera, main_compress_gate=True, device="cpu"
        )
