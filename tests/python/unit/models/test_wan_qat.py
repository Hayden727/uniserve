"""Native checkpoint activation and public QAT projection eligibility."""

import json
from dataclasses import asdict, replace

import pytest
import torch
from safetensors.torch import save_file

from uniserve.nn.attention.vsa import Recipe
from uniserve.quantization.nvfp4_qat import global_scale
from uniserve_models.wan.config import (
    Config,
    NVFP4Config,
    QATConfig,
    TransformerConfig,
    VsaConfig,
    read_config,
)
from uniserve_models.wan.loading import (
    load_ucpe_transformer,
    qat_projection_names,
    quantize_transformer,
)
from uniserve_models.wan.ucpe import CameraConfig, UcpeTransformer

pytestmark = pytest.mark.unit


def test_formal_config_round_trip_retains_numerical_choices(
    tmp_path,
):
    config = Config(
        TransformerConfig(
            qat=QATConfig(600, 300, 2),
            fused_modulation=False,
            vsa=VsaConfig((1, 32, 8), 0.5, Recipe.FA4_128X128),
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["WanCameraPipeline"], **asdict(config)})
    )
    assert read_config(tmp_path, None, sources=()) == config
    with pytest.raises(ValueError, match="distinct recipes"):
        replace(config.transformer, nvfp4=NVFP4Config())


def test_qat_retains_exact_edges_camera_and_gate_precision():
    config = TransformerConfig(
        qat=QATConfig(600, 300, 2), use_scale_embedding=False
    )
    with torch.device("meta"):
        model = UcpeTransformer(
            config, CameraConfig(8, 128), main_compress_gate=True
        ).to(torch.bfloat16)
    names = qat_projection_names(model)
    expected = {
        f"blocks.{block}.{projection}"
        for block in range(2, 38)
        for projection in (
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out",
            "ffn.fc_in",
            "ffn.fc_out",
        )
    }
    assert set(names) == expected
    assert len(names) == 360
    assert all("cam_" not in name and "gate" not in name for name in names)
    with pytest.raises(ValueError, match="placed CUDA BF16"):
        quantize_transformer(model)


@pytest.mark.parametrize(
    "step,active", [(299, False), (300, True), (600, True)]
)
def test_checkpoint_activation_boundary(step, active):
    recipe = QATConfig(step, 300, 2)
    assert recipe.active is active
    if not active:
        with torch.device("meta"):
            model = UcpeTransformer(
                TransformerConfig(qat=recipe), CameraConfig()
            )
        assert quantize_transformer(model) == ()


def test_qat_global_scale_has_fp32_clamp_and_no_bf16_round_trip():
    values = torch.tensor([0, -0.310546875, 0.25], dtype=torch.bfloat16)
    actual = global_scale(values)
    expected = 2688.0 / values.float().abs().max()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual != actual.bfloat16().float()
    torch.testing.assert_close(
        global_scale(torch.zeros_like(values)),
        2688.0 / torch.tensor(1e-12),
        rtol=0,
        atol=0,
    )


def test_public_loader_replays_qat_activation_after_complete_weights(tmp_path):
    config = TransformerConfig(
        num_layers=3,
        num_heads=2,
        head_dim=32,
        ffn_dim=96,
        text_dim=32,
        freq_dim=32,
        use_scale_embedding=False,
        qat=QATConfig(299, 300, 1),
    )
    camera = CameraConfig(1, 32)
    source = UcpeTransformer(config, camera, main_compress_gate=True).to(
        torch.bfloat16
    )
    with torch.no_grad():
        for value in source.parameters():
            value.fill_(0.125)
    expected = source.state_dict()
    save_file(expected, tmp_path / "model.safetensors")
    loaded = load_ucpe_transformer(
        tmp_path, config, camera, main_compress_gate=True, device="cpu"
    )
    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    active = replace(config, qat=QATConfig(300, 300, 1))
    with pytest.raises(ValueError, match="placed CUDA BF16"):
        load_ucpe_transformer(
            tmp_path, active, camera, main_compress_gate=True, device="cpu"
        )
