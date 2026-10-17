"""UMT5 loading covers fused projections and tied embeddings exactly."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

pytestmark = pytest.mark.unit


@pytest.fixture
def text_checkpoint(tmp_path):
    from uniserve_models.wan.text import UMT5, TextConfig

    config = TextConfig(
        vocab_size=128,
        hidden_size=16,
        head_dim=8,
        ffn_dim=32,
        num_layers=1,
        num_heads=2,
    )
    torch.manual_seed(53)
    model = UMT5(config)
    expected = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    tensors = {}
    for name, value in expected.items():
        if name == "encoder.embed_tokens.weight":
            continue
        if name.endswith("qkv_proj.weight"):
            for branch, part in zip(
                ("q", "k", "v"), value.chunk(3), strict=True
            ):
                tensors[name.replace("qkv_proj", branch)] = part.contiguous()
        else:
            tensors[name] = value
    save_file(tensors, tmp_path / "model.safetensors")
    return tmp_path, config, expected, tensors


def test_text_loads_fused_qkv_and_shared_embedding(text_checkpoint):
    from uniserve_models.wan.loading import load_text_encoder

    path, config, expected, _ = text_checkpoint
    model = load_text_encoder(path, config, device="cpu").model
    assert model.shared.weight is model.encoder.embed_tokens.weight
    assert model.shared.weight.dtype == torch.float32
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@torch.inference_mode()
def test_text_capability_preserves_actual_sequence_lengths(text_checkpoint):
    from uniserve.model import TextEncoder
    from uniserve_models.wan.text import Encoder

    _, config, expected, _ = text_checkpoint
    encoder = Encoder(config)
    encoder.network.load_state_dict(expected)
    tokens = (torch.tensor([2]), torch.tensor([3, 4, 5]), torch.tensor([2]))
    results = encoder.encode(tokens)
    assert isinstance(encoder, TextEncoder)
    assert [value.shape for value in results] == [(1, 16), (3, 16), (1, 16)]
    torch.testing.assert_close(results[0], results[2], rtol=0, atol=0)
    torch.testing.assert_close(
        results[1], encoder.network(tokens[1][None])[0], rtol=0, atol=0
    )
    assert encoder.encode(()) == ()
    layout = encoder.output_layout(3, torch.bfloat16)["conditioning"]
    assert layout.dtype == torch.float32 and layout.shape == (3, 16)
    with pytest.raises(ValueError, match="1..512"):
        encoder.encode((torch.ones(513, dtype=torch.long),))


def test_catalog_loads_distinct_text_and_transformer_precisions(
    text_checkpoint,
):
    from uniserve.model import TextEncoder
    from uniserve_models import loading
    from uniserve_models.wan.config import TransformerConfig
    from uniserve_models.wan.transformer import PluckerTransformer
    from uniserve_worker.bootstrap.components import describe_components

    root, text_config, expected, _ = text_checkpoint
    text_root = root / "text_encoder"
    text_root.mkdir()
    (root / "model.safetensors").rename(text_root / "model.safetensors")
    transformer = TransformerConfig(
        num_layers=1,
        num_heads=2,
        head_dim=8,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
        scale_bands=2,
    )
    values = (
        PluckerTransformer(transformer).to(dtype=torch.bfloat16).state_dict()
    )
    save_file(values, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WanCameraPipeline"],
                "transformer": asdict(transformer),
                "text_encoder": asdict(text_config),
            }
        )
    )
    resolved = loading.read_config(root)
    model = loading.load_model(resolved, device="cpu").model
    assert isinstance(model.text_encoder, TextEncoder)
    assert next(model.transformer.parameters()).dtype == torch.bfloat16
    assert next(model.text_encoder.parameters()).dtype == torch.float32
    for name, value in model.text_encoder.network.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    calls = describe_components(model)
    assert calls["text_encoder"][0].entry_point.method == "encode"


@pytest.mark.parametrize("defect", ("missing_q", "extra", "bad_shape"))
def test_text_rejects_incomplete_or_extra_weights(text_checkpoint, defect):
    from uniserve_models.wan.loading import load_text_encoder

    path, config, _, tensors = text_checkpoint
    name = "encoder.block.0.layer.0.SelfAttention.q.weight"
    if defect == "missing_q":
        tensors.pop(name)
    elif defect == "extra":
        tensors["unused_branch.weight"] = torch.ones(1)
    else:
        tensors[name] = torch.ones(2, 2)
    save_file(tensors, path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError)):
        load_text_encoder(path, config, device="cpu")
