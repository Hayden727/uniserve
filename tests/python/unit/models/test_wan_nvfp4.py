"""Explicit Wan FP4 metadata and strict BF16 materialization requirements."""

import json
from dataclasses import asdict, replace

import pytest
import torch
from safetensors.torch import save_file

from uniserve_models.wan.config import (
    Config,
    NVFP4Config,
    TransformerConfig,
    VsaConfig,
    read_config,
)
from uniserve_models.wan.loading import load_transformer
from uniserve_models.wan.transformer import PluckerTransformer

pytestmark = pytest.mark.unit


def test_fp4_config_round_trip_keeps_epilogue_and_sparse_choices(tmp_path):
    config = Config(
        TransformerConfig(
            vsa=VsaConfig((4, 8, 8)),
            nvfp4=NVFP4Config(fold_gate_alpha=False),
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WanCameraPipeline"],
                **asdict(config),
            }
        )
    )
    assert read_config(tmp_path, None, sources=()) == config


@pytest.mark.parametrize("field", ["absorb_qk_alpha", "fold_gate_alpha"])
def test_alpha_omission_requires_uniformly_scaled_bias(field):
    options = {
        "fuse_bias": False,
        "absorb_qk_alpha": False,
        "fold_gate_alpha": False,
    }
    options[field] = True
    with pytest.raises(ValueError, match="fused NVFP4 bias"):
        NVFP4Config(**options)


def test_public_fp4_load_requires_real_cuda_conversion(tmp_path):
    config = TransformerConfig(
        num_layers=1,
        num_heads=2,
        head_dim=32,
        ffn_dim=64,
        text_dim=32,
        freq_dim=32,
        scale_bands=2,
        vsa=VsaConfig((4, 8, 8)),
    )
    source = PluckerTransformer(config).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    values = source.state_dict()
    save_file(values, tmp_path / "model.safetensors")
    quantized = replace(config, nvfp4=NVFP4Config())
    with pytest.raises(ValueError, match="placed CUDA BF16 weights"):
        load_transformer(tmp_path, quantized, device="cpu")
    # Completeness is established before conversion; the required learned
    # sparse gate cannot be synthesized or replaced by dense execution.
    values.pop("blocks.0.to_gate_compress.weight")
    save_file(values, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="checkpoint load mismatch"):
        load_transformer(tmp_path, quantized, device="cpu")


def _candidate_parity():
    """Use the public loader, then compare actual FP4 payloads and full DiTs."""
    import argparse
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from torch.nn.attention import SDPBackend, sdpa_kernel

    from tests.python.integration.runtime.native_reference import load_case
    from tests.python.unit.models.test_wan_vsa import _camera, _config
    from uniserve.model import TextSize
    from uniserve.nn.attention.vsa.tiles import TileLayout
    from uniserve.nn.nvfp4 import NVFP4Linear
    from uniserve.runtime.execution import ExecutionContext
    from uniserve_models.wan.loading import load_ucpe_transformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for backend in ("cublas", "cublaslt"):
        torch.backends.cuda.blas_workspace_size(0, backend=backend)
    for kind in ("plucker", "ucpe"):
        archive, case = load_case(args.reference, f"wan_nvfp4_{kind}")
        ucpe = kind == "ucpe"
        config = replace(_config(ucpe=ucpe), nvfp4=NVFP4Config())
        if case["config"]["nvfp4"] != asdict(config.nvfp4):
            raise ValueError(
                "native FP4 reference has a different numerical recipe"
            )
        weights = {
            name.removeprefix("weight."): value
            for name, value in archive.items()
            if name.startswith("weight.")
        }
        with TemporaryDirectory() as directory:
            save_file(weights, Path(directory) / "model.safetensors")
            loaded = (
                load_ucpe_transformer(
                    directory,
                    config,
                    _camera(),
                    main_compress_gate=True,
                    device="cuda",
                )
                if ucpe
                else load_transformer(directory, config, device="cuda")
            )
        model = loaded.model
        encoded = {
            name: layer
            for name, layer in model.named_modules()
            if isinstance(layer, NVFP4Linear)
        }
        if encoded.keys() != case["projection_epilogues"].keys():
            raise AssertionError(
                "native FP4 projection set differs, including checkpoint "
                "gates and edge blocks"
            )
        for name, layer in encoded.items():
            assert asdict(layer.epilogue) == case["projection_epilogues"][name]
            fields = layer.weight.buffers()
            torch.testing.assert_close(
                fields["values"].cpu(),
                archive[f"encoded.{name}.values"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                fields["block_scale"].flatten().cpu(),
                archive[f"encoded.{name}.scales"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                layer.alpha.cpu(),
                archive[f"encoded.{name}.alpha"],
                rtol=0,
                atol=0,
            )
        for index, alphas in case["gate_alphas"].items():
            block = model.blocks[int(index)]
            assert (
                block.output_alpha == alphas["output"]
                and block.ffn_alpha == alphas["ffn"]
            )
        for name, value in model.named_parameters():
            if name.removesuffix(".weight") not in encoded:
                torch.testing.assert_close(
                    value.cpu(), weights[name], rtol=0, atol=0
                )
        tensors = {
            name: value.to("cuda")
            for name, value in archive.items()
            if not name.startswith(("weight.", "encoded."))
        }
        sample, context = tensors["sample"], tensors["context"]
        rotary = model.rotary_embedding(sample.shape[2:], device="cuda")
        layout = TileLayout.from_grid(
            (5, 5, 9), config.vsa.tile_shape, device="cuda"
        )
        camera = (
            (tensors["viewmats"], rotary)
            if ucpe
            else (tensors["cam_emb"], tensors["scale_emb"], rotary)
        )
        kwargs = {"tiles": layout, **({"camera_tiles": layout} if ucpe else {})}
        taps = {}

        def capture(name):
            def hook(module, inputs, output):
                taps[name] = output[0] if isinstance(output, tuple) else output

            return hook

        handles = [
            model.get_submodule(name).register_forward_hook(capture(name))
            for name in case["tap_names"]
        ]
        try:
            with (
                ExecutionContext(
                    model, attention="torch", matmul="torch", vsa="sm100"
                ) as execution,
                sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                ),
            ):
                execution.prepare(TextSize(225, 1))
                for index in range(2):
                    expected = tensors[f"output.{index}"]
                    actual = model(
                        sample,
                        context,
                        tensors[f"time.{index}"],
                        *camera,
                        **kwargs,
                    )
                    left, right = (
                        actual.float().flatten(),
                        expected.float().flatten(),
                    )
                    nrmse = (
                        (left - right).square().mean().sqrt()
                        / right.square().mean().sqrt()
                    ).item()
                    cosine = torch.nn.functional.cosine_similarity(
                        left, right, dim=0
                    ).item()
                    print(
                        json.dumps(
                            {
                                "camera": kind,
                                "step": index,
                                "normalized_rms": nrmse,
                                "cosine": cosine,
                                "bitwise": torch.equal(actual, expected),
                                "max_abs": (left - right).abs().max().item(),
                                "verified_encoded_projections": len(encoded),
                            }
                        ),
                        flush=True,
                    )
                    if nrmse > 0.001 or cosine < 0.99999:
                        for name in case["tap_names"]:
                            native, candidate = (
                                tensors[f"tap.{index}.{name}"],
                                taps[name],
                            )
                            if not torch.equal(native, candidate):
                                print(
                                    json.dumps(
                                        {
                                            "first_differing_module": name,
                                            "max_abs": (
                                                native.float()
                                                - candidate.float()
                                            )
                                            .abs()
                                            .max()
                                            .item(),
                                        }
                                    ),
                                    flush=True,
                                )
                                break
                    assert (
                        torch.isfinite(actual).all()
                        and torch.isfinite(expected).all()
                    )
                    assert nrmse <= 0.001 and cosine >= 0.99999
        finally:
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    _candidate_parity()
