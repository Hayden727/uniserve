"""Fast checkpoint gates, typed sparse architecture and native DiT parity.

Execute this file with --reference in the candidate GPU environment to
compare a separately captured native model. The BF16 diagnostic uses actual
sparse attention; it does not qualify NVFP4 Fast inference.
"""

import json
from dataclasses import asdict, replace

import pytest
import torch
from safetensors.torch import save_file

from uniserve.nn.attention.vsa.tiles import TileLayout
from uniserve_models.wan.config import (
    Config,
    TransformerConfig,
    VsaConfig,
    read_config,
)
from uniserve_models.wan.denoiser import Denoiser
from uniserve_models.wan.inputs import LatentSize
from uniserve_models.wan.loading import load_transformer, load_ucpe_transformer
from uniserve_models.wan.transformer import PluckerTransformer
from uniserve_models.wan.ucpe import CameraConfig, UcpeConfig, UcpeTransformer

pytestmark = pytest.mark.unit


def _config(*, ucpe):
    return TransformerConfig(
        num_layers=2,
        num_heads=2,
        head_dim=128,
        ffn_dim=384,
        text_dim=32,
        freq_dim=16,
        scale_bands=2,
        use_scale_embedding=not ucpe,
        rms_reduction_order="ascending",
        patch_convolution="torch_unfold",
        vsa=VsaConfig((4, 4 if ucpe else 8, 8)),
    )


def _camera():
    return CameraConfig(1, 128, compress_gate=True, vsa=VsaConfig((4, 4, 8)))


@pytest.mark.parametrize("ucpe", [False, True])
def test_sparse_config_and_public_loader_preserve_every_gate(tmp_path, ucpe):
    config = _config(ucpe=ucpe)
    camera = _camera()
    torch.manual_seed(84)
    source = (
        UcpeTransformer(config, camera, main_compress_gate=True)
        if ucpe
        else PluckerTransformer(config)
    ).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(std=0.1)
    expected = source.state_dict()
    save_file(expected, tmp_path / "model.safetensors")
    declaration = Config(
        config, ucpe=UcpeConfig(camera, True) if ucpe else None
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WanCameraPipeline"],
                **asdict(declaration),
            }
        )
    )
    restored = read_config(tmp_path, None, sources=())
    assert restored == declaration
    loaded = (
        load_ucpe_transformer(
            tmp_path, config, camera, main_compress_gate=True, device="cpu"
        )
        if ucpe
        else load_transformer(tmp_path, config, device="cpu")
    )
    actual = loaded.model.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    # The original checkpoint's trained gate is required, not optional when
    # the architecture declares sparse attention. No strict=False loading.
    expected.pop("blocks.0.to_gate_compress.weight")
    save_file(expected, tmp_path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError)):
        if ucpe:
            load_ucpe_transformer(
                tmp_path, config, camera, main_compress_gate=True, device="cpu"
            )
        else:
            load_transformer(tmp_path, config, device="cpu")


def test_sparse_plucker_checkpoint_cannot_load_as_quality(tmp_path):
    config = _config(ucpe=False)
    source = PluckerTransformer(config).to(torch.bfloat16)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError)):
        load_transformer(tmp_path, replace(config, vsa=None), device="cpu")


def test_ucpe_sparse_architecture_requires_both_trained_gates():
    with pytest.raises(ValueError, match="camera compression gate"):
        CameraConfig(vsa=VsaConfig((4, 4, 8)))
    with pytest.raises(ValueError, match="main compression gate"):
        Config(_config(ucpe=True), ucpe=UcpeConfig(_camera(), False))
    with pytest.raises(ValueError, match="main compression gate"):
        UcpeTransformer(_config(ucpe=True), _camera())


@pytest.mark.parametrize("ucpe,keep", [(False, 30), (True, 59)])
def test_sparse_constants_use_unpadded_token_budget(ucpe, keep):
    config = _config(ucpe=ucpe)
    model = Denoiser(
        Config(config, ucpe=UcpeConfig(_camera(), True) if ucpe else None)
    )
    size = LatentSize(48, 60, 104, 512)
    declared = model.constant_buffers(size)
    constants = {
        name: torch.empty(buffer.shape, dtype=buffer.dtype)
        for name, buffer in declared.items()
    }
    model.prepare_constants(size, out=constants)
    grid = (48, 30, 52)
    for branch in ("main", "camera") if ucpe else ("main",):
        layout = TileLayout(
            grid,
            config.vsa.tile_shape,
            *(
                constants[f"{branch}_vsa_{field}"]
                for field in ("permutation", "slots", "restore", "valid_counts")
            ),
        )
        assert layout.keep(config.vsa.sparsity) == keep
        assert layout.valid_counts.sum().item() == 74880
        # Verify the public constants describe the full native raster even
        # at partial spatial tile boundaries.
        rows = torch.arange(74880)
        torch.testing.assert_close(
            layout.unpack(layout.pack(rows)), rows, rtol=0, atol=0
        )


def _candidate_parity():
    """Replay a hashed native archive without importing the native runtime."""
    import argparse
    from pathlib import Path

    from torch.nn.attention import SDPBackend, sdpa_kernel

    from tests.python.integration.runtime.native_reference import load_case
    from uniserve.model import TextSize
    from uniserve.runtime.execution import ExecutionContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    for backend in ("cublas", "cublaslt"):
        torch.backends.cuda.blas_workspace_size(0, backend=backend)
    torch.set_grad_enabled(False)
    for kind in ("plucker", "ucpe"):
        archive, case = load_case(args.reference, f"wan_vsa_{kind}")
        ucpe = kind == "ucpe"
        config = _config(ucpe=ucpe)
        if case["config"] != {
            "layers": 2,
            "heads": 2,
            "head_dim": 128,
            "ffn_dim": 384,
            "text_dim": 32,
            "freq_dim": 16,
            "scale_bands": 2,
            "grid": [5, 5, 9],
            "sparsity": 0.9,
            "tile": [4, 4 if ucpe else 8, 8],
        }:
            raise ValueError("native model architecture does not match replay")
        weights = {
            name.removeprefix("weight."): value
            for name, value in archive.items()
            if name.startswith("weight.")
        }
        tensors = {
            name: value.to("cuda")
            for name, value in archive.items()
            if not name.startswith("weight.")
        }
        model = (
            UcpeTransformer(config, _camera(), main_compress_gate=True)
            if ucpe
            else PluckerTransformer(config)
        ).to("cuda", torch.bfloat16)
        model.load_state_dict(weights, strict=True)
        sample, context = tensors["sample"], tensors["context"]
        rotary = model.rotary_embedding(sample.shape[2:], device="cuda")
        layout = TileLayout.from_grid(
            (5, 5, 9), config.vsa.tile_shape, device="cuda"
        )
        if ucpe:
            camera = (tensors["viewmats"], rotary)
            kwargs = {"tiles": layout, "camera_tiles": layout}
        else:
            camera = (tensors["cam_emb"], tensors["scale_emb"], rotary)
            kwargs = {"tiles": layout}
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
                for index, time in enumerate((1000.0, 625.0)):
                    timestep = tensors[f"time.{index}"]
                    expected = tensors[f"output.{index}"]
                    actual = model(sample, context, timestep, *camera, **kwargs)
                    delta = (actual.float() - expected.float()).flatten()
                    reference = expected.float().flatten()
                    nrmse = (
                        delta.square().mean().sqrt()
                        / reference.square().mean().sqrt()
                    ).item()
                    cosine = torch.nn.functional.cosine_similarity(
                        actual.float().flatten(), reference, dim=0
                    ).item()
                    print(
                        json.dumps(
                            {
                                "camera": kind,
                                "time": time,
                                "normalized_rms": nrmse,
                                "cosine": cosine,
                                "max_abs": delta.abs().max().item(),
                                "bitwise": torch.equal(actual, expected),
                            }
                        ),
                        flush=True,
                    )
                    if nrmse > 0.001 or cosine < 0.99999:
                        for name in case["tap_names"]:
                            native = tensors[f"tap.{index}.{name}"]
                            candidate = taps[name]
                            if not torch.equal(candidate, native):
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
                                            "native_stride": list(
                                                native.stride()
                                            ),
                                            "candidate_stride": list(
                                                candidate.stride()
                                            ),
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
