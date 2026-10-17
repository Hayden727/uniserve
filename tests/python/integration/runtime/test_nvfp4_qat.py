"""Replay native QAT references through the shared public capability."""

import pytest
import torch

from tests.python.integration.runtime.native_reference import (
    load_case,
    reference_root,
)
from uniserve.nn import Linear
from uniserve.nn.nvfp4_qat import NVFP4QATLinear
from uniserve.quantization.nvfp4 import quantize
from uniserve.quantization.nvfp4_qat import global_scale
from uniserve.runtime import CUDAGraph, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def replay(root, name):
    fields, case = load_case(root, name)
    assert case["kind"] == "nvfp4_qat"
    config = case["config"]
    source = Linear(
        config["width"],
        config["channels"],
        bias=config["bias"],
        device="meta",
        dtype=torch.bfloat16,
    )
    source.weight = torch.nn.Parameter(
        fields["weight"].cuda(), requires_grad=False
    )
    if config["bias"]:
        source.bias = torch.nn.Parameter(
            fields["bias"].cuda(), requires_grad=False
        )
    layer = NVFP4QATLinear.from_linear(source)
    storage = torch.empty(
        config["rows"], 2 * config["width"], device="cuda", dtype=torch.bfloat16
    )
    values = storage[:, ::2]
    values.copy_(fields["input"])
    scale = global_scale(values)
    packed, blocks = quantize(values.contiguous(), scale)
    for actual, key in (
        (layer.global_scale, "weight_global_scale"),
        (scale, "activation_global_scale"),
        (packed, "activation_fp4"),
        (blocks.reshape(-1), "activation_scales"),
        (layer.weight.buffers()["values"], "weight_fp4"),
        (layer.weight.buffers()["block_scale"], "weight_scales"),
    ):
        torch.testing.assert_close(
            actual.cpu().reshape(-1),
            fields[key].reshape(-1),
            rtol=0,
            atol=0,
            msg=lambda message, key=key: f"{name}: {key}: {message}",
        )
    expected = fields["output"].cuda()
    torch.testing.assert_close(layer(values), expected, rtol=0, atol=0)
    with ExecutionContext(layer, matmul="torch") as context:
        context.prepare(None)
        output = torch.empty(
            config["rows"],
            2 * config["channels"],
            device="cuda",
            dtype=torch.bfloat16,
        )[:, ::2]
        assert layer(values, out=output) is output
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert any(
            item["provider"] == "nvfp4_qat" for item in context.kernels()
        )
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: layer(values, out=output))
            values.copy_(fields["input_replay"])
            graph.replay()
            torch.testing.assert_close(
                output.cpu(), fields["output_replay"], rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "name",
    [
        "qat-m1-k256-n128-b0-none",
        "qat-m32-k256-n128-b1-none",
        "qat-m33-k256-n128-b1-none",
        "qat-m128-k256-n64-b1-none",
        "qat-m33-k256-n128-b1-activation",
        "qat-m33-k256-n128-b1-weight",
    ],
)
def test_native_qat_projection(name):
    replay(reference_root(), name)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("native QAT capture did not complete")
    for name in manifest["cases"]:
        replay(args.reference, name)
        print(json.dumps({"passed": name}), flush=True)
