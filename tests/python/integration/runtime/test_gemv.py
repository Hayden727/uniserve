"""Real native FP32 references through standalone, bound and captured Linear."""

import json

import pytest
import torch
import torch.nn.functional as F

from tests.python.integration.runtime.native_reference import (
    load_case,
    reference_root,
)
from uniserve.nn import Linear
from uniserve.runtime import CUDAGraph, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def replay(root, name):
    fields, case = load_case(root, name)
    assert case["kind"] == "fp32_gemv"
    n, k = fields["weight"].shape
    layer = Linear(
        k,
        n,
        bias=False,
        device="meta",
        dtype=torch.float32,
        gemv_reduction="strided16",
    )
    layer.weight = torch.nn.Parameter(
        fields["weight"].cuda(), requires_grad=False
    )
    values = torch.empty(1, 2 * k, device="cuda", dtype=torch.float32)[:, ::2]
    values.copy_(fields["input"])
    expected = fields["output"].cuda()
    torch.testing.assert_close(layer(values), expected, rtol=0, atol=0)
    with ExecutionContext(layer, matmul="torch") as context:
        context.prepare(None)
        output = torch.empty(1, 2 * n, device="cuda", dtype=torch.float32)[
            :, ::2
        ]
        assert layer(values, out=output) is output
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        # The declared recipe concerns one row. Compare2 and7 rows against
        # this environment's unchanged public dense backend, not another BLAS.
        for rows in (2, 7):
            matrix = fields["input"].cuda().repeat(rows, 1)
            matrix[1::2].copy_(fields["input_replay"].cuda())
            torch.testing.assert_close(
                layer(matrix), F.linear(matrix, layer.weight), rtol=0, atol=0
            )
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: layer(values, out=output))
            values.copy_(fields["input_replay"])
            graph.replay()
            torch.testing.assert_close(
                output.cpu(), fields["output_replay"], rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "name", ["gemv-n4096-k4096", "gemv-n4096-k10240", "gemv-n12288-k4096"]
)
def test_native_single_row_projection(name):
    replay(reference_root(), name)


@torch.inference_mode()
def test_partial_lanes_and_output_tiles_with_strided_weights_and_bias():
    # Integer products/sums stay exactly representable, giving an independent
    # FP64 oracle without depending on any cuBLAS reduction heuristic.
    layer = Linear(
        17, 13, device="cuda", dtype=torch.float32, gemv_reduction="strided16"
    )
    weight = torch.empty(13, 34, device="cuda")[:, ::2]
    weight.copy_(
        (torch.arange(13 * 17, device="cuda").reshape(13, 17) % 7 - 3).float()
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias.copy_(torch.arange(13, device="cuda"))
    values = torch.arange(17, device="cuda").float()[None]
    expected = F.linear(
        values.double(), weight.double(), layer.bias.double()
    ).float()
    with ExecutionContext(layer, matmul="torch") as context:
        context.prepare(None)
        torch.testing.assert_close(layer(values), expected, rtol=0, atol=0)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("native capture is incomplete")
    for name in manifest["cases"]:
        replay(args.reference, name)
        print(json.dumps({"passed": name}), flush=True)
    test_partial_lanes_and_output_tiles_with_strided_weights_and_bias()
