"""Full fused Plucker block preserves current inputs through graph reuse."""

import pytest
import torch

from uniserve.model import TextSize
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve_models.wan.config import TransformerConfig
from uniserve_models.wan.transformer import PluckerBlock

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_full_plucker_block_graph_camera_a_b_a():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = TransformerConfig(
        num_layers=1,
        num_heads=2,
        head_dim=32,
        ffn_dim=96,
        compile_elementwise=True,
        fusion_recipe="plucker_prefix",
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        block = PluckerBlock(config).to("cuda", torch.bfloat16).eval()
        for parameter in block.parameters():
            parameter.normal_(std=0.1)
        hidden = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16)
        modulation = torch.randn(1, 6, 64, device="cuda", dtype=torch.bfloat16)
        camera = torch.randn(1, 8, 6, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(8, 32, device="cuda")
        rotary = angles.cos(), angles.sin()
    inputs = (hidden, context, modulation, *rotary, camera, scale)
    originals = [value.clone() for value in inputs]
    stream = CUDAStream.external(torch.cuda.Stream())
    stream.wait(torch.cuda.current_stream())
    try:
        with (
            stream,
            ExecutionContext(
                block, attention="torch", matmul="torch", stream=stream
            ) as execution,
        ):
            execution.prepare(TextSize(8, 1))

            def run():
                return block(hidden, context, modulation, rotary, camera, scale)

            expected_a = run().clone()
            assert torch.isfinite(expected_a).all()
            with CUDAGraph(context=execution) as graph:
                graph.capture(run)
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
                camera.add_(0.5)
                expected_b = run().clone()
                assert torch.isfinite(expected_b).all()
                assert not torch.equal(expected_a, expected_b)
                torch.testing.assert_close(
                    graph.replay(), expected_b, rtol=0, atol=0
                )
                camera.copy_(originals[-2])
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
            for value, original in zip(inputs, originals, strict=True):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
    finally:
        stream.close()
