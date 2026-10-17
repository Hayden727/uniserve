"""Projection fusion reads current inputs through bound GEMMs during replay."""

import pytest
import torch

from uniserve.model import TextSize
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve_models.wan.config import TransformerConfig
from uniserve_models.wan.transformer import PluckerBlock

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_plucker_prefix_graph_replays_current_camera_and_restores_a():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA graph execution")
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
        block = PluckerBlock(config).to("cuda", torch.bfloat16)
        for parameter in block.parameters():
            parameter.normal_(std=0.1)
        hidden = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        camera = torch.randn(1, 8, 6, device="cuda", dtype=torch.bfloat16)
        scale_input = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
        scale = shift = torch.zeros(
            1, 1, 64, device="cuda", dtype=torch.bfloat16
        )
        angles = torch.randn(8, 32, device="cuda")
        rotary = angles.cos(), angles.sin()
    original_camera = camera.clone()
    stream = CUDAStream.external(torch.cuda.Stream())
    stream.wait(torch.cuda.current_stream())
    try:
        with (
            stream,
            ExecutionContext(
                block, attention="torch", matmul="torch", stream=stream
            ) as context,
        ):
            context.prepare(TextSize(8, 1))

            def run():
                value = block._projection_input(
                    hidden, scale, shift, camera, scale_input
                )
                return block._project_qk(
                    block.to_q, block.norm_q, value, rotary
                )

            expected_a = run().clone()
            with CUDAGraph(context=context) as graph:
                graph.capture(run)
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
                camera.add_(0.25)
                expected_b = run().clone()
                assert not torch.equal(expected_a, expected_b)
                torch.testing.assert_close(
                    graph.replay(), expected_b, rtol=0, atol=0
                )
                camera.copy_(original_camera)
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
    finally:
        stream.close()
