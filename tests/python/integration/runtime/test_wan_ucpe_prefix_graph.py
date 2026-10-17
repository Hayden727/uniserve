"""A complete fused UCPE block reads current cameras through graph replay."""

import pytest
import torch

from uniserve.model import TextSize
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve_models.wan.config import TransformerConfig
from uniserve_models.wan.ucpe import CameraConfig, UcpeBlock

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_ucpe_prefix_block_graph_camera_alternation():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA graph execution")
    config = TransformerConfig(
        num_layers=1,
        num_heads=2,
        head_dim=32,
        ffn_dim=96,
        compile_elementwise=True,
        fusion_recipe="ucpe_prefix",
        use_scale_embedding=False,
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(71)
        block = UcpeBlock(
            config,
            CameraConfig(num_heads=1, head_dim=32, projection="elementwise"),
        ).to("cuda", torch.bfloat16)
        for parameter in block.parameters():
            parameter.normal_(std=0.15)
        hidden = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 5, 64, device="cuda", dtype=torch.bfloat16)
        time = torch.randn(1, 6, 64, device="cuda", dtype=torch.bfloat16)
        pose = torch.eye(4, device="cuda", dtype=torch.bfloat16).repeat(
            1, 8, 1, 1
        )
        angles = torch.randn(8, 32, device="cuda")
        rotary = angles.cos(), angles.sin()
    original = pose.clone()
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
                return block(hidden, context, time, rotary, pose)

            expected_a = run().clone()
            assert torch.isfinite(expected_a).all()
            with CUDAGraph(context=execution) as graph:
                graph.capture(run)
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
                pose[:, 4:, 0, 3].add_(0.5)
                expected_b = run().clone()
                assert torch.isfinite(expected_b).all()
                assert not torch.equal(expected_a, expected_b)
                torch.testing.assert_close(
                    graph.replay(), expected_b, rtol=0, atol=0
                )
                pose.copy_(original)
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
    finally:
        stream.close()
