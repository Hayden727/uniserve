"""One production UCPE block warms many static layouts before graph capture."""

from contextlib import ExitStack

import pytest
import torch

from uniserve.model import TextSize
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve_models.wan.config import TransformerConfig
from uniserve_models.wan.ucpe import CameraConfig, UcpeBlock

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_ucpe_prefix_twelve_layouts_then_graph_camera_a_b_a():
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
        inputs = []
        for tokens in range(48, 0, -4):
            hidden = torch.randn(
                1, tokens, 64, device="cuda", dtype=torch.bfloat16
            )
            context = torch.randn(1, 5, 64, device="cuda", dtype=torch.bfloat16)
            time = torch.randn(1, 6, 64, device="cuda", dtype=torch.bfloat16)
            pose = torch.eye(4, device="cuda", dtype=torch.bfloat16).repeat(
                1, tokens, 1, 1
            )
            angles = torch.randn(tokens, 32, device="cuda")
            inputs.append(
                (hidden, context, time, (angles.cos(), angles.sin()), pose)
            )

    stream = CUDAStream.external(torch.cuda.Stream())
    stream.wait(torch.cuda.current_stream())
    try:
        with (
            stream,
            ExecutionContext(
                block, attention="torch", matmul="torch", stream=stream
            ) as execution,
        ):
            # This context owns capacity for all twelve numerical layouts;
            # borrowed inputs and each captured graph remain caller-owned.
            execution.prepare(TextSize(48, 1))
            expected = []
            for args in inputs:
                value = block(*args).clone()
                assert torch.isfinite(value).all()
                expected.append(value)

            with (
                ExitStack() as graphs,
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                captured = []
                # Match worker ownership: finish every warmup before capture.
                for args in inputs:
                    graph = graphs.enter_context(CUDAGraph(context=execution))
                    graph.capture(lambda args=args: block(*args))
                    captured.append(graph)
                for index in (*range(len(inputs)), 0):
                    args, graph, expected_a = (
                        inputs[index],
                        captured[index],
                        expected[index],
                    )
                    pose = args[-1]
                    original = pose.clone()
                    torch.testing.assert_close(
                        graph.replay(), expected_a, rtol=0, atol=0
                    )
                    pose[:, pose.shape[1] // 2 :, 0, 3].add_(0.5)
                    expected_b = block(*args).clone()
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
