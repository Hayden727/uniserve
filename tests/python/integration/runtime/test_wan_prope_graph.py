"""Configured camera projection reads current poses during graph replay."""

import pytest
import torch

from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve_models.wan.ucpe import CameraAttention, CameraConfig

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_elementwise_camera_graph_reads_changed_inputs_and_restores_a():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA graph execution")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(937)
        model = CameraAttention(
            16, CameraConfig(2, 8, projection="elementwise")
        ).to("cuda", torch.bfloat16)
        model.out_proj.weight.normal_(std=0.2)
        hidden = torch.randn(1, 5, 16, device="cuda", dtype=torch.bfloat16)
    pose = torch.eye(4, device="cuda", dtype=torch.bfloat16).repeat(1, 5, 1, 1)
    pose[0, :, 0, 3] = torch.arange(5, device="cuda") * 0.25
    original_hidden, original_pose = hidden.clone(), pose.clone()
    stream = CUDAStream.external(torch.cuda.Stream())
    stream.wait(torch.cuda.current_stream())
    try:
        with (
            stream,
            ExecutionContext(
                model, attention="torch", matmul="torch", stream=stream
            ) as context,
        ):
            context.prepare(None)

            def run():
                return model(hidden, pose)

            expected_a = run().clone()
            with CUDAGraph(context=context) as graph:
                graph.capture(run)
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
                hidden.mul_(0.5)
                pose[0, 2:, 0, 3].add_(0.75)
                expected_b = run().clone()
                assert not torch.equal(expected_a, expected_b)
                torch.testing.assert_close(
                    graph.replay(), expected_b, rtol=0, atol=0
                )
                hidden.copy_(original_hidden)
                pose.copy_(original_pose)
                torch.testing.assert_close(
                    graph.replay(), expected_a, rtol=0, atol=0
                )
    finally:
        stream.close()
