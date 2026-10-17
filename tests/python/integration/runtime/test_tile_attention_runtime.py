"""Tile compression and fine attention use bound kernels and graph inputs."""

import pytest
import torch
from uniserve_kernels.attention import vsa_native

from uniserve.nn.attention.vsa import BlockAttention
from uniserve.nn.attention.vsa.tiles import TileAttention, TileLayout
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
@pytest.mark.parametrize("with_rotary", [False, True])
def test_tile_attention_obeys_selection_gate_and_mask_through_bound_provider(
    with_rotary,
):
    if not vsa_native.supported():
        pytest.skip("128-row tile attention requires SM100")
    device = torch.device("cuda")
    layout = TileLayout.from_grid((1, 1, 192), (1, 1, 128), device=device)
    q = torch.zeros(192, 2, 128, dtype=torch.bfloat16, device=device)
    v = torch.full_like(q, 2)
    v[128:] = 10
    gate = torch.zeros_like(q)
    module = TileAttention(BlockAttention(128**-0.5))
    rotary = (
        (
            torch.ones(192, 128, device=device),
            torch.zeros(192, 128, device=device),
        )
        if with_rotary
        else None
    )
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    stream.wait(torch.cuda.current_stream(device))
    try:
        with (
            stream,
            ExecutionContext(module, stream=stream, vsa="sm100") as context,
        ):
            context.prepare(None)
            # Equal pooled scores pick the first tile; the partial second
            # tile contributes one equally weighted mean to compression.
            output = module(q, q, v, gate, layout, keep=1, rotary=rotary)
            torch.testing.assert_close(
                output, torch.full_like(q, 2), rtol=0, atol=0
            )
            gate.fill_(2)
            output = module(q, q, v, gate, layout, keep=1, rotary=rotary)
            torch.testing.assert_close(
                output, torch.full_like(q, 14), rtol=0, atol=0
            )

            allowed = torch.ones(2, 2, dtype=torch.bool, device=device)
            replayed = torch.empty_like(q)
            with CUDAGraph(context=context) as graph:
                graph.capture(
                    lambda: replayed.copy_(
                        module(
                            q,
                            q,
                            v,
                            gate,
                            layout,
                            keep=1,
                            allowed=allowed,
                            rotary=rotary,
                        )
                    )
                )
                # Each query tile now sees only itself in both branches.
                # Mutable gate and mask inputs survive graph capture; the
                # negative raw gate cancels each constant fine output.
                gate.fill_(-1)
                allowed.copy_(torch.eye(2, dtype=torch.bool, device=device))
                graph.replay()
                torch.testing.assert_close(
                    replayed, torch.zeros_like(q), rtol=0, atol=0
                )
    finally:
        stream.close()


@torch.inference_mode()
def test_fused_tile_preparation_keeps_lanes_padding_and_live_graph_inputs():
    """Exact integer-valued rotations/means expose lane and padding mistakes."""
    from uniserve_kernels.attention.tile_rope_pool import tile_rope_pool

    if not vsa_native.supported():
        pytest.skip("the fused tile provider requires SM100")
    device = torch.device("cuda")
    layout = TileLayout.from_grid((1, 1, 192), (1, 1, 128), device=device)
    source = torch.full((256,), -1, dtype=torch.int64, device=device)
    source[layout.slots] = layout.permutation
    rows = (
        (torch.arange(192 * 2 * 128, device=device).remainder(8) - 4)
        .to(torch.bfloat16)
        .reshape(192, 2, 128)
    )
    inputs = tuple(rows.clone() + index for index in range(4))
    cos = torch.zeros(192, 128, device=device)
    cos[:, 1::2] = 1
    sin = 1 - cos  # Distinct even/odd table lanes must not be compacted.
    tiled = tuple(
        torch.empty(256, 2, 128, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    )
    pooled = tuple(
        torch.empty(2, 2, 128, device=device, dtype=torch.bfloat16)
        for _ in range(3)
    )

    def call():
        tile_rope_pool(
            *inputs,
            cos,
            sin,
            source,
            layout.valid_counts,
            tiled=tiled,
            pooled=pooled,
        )

    def check():
        expected = []
        for index, value in enumerate(inputs):
            if index < 2:
                real, imaginary = (
                    value.float().unflatten(-1, (-1, 2)).unbind(-1)
                )
                rotated = torch.stack((-imaginary, real), -1).flatten(-2)
                value = (
                    value.float() * cos[:, None] + rotated * sin[:, None]
                ).to(value.dtype)
            expected.append(layout.pack(value))
        for actual, wanted in zip(tiled, expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for actual, wanted in zip(pooled, expected[:3], strict=True):
            first = wanted[:128].float().mean(0)
            second = wanted[128:192].float().mean(0)
            means = torch.stack((first, second), dim=1).to(torch.bfloat16)
            torch.testing.assert_close(actual, means, rtol=0, atol=0)

    call()
    check()
    # The pure kernel has caller-owned buffers and uses the caller's stream.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    inputs[0].add_(1)
    cos.fill_(1)
    sin.zero_()
    graph.replay()
    check()
