"""VSA transports all four projections over the complete global tile domain."""

import pytest
import torch
import torch.multiprocessing as mp
from uniserve_kernels.attention import vsa_native

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.distributed.tokens import TokenShard
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.nn.attention.vsa import BlockAttention
from uniserve.nn.attention.vsa.tiles import TileAttention, TileLayout
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def _run(rank, rendezvous, members):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=members,
        device=device,
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(
                ranks=tuple(reversed(range(members))),
                shape=(members,),
                axes=("tokens",),
                rank=rank,
            ),
            device=device,
        )
        group = mesh.get_group("tokens")
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        try:
            stream.wait(torch.cuda.current_stream(device))
            for heads in (40, 8):
                distributed = TileAttention(
                    BlockAttention(128**-0.5), num_heads=heads, head_dim=128
                )
                parallelize_(
                    distributed,
                    mesh,
                    attention=AttentionParallelConfig(heads=Ulysses("tokens")),
                )
                reference = TileAttention(
                    BlockAttention(128**-0.5), num_heads=heads, head_dim=128
                )
                root = torch.nn.ModuleList((distributed, reference))
                with (
                    stream,
                    ExecutionContext(
                        root, stream=stream, vsa="sm100"
                    ) as context,
                ):
                    context.prepare(None)
                    # The 645 real tokens divide neither SP2 nor SP8, and
                    # boundary tiles carry distinct valid-prefix lengths.
                    layout = TileLayout.from_grid(
                        (3, 5, 43), (1, 4, 32), device=device
                    )
                    partition = TokenShard(layout.num_tokens, group)
                    generator = torch.Generator().manual_seed(853 + heads)
                    full = tuple(
                        torch.randn(
                            layout.num_tokens, heads, 256, generator=generator
                        )
                        .bfloat16()
                        .to(device)[..., ::2]
                        for _ in range(4)
                    )
                    local = tuple(partition.local(value) for value in full)
                    # Unequal visible domains exercise mask intersection;
                    # keep is the host-computed maximum permitted budget.
                    cells = torch.arange(layout.num_tiles, device=device)
                    allowed = (cells[:, None] % 3 == cells[None, :] % 3) | (
                        cells[None, :] == 0
                    )
                    keep = layout.keep(
                        0.5, allowed_count=int(allowed.sum(-1).max().item())
                    )
                    expected = reference(
                        *full, layout, keep=keep, allowed=allowed
                    )
                    result = torch.empty_like(local[0])
                    returned = distributed(
                        *local, layout, keep=keep, allowed=allowed, out=result
                    )
                    assert returned is result
                    torch.testing.assert_close(
                        result, partition.local(expected), rtol=0, atol=0
                    )
                    with CUDAGraph(context=context) as graph:
                        graph.capture(
                            lambda: distributed(
                                *local,
                                layout,
                                keep=keep,
                                allowed=allowed,
                                out=result,
                            )
                        )
                        # Change each rank's raw gate and V after capture.
                        # If either projection missed the exchange, the
                        # complete-domain reference would expose it.
                        full[2].mul_(0.5)
                        full[3].neg_()
                        expected = reference(
                            *full, layout, keep=keep, allowed=allowed
                        )
                        graph.replay()
                        torch.testing.assert_close(
                            result, partition.local(expected), rtol=0, atol=0
                        )
        finally:
            stream.close()


@pytest.mark.parametrize("members", (2, 8))
def test_tile_ulysses_preserves_global_mask_heads_and_raw_gate(
    tmp_path, members
):
    if torch.cuda.device_count() < members or not vsa_native.supported():
        pytest.skip(f"VSA SP{members} requires {members} SM100 devices")
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(), members),
        nprocs=members,
        join=True,
    )
