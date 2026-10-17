"""Global grid visibility and replicated text KV survive Ulysses ownership."""

from contextlib import nullcontext

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.distributed.tokens import TokenShard
from uniserve.nn.attention import (
    Attention,
    AttentionBatch,
    AttentionParallelConfig,
    DenseInput,
    Ulysses,
)
from uniserve.nn.attention.grid import GridAttention, GridLayout
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)

pytestmark = pytest.mark.integration


def _run(rank, rendezvous, gpu, members):
    device = torch.device("cuda", rank) if gpu else torch.device("cpu")
    if gpu:
        torch.cuda.set_device(device)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=members,
        device=device,
        init_method=rendezvous,
    ) as owner:
        # Nontrivial rank order verifies that canonical cells are global
        # mathematical positions, not backend rank numbers.
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
        main_heads, camera_heads, width = (40, 8, 128) if gpu else (8, 4, 8)
        root = torch.nn.ModuleDict(
            {
                "main": GridAttention(main_heads, width),
                "camera": GridAttention(camera_heads, width),
                "text": Attention(
                    main_heads, main_heads, width, replicated_kv=True
                ),
            }
        )
        parallelize_(
            root,
            mesh,
            attention=AttentionParallelConfig(
                heads=Ulysses("tokens"),
            ),
        )
        stream = (
            CUDAStream.external(torch.cuda.Stream(device=device))
            if gpu
            else None
        )
        try:
            if stream is not None:
                stream.wait(torch.cuda.current_stream(device))
            with (
                stream if stream is not None else nullcontext(),
                ExecutionContext(
                    root,
                    stream=stream,
                    grid="flash_attn_4" if gpu else "torch",
                    attention="flash_attn_4" if gpu else "torch",
                ) as context,
            ):
                context.prepare(None)
                for views, columns in ((2, 3), (32, 16)):
                    camera = torch.arange(views).repeat(columns)
                    column = torch.arange(columns).repeat_interleave(views)
                    layout = GridLayout.from_cells(
                        camera, column, 1, device=device
                    )
                    partition = TokenShard(layout.num_tokens, group)
                    mask = (
                        (
                            (camera[:, None] == camera[None, :])
                            | (column[:, None] == column[None, :])
                            | (camera[None, :] == 0)
                        )
                        .float()
                        .to(device)
                    )
                    weights = mask / mask.sum(-1, keepdim=True)
                    # Distinct per-head and per-cell values catch head mixing
                    # and inverse-exchange ordering, including empty shards.
                    cell_values = torch.arange(
                        layout.num_tokens, device=device
                    ).float()
                    cell_values = cell_values / layout.num_tokens
                    for name, heads in (
                        ("main", main_heads),
                        ("camera", camera_heads),
                    ):
                        q = torch.zeros(
                            partition.count,
                            heads,
                            width,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        values = (
                            cell_values[:, None, None]
                            + torch.arange(heads, device=device)[None, :, None]
                            / heads
                        )
                        values = (
                            values.expand(-1, -1, width).bfloat16().contiguous()
                        )
                        v = partition.local(values).contiguous()
                        expected = torch.einsum(
                            "qk,khd->qhd", weights, values.float()
                        ).bfloat16()
                        result = root[name](q, q, v, layout)
                        torch.testing.assert_close(
                            result,
                            partition.local(expected),
                            rtol=0,
                            atol=4 * torch.finfo(torch.bfloat16).eps,
                        )
                        if gpu:
                            with CUDAGraph(context=context) as graph:
                                graph.capture(
                                    lambda: root[name](
                                        q, q, v, layout, out=result
                                    )
                                )
                                v.fill_(3)
                                graph.replay()
                                torch.testing.assert_close(
                                    result,
                                    torch.full_like(q, 3),
                                    rtol=0,
                                    atol=0,
                                )
                    # The query interval is local; text keys are complete on
                    # every rank. Self-attention's exchange must not touch them.
                    query = torch.zeros(
                        partition.count,
                        main_heads,
                        width,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    keys = torch.zeros(
                        7,
                        main_heads,
                        width,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    values = torch.arange(
                        7, device=device, dtype=torch.bfloat16
                    )
                    values = values[:, None, None].expand_as(keys).contiguous()
                    result = root["text"](
                        query,
                        keys,
                        values,
                        AttentionBatch.single(DenseInput(False, None)),
                    )
                    torch.testing.assert_close(
                        result, torch.full_like(query, 3), rtol=0, atol=0
                    )
        finally:
            if stream is not None:
                stream.close()


def test_grid_ulysses_keeps_global_order_and_replicated_conditioning(tmp_path):
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(), False, 4),
        nprocs=4,
        join=True,
    )


@pytest.mark.gpu
def test_grid_sp8_fa4_and_capture_keep_main_and_camera_heads(tmp_path):
    if torch.cuda.device_count() < 8:
        pytest.skip("grid SP8 requires eight CUDA devices")
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(), True, 8),
        nprocs=8,
        join=True,
    )
