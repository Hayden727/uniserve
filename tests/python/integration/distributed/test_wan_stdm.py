"""Real process groups preserve whole-cell state and global Wan outputs."""

from contextlib import nullcontext

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.diffusion import DenoisingStep
from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.distributed.tokens import TokenShard
from uniserve.media import image
from uniserve.model import LatentInput
from uniserve.model.camera import CameraCondition, CameraGrid
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)
from uniserve_models.wan.config import Config, TransformerConfig
from uniserve_models.wan.model import Model
from uniserve_models.wan.stdm import GridSize
from uniserve_models.wan.ucpe import CameraConfig, UcpeConfig

pytestmark = pytest.mark.integration


def _allocate(fields, device):
    return {
        name: torch.empty(field.shape, dtype=field.dtype, device=device)
        for name, field in fields.items()
    }


@torch.inference_mode()
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
        mesh = owner.bind(
            DeviceMesh(
                ranks=tuple(reversed(range(members))),
                shape=(members,),
                axes=("tokens",),
                rank=rank,
            ),
            device=device,
        )
        grid = (
            CameraGrid(32, 16, "time_major")
            if gpu
            else CameraGrid(4, 3, "time_major")
        )
        main, camera, width = (40, 8, 128) if gpu else (4, 2, 8)
        config = Config(
            TransformerConfig(
                num_layers=1,
                num_heads=main,
                head_dim=width,
                ffn_dim=64,
                in_channels=4,
                out_channels=4,
                text_dim=12,
                freq_dim=8,
                use_scale_embedding=False,
            ),
            ucpe=UcpeConfig(CameraConfig(camera, width, False)),
            grid=grid,
        )
        torch.manual_seed(139)
        reference = Model(config).transformer.bfloat16()
        for name, value in reference.named_parameters():
            value.normal_(std=0.02)
            if "norm" in name and name.endswith("weight"):
                value.add_(1)
        distributed = Model(config).transformer.bfloat16()
        distributed.load_state_dict(reference.state_dict(), strict=True)
        reference.to(device)
        distributed.to(device)
        parallelize_(
            distributed,
            mesh,
            attention=AttentionParallelConfig(heads=Ulysses("tokens")),
        )
        partition = TokenShard(grid.num_cells, mesh.get_group("tokens"))
        size = GridSize(
            distributed.camera_video.num_frames, image.Config(16, 16), 1, grid
        )
        poses = torch.eye(4).repeat(grid.num_cells, 1, 1)
        poses[:, 0, 3] = torch.arange(grid.num_cells).float() / grid.num_cells
        clean_cells = tuple(range(grid.views)) + tuple(
            range(grid.views, grid.num_cells, grid.views)
        )
        condition = CameraCondition(
            poses,
            torch.tensor([[16, 0, 8], [0, 16, 8], [0, 0, 1]]).float(),
            clean_cells,
            "prepared_camera_to_world",
            clean_latents=torch.full((4, grid.num_cells, 2, 2), 0.25),
            cell_time_index=torch.arange(grid.columns).repeat_interleave(
                grid.views
            ),
            grid=grid,
        )
        generator = torch.Generator().manual_seed(91)
        noise, renoise = [
            torch.randn((1, 4, grid.num_cells, 2, 2), generator=generator)
            .bfloat16()
            .to(device)
            for _ in range(2)
        ]
        features = (
            torch.randn(512, 12, generator=generator).bfloat16().to(device)
        )
        stream = (
            CUDAStream.external(torch.cuda.Stream(device=device))
            if gpu
            else None
        )
        try:
            if stream is not None:
                stream.wait(torch.cuda.current_stream(device))
            root = torch.nn.ModuleList((reference, distributed))
            with (
                stream if stream is not None else nullcontext(),
                ExecutionContext(
                    root,
                    stream=stream,
                    grid="flash_attn_4" if gpu else "torch",
                    attention="torch_fused" if gpu else "torch",
                ) as runtime,
            ):
                runtime.prepare(None)
                steps, samples, states = [], [], []
                for model in (reference, distributed):
                    state = _allocate(model.camera_buffers(size), device)
                    model.prepare_camera(size, condition, out=state)
                    constants = _allocate(model.constant_buffers(size), device)
                    model.prepare_constants(size, out=constants)
                    sample = (
                        noise.clone()
                        if model is reference
                        else partition.local(noise, dim=2).clone()
                    )
                    schedules = model.make_schedules(device=device)
                    inputs = model.bind_camera_inputs(
                        latents={
                            "video": (
                                LatentInput(
                                    sample, schedules["video"].timesteps[0]
                                ),
                            )
                        },
                        sizes=(size,),
                        step=schedules["video"].step(0),
                        text_features=(features,),
                        state=state,
                        noise=(renoise,),
                    )
                    steps.append(
                        DenoisingStep(
                            model, inputs, schedules, {}, constants, {}
                        )
                    )
                    samples.append(sample)
                    states.append(state)
                for step in steps:
                    step()
                expected, actual = samples
                layout = distributed.output_layout(size)["video"]
                assert layout.shape == tuple(noise.shape)
                assert layout.local_slice[2] == partition.token_slice
                torch.testing.assert_close(
                    actual,
                    partition.local(expected, dim=2),
                    rtol=0,
                    atol=4 * torch.finfo(torch.bfloat16).eps,
                )
                gathered = mesh.get_group("tokens").all_gather(actual, dim=2)
                torch.testing.assert_close(
                    gathered,
                    expected,
                    rtol=0,
                    atol=4 * torch.finfo(torch.bfloat16).eps,
                )
                mask = states[1]["clean_mask"].expand_as(actual)
                assert torch.equal(actual[mask], states[1]["clean"][mask])
                if gpu:
                    with CUDAGraph(context=runtime) as graph:
                        graph.capture(steps[1])
                        actual.copy_(partition.local(noise, dim=2))
                        graph.replay()
                        torch.testing.assert_close(
                            actual,
                            partition.local(expected, dim=2),
                            rtol=0,
                            atol=4 * torch.finfo(torch.bfloat16).eps,
                        )
        finally:
            if stream is not None:
                stream.close()


def test_grid_model_sp2_preserves_cells_noise_positions_and_output_ownership(
    tmp_path,
):
    mp.spawn(
        _run, args=((tmp_path / "rdzv").as_uri(), False, 2), nprocs=2, join=True
    )


@pytest.mark.gpu
def test_full_cell_domain_sp8_keeps_40_main_8_camera_heads_and_replays(
    tmp_path,
):
    if torch.cuda.device_count() < 8:
        pytest.skip("Wan grid SP8 requires eight CUDA devices")
    mp.spawn(
        _run, args=((tmp_path / "rdzv").as_uri(), True, 8), nprocs=8, join=True
    )
