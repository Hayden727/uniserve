"""Public rectangular camera computation and per-camera video contracts."""

from dataclasses import replace

import pytest
import torch

from uniserve.diffusion import DenoisingStep
from uniserve.media import image, video
from uniserve.model import LatentInput
from uniserve.model.camera import CameraCondition, CameraGrid, CameraVideo
from uniserve.runtime import ExecutionContext
from uniserve_models.wan.config import Config, TransformerConfig, VsaConfig
from uniserve_models.wan.model import Model
from uniserve_models.wan.stdm import GridSize
from uniserve_models.wan.stdm_video import Decoder, Postprocessor
from uniserve_models.wan.ucpe import CameraConfig, UcpeConfig
from uniserve_models.wan.vae import VAEConfig
from uniserve_worker.bootstrap.report import frame_clips, video_denoiser_info

pytestmark = pytest.mark.unit


def _config(grid, sparse=False):
    return Config(
        TransformerConfig(
            num_layers=1,
            num_heads=2,
            head_dim=128 if sparse else 8,
            ffn_dim=24,
            in_channels=4,
            out_channels=4,
            text_dim=12,
            freq_dim=8,
            use_scale_embedding=False,
            vsa=VsaConfig((1, 32, 8), 0.5) if sparse else None,
        ),
        ucpe=UcpeConfig(
            CameraConfig(1, 128 if sparse else 8, False),
            main_compress_gate=sparse,
        ),
        grid=grid,
    )


def _buffers(fields, device="cpu"):
    return {
        name: torch.empty(field.shape, dtype=field.dtype, device=device)
        for name, field in fields.items()
    }


def _condition(size):
    # Explicit native time-major fixture, independent of model table helpers.
    assert size.grid.order == "time_major"
    views, columns = size.grid.views, size.grid.columns
    poses = torch.eye(4).repeat(size.frames, 1, 1)
    poses[:, 0, 3] = torch.arange(views).repeat(columns).float() / 8
    poses[:, 1, 3] = torch.arange(columns).repeat_interleave(views).float() / 8
    return CameraCondition(
        poses,
        torch.tensor([[16, 0, 8], [0, 16, 8], [0, 0, 1]]).float(),
        tuple(range(views)) + tuple(range(views, views * columns, views)),
        "prepared_camera_to_world",
        clean_latents=torch.full(
            (4, size.frames, size.height, size.width), 0.25
        ),
        cell_time_index=torch.arange(columns).repeat_interleave(views),
        grid=size.grid,
    )


def test_full_grid_is_independent_from_the_output_timeline_and_capability():
    grid = CameraGrid(32, 16, "time_major")
    with torch.device("meta"):
        denoiser = Model(_config(grid)).transformer
    size = denoiser.make_camera_size(
        grid, 1824, 1, canvas=image.Config(480, 832)
    )
    assert denoiser.noise_shape("video", size) == (1, 4, 512, 60, 104)
    assert denoiser.camera_buffers(size)["viewmats"].shape == (1, 798720, 4, 4)
    assert denoiser.output_layout(size)["video"].shape == (1, 4, 512, 60, 104)
    assert denoiser.camera_video == CameraVideo(grid, 57, 15)
    assert denoiser.legal_frame_count(1) == 1824
    assert denoiser.layout_size(size).num_text_tokens == 512
    clips = frame_clips(denoiser, None)
    assert (clips.min_frames, clips.clip_frames) == (57, 57)
    assert (clips.clip_latent_frames, clips.tail_latent_frames) == (15, 0)
    info = video_denoiser_info(
        denoiser, frame_rate=8, audio_sample_rate=0, frames=clips
    ).to_mapping()
    assert info["camera_layout"] == {
        "views": 32,
        "columns": 16,
        "order": "time_major",
        "frames_per_view": 57,
        "latent_frames_per_view": 15,
    }
    for wrong_grid, frames in (
        (grid, 1664),
        (CameraGrid(16, 32, "time_major"), 1824),
    ):
        with pytest.raises(ValueError, match="grid and output timeline"):
            denoiser.make_camera_size(wrong_grid, frames, 1, canvas=size.canvas)


def test_full_grid_preparation_keeps_exact_47_anchors_and_column_phases():
    grid = CameraGrid(32, 16, "time_major")
    denoiser = Model(_config(grid)).transformer
    size = GridSize(1824, image.Config(16, 16), 1, grid)
    condition = _condition(size)
    before = condition.poses.clone()
    out = _buffers(denoiser.camera_buffers(size))
    denoiser.prepare_camera(size, condition, out=out)
    assert out["clean_mask"].sum().item() == 47
    assert out["clean_mask"].flatten().nonzero().flatten().tolist() == list(
        condition.clean_cells
    )
    torch.testing.assert_close(condition.poses, before, rtol=0, atol=0)
    torch.testing.assert_close(
        out["camera_cos"][0], out["camera_cos"][31], rtol=0, atol=0
    )
    assert not torch.equal(out["camera_cos"][0], out["camera_cos"][32])
    for broken in (
        replace(condition, clean_cells=(0,)),
        replace(condition, cell_time_index=torch.arange(512)),
        replace(condition, clean_latents=None),
    ):
        with pytest.raises(ValueError):
            denoiser.prepare_camera(size, broken, out=out)


@torch.inference_mode()
@pytest.mark.parametrize(
    "sparse", [False, pytest.param(True, marks=pytest.mark.gpu)]
)
def test_grid_dmd_reuses_request_storage_and_pins_clean_values(sparse):
    if sparse and not torch.cuda.is_available():
        pytest.skip("the native VSA provider requires CUDA")
    device = torch.device("cuda", 0) if sparse else torch.device("cpu")
    grid = CameraGrid(3, 3, "time_major")
    torch.manual_seed(321)
    denoiser = Model(_config(grid, sparse)).transformer.bfloat16()
    for name, parameter in denoiser.named_parameters():
        parameter.normal_(std=0.1)
        if "norm" in name and name.endswith("weight"):
            parameter.add_(1)
    denoiser.to(device)
    size = GridSize(15, image.Config(16, 16), 1, grid)
    state = _buffers(denoiser.camera_buffers(size), device)
    constants = _buffers(denoiser.constant_buffers(size), device)
    denoiser.prepare_constants(size, out=constants)
    schedules = denoiser.make_schedules(device=device)
    schedule = schedules["video"]
    generator = torch.Generator().manual_seed(19)
    draws = [
        torch.randn(denoiser.noise_shape("video", size), generator=generator)
        .bfloat16()
        .to(device)
        for _ in range(4)
    ]
    context = torch.randn(512, 12).bfloat16().to(device)
    first = _condition(size)
    second = replace(first, poses=first.poses.flip(0))
    results = []
    with ExecutionContext(
        denoiser,
        grid="flash_attn_4" if sparse else "torch",
        attention="torch_fused" if sparse else "torch",
        vsa="sm100" if sparse else "auto",
    ) as runtime:
        runtime.prepare(size)
        for condition in (first, second, first):
            denoiser.prepare_camera(size, condition, out=state)
            sample = draws[0].clone()
            for index in range(4):
                inputs = denoiser.bind_camera_inputs(
                    latents={
                        "video": (
                            LatentInput(sample, schedule.timesteps[index]),
                        )
                    },
                    sizes=(size,),
                    step=schedule.step(index),
                    text_features=(context,),
                    state=state,
                    noise=(draws[min(index + 1, 3)],),
                )
                DenoisingStep(denoiser, inputs, schedules, {}, constants, {})()
                mask = state["clean_mask"].expand_as(sample)
                torch.testing.assert_close(
                    sample[mask], state["clean"][mask], rtol=0, atol=0
                )
            results.append(sample.clone())
    assert torch.equal(results[0], results[2])
    assert not torch.equal(results[0], results[1])


@pytest.mark.parametrize("order", ["view_major", "time_major"])
def test_camera_decoder_selects_head_and_middle_without_resampling(order):
    layout = CameraVideo(CameraGrid(32, 16, order), 57, 15)
    with torch.device("meta"):
        decoder = Decoder(VAEConfig(base_dim=8, num_res_blocks=1), layout)
    size = video.Config(1824, image.Config(16, 24))
    native = torch.arange(512).reshape(1, 1, 512, 1, 1)
    native = native.expand(1, 16, 512, 2, 3).bfloat16().contiguous()
    original = native.clone()
    assert len(decoder.clips(size.num_frames)) == 32
    assert decoder.output_layout(size)["video"].shape == (32, 1, 3, 57, 16, 24)
    for view, clip in enumerate(decoder.clips(size.num_frames)):
        out = torch.empty((1, 16, 15, 2, 3))
        decoder.unpack_latents(native, clip, size, out=out)
        expected = (
            torch.arange(view * 16, view * 16 + 15)
            if order == "view_major"
            else torch.arange(view, 480, 32)
        )
        torch.testing.assert_close(
            out, native[:, :, expected].float(), rtol=0, atol=0
        )
    torch.testing.assert_close(native, original, rtol=0, atol=0)
    processor = Postprocessor(layout)
    assert processor.frame_rate == 8
    assert processor.workspace_buffers(size)["rgb_frames"].shape == (
        57,
        16,
        24,
        3,
    )
    assert processor.output_layout(size)["video"].shape == (1824, 16, 24, 3)
