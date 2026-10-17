"""UCPE public camera preparation and shared four-step denoising behavior."""

from dataclasses import replace

import pytest
import torch

from uniserve.diffusion import DenoisingStep
from uniserve.media import image
from uniserve.model import LatentInput
from uniserve.model.camera import CameraCondition
from uniserve_models.wan.config import Config, TransformerConfig
from uniserve_models.wan.model import Model
from uniserve_models.wan.serving import VideoSize
from uniserve_models.wan.ucpe import CameraConfig, UcpeConfig, UcpeTransformer

pytestmark = pytest.mark.unit


def _config(spatial_rope=False):
    return Config(
        TransformerConfig(
            num_layers=1,
            num_heads=2,
            head_dim=8,
            ffn_dim=24,
            in_channels=4,
            out_channels=4,
            text_dim=12,
            freq_dim=8,
            use_scale_embedding=False,
        ),
        ucpe=UcpeConfig(
            CameraConfig(1, 8, spatial_rope, compress_gate=True),
            main_compress_gate=True,
        ),
    )


def _buffers(fields):
    return {
        name: torch.empty(field.shape, dtype=field.dtype)
        for name, field in fields.items()
    }


def _condition(size):
    poses = torch.eye(4).repeat(size.frames, 1, 1)
    poses[:, 0, 3] = torch.arange(size.frames).float()
    return CameraCondition(
        poses=poses,
        intrinsics=torch.tensor([[16, 0, 8], [0, 16, 8], [0, 0, 1]]).float(),
        clean_cells=(0, size.frames - 1),
        pose_space="prepared_camera_to_world",
        clean_latents=torch.full(
            (4, size.frames, size.height, size.width), 0.25
        ),
    )


def test_ucpe_preparation_preserves_metric_poses_and_request_timestamps():
    denoiser = Model(_config()).transformer
    size = VideoSize(156, image.Config(16, 16), 5)
    raw = torch.eye(4).repeat(160, 1, 1)
    centers = torch.tensor([[2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0]])
    raw[:, :3, 3] = -centers.repeat_interleave(40, dim=0)
    condition = CameraCondition(
        poses=raw,
        intrinsics=torch.tensor([[16, 0, 8], [0, 16, 8], [0, 0, 1]]).float(),
        clean_cells=(0, 24),
        pose_space="raw_world_to_camera",
        cell_time_index=torch.zeros(size.frames, dtype=torch.int64),
    )
    canonical = denoiser.output_camera(size, condition)
    assert canonical.pose_space == "prepared_camera_to_world"
    assert canonical.poses[24, 0, 3] == -4
    state = _buffers(denoiser.camera_buffers(size))
    denoiser.prepare_camera(size, condition, out=state)
    expected = torch.eye(4, dtype=torch.bfloat16)
    expected[0, 3] = 4
    torch.testing.assert_close(
        state["viewmats"][0, 24], expected, rtol=0, atol=0
    )
    assert state["clean_mask"].flatten().nonzero().flatten().tolist() == [0, 24]
    assert denoiser.input_frame_indices(size, condition, 160) == tuple(
        index for index in range(160) if index not in (39, 79, 119, 159)
    )
    same_time = state["camera_cos"].clone()
    torch.testing.assert_close(
        same_time, same_time[:1].expand_as(same_time), rtol=0, atol=0
    )
    denoiser.prepare_camera(
        size, replace(canonical, cell_time_index=None), out=state
    )
    assert not torch.equal(state["camera_cos"], same_time)
    # Already prepared metric poses are reused without a second inverse or
    # orbit normalization when the runtime prepares the device-side fields.
    torch.testing.assert_close(
        state["viewmats"][0, 24], expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("spatial_rope", [False, True])
@torch.inference_mode()
def test_ucpe_serving_matches_explicit_four_step_sampling_and_repeated_cameras(
    spatial_rope,
):
    config = _config(spatial_rope)
    torch.manual_seed(321)
    denoiser = Model(config).transformer.bfloat16()
    for name, parameter in denoiser.named_parameters():
        parameter.normal_(std=0.2)
        if "norm" in name and name.endswith("weight"):
            parameter.add_(1)
    direct = UcpeTransformer(
        config.transformer, config.ucpe.camera, main_compress_gate=True
    ).bfloat16()
    direct.load_state_dict(denoiser.network.state_dict(), strict=True)
    size = VideoSize(13, image.Config(16, 16), 5)
    state = _buffers(denoiser.camera_buffers(size))
    constants = _buffers(denoiser.constant_buffers(size))
    denoiser.prepare_constants(size, out=constants)
    schedules = denoiser.make_schedules(device="cpu")
    schedule = schedules["video"]
    generator = torch.Generator().manual_seed(19)
    draws = [
        torch.randn(
            denoiser.noise_shape("video", size), generator=generator
        ).bfloat16()
        for _ in range(4)
    ]
    context = torch.randn(5, 12).bfloat16()
    context = torch.cat((context, torch.zeros(507, 12, dtype=context.dtype)))
    first = _condition(size)
    second = replace(
        first,
        poses=first.poses.flip(0),
        cell_time_index=torch.tensor([0, 0, 1, 1]),
    )
    outputs = []
    for condition in (first, second, first):
        denoiser.prepare_camera(size, condition, out=state)
        reference, sample = draws[0].clone(), draws[0].clone()
        rotary = direct.rotary_embedding(
            sample.shape[2:],
            device="cpu",
            cell_time_index=condition.cell_time_index,
        )
        camera_rotary = direct.camera_rotary_embedding(
            sample.shape[2:], device="cpu", dtype=torch.bfloat16
        )
        mask = state["clean_mask"]
        # The direct oracle follows native orbit VAE storage, independently
        # of the runtime's contiguous request buffers. Native clean pinning
        # preserves channels-last input to patch embedding and thus camera
        # projection bias rounding. CPU convolution backends can hide this
        # distinction by returning the same output layout for both inputs.
        clean = state["clean"].contiguous(memory_format=torch.channels_last_3d)
        for index in range(4):
            timestep = schedule.timesteps[index]
            noise = draws[index + 1] if index < 3 else draws[-1]
            inputs = denoiser.bind_camera_inputs(
                latents={"video": (LatentInput(sample, timestep),)},
                sizes=(size,),
                step=schedule.step(index),
                text_features=(context,),
                state=state,
                noise=(noise,),
            )
            DenoisingStep(denoiser, inputs, schedules, {}, constants, {})()
            pinned = torch.where(mask, clean, reference)
            assert pinned.is_contiguous(memory_format=torch.channels_last_3d)
            velocity = direct(
                pinned,
                context[None],
                timestep.reshape(1),
                state["viewmats"],
                rotary,
                camera_rotary,
            )
            sigma = (timestep.float() / 1000).bfloat16()
            predicted = torch.where(mask, clean, reference - sigma * velocity)
            if index < 3:
                level = (
                    schedule.timesteps[index + 1].float() / 1000
                ).bfloat16()
                reference = torch.where(
                    mask, clean, (1 - level) * predicted + level * noise
                )
            else:
                reference = predicted
            torch.testing.assert_close(sample, reference, rtol=0, atol=0)
            torch.testing.assert_close(
                sample[mask.expand_as(sample)],
                clean[mask.expand_as(clean)],
                rtol=0,
                atol=0,
            )
        outputs.append(sample.clone())
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    assert not torch.equal(outputs[0], outputs[1])
