"""Camera capability preparation and complete Wan DMD input binding."""

import pytest
import torch

from uniserve.diffusion import DenoisingStep
from uniserve.media import image
from uniserve.model import LatentInput, VideoDenoiser
from uniserve.model.camera import CameraCondition, CameraDenoiser
from uniserve_models.wan.config import Config, TransformerConfig
from uniserve_models.wan.denoiser import Denoiser
from uniserve_models.wan.inputs import LatentSize
from uniserve_models.wan.model import Model
from uniserve_models.wan.serving import ServingDenoiser, VideoSize

pytestmark = pytest.mark.unit


@pytest.fixture
def config():
    return Config(
        TransformerConfig(
            num_layers=1,
            num_heads=2,
            head_dim=8,
            ffn_dim=32,
            in_channels=4,
            out_channels=4,
            text_dim=12,
            freq_dim=8,
            scale_bands=2,
        )
    )


def _buffers(fields):
    return {
        name: torch.empty(field.shape, dtype=field.dtype)
        for name, field in fields.items()
    }


def _condition(size, *, clean=True):
    poses = torch.eye(4).repeat(size.frames, 1, 1)
    poses[:, 0, 3] = 2
    return CameraCondition(
        poses=poses,
        intrinsics=torch.eye(3),
        clean_cells=(0, size.frames - 1),
        pose_space="prepared_camera_to_world",
        scene_scale=torch.tensor([3.0]),
        clean_latents=torch.full(
            (4, size.frames, size.height, size.width), 0.25
        )
        if clean
        else None,
    )


def test_model_exposes_camera_video_capability_and_padded_layout(config):
    model = Model(config).transformer
    assert isinstance(model, CameraDenoiser)
    assert isinstance(model, VideoDenoiser)
    assert model.tasks == ("camera2v",) and not model.text_only
    assert model.select_canvases(("480p",), ("26:15",)) == (
        image.Config(480, 832),
    )
    assert model.legal_frame_count(14) == 26
    size = model.make_size(156, 19, canvas=image.Config(480, 832))
    assert (size.frames, size.height, size.width) == (48, 60, 104)
    layout = model.layout_size(size)
    assert layout.num_text_tokens == 512
    other = model.make_size(156, 20, canvas=size.canvas)
    assert model.holds(layout, size)
    assert model.holds(layout, other)
    assert model.text_condition_rows(size) == 512
    assert model.output_layout(size)["video"].shape == (1, 4, 48, 60, 104)
    assert model.sampling_noise_shapes(size) == ((1, 4, 48, 60, 104),) * 3
    with pytest.raises(ValueError, match="13-frame"):
        model.make_size(14, 19, canvas=size.canvas)


def test_prepared_poses_preserve_translation_and_explicit_clean_cells(config):
    model = ServingDenoiser(config)
    size = VideoSize(13, image.Config(16, 16), 5)
    out = _buffers(model.camera_buffers(size))
    condition = _condition(size)
    original = condition.poses.clone()
    model.prepare_camera(size, condition, out=out)
    # With identity rotation and a +2 x camera center, the z ray's moment
    # is (0,-2,0). Inverting or normalizing prepared poses changes this.
    torch.testing.assert_close(
        out["camera_rays"][0, :, 0, 0, 0],
        torch.tensor([0, -2, 0, 0, 0, 1], dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(condition.poses, original, rtol=0, atol=0)
    assert out["clean_mask"].flatten().tolist() == [True, False, False, True]
    assert torch.equal(out["clean"], condition.clean_latents[None].bfloat16())
    assert out["scale_embedding"].shape == (1, 4)


def test_raw_ring_uses_unit_diameter_and_native_rgb_selection(config):
    model = ServingDenoiser(config)
    size = VideoSize(156, image.Config(16, 16), 5)
    raw = torch.eye(4).repeat(160, 1, 1)
    centers = torch.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    raw[:, :3, 3] = -centers.repeat_interleave(40, dim=0)
    condition = CameraCondition(
        poses=raw,
        intrinsics=torch.eye(3),
        clean_cells=(0,),
        pose_space="raw_world_to_camera",
        scene_scale=torch.tensor([2.0]),
    )
    out = _buffers(model.camera_buffers(size))
    model.prepare_camera(size, condition, out=out)
    assert model.input_frame_indices(size, condition, 160) == tuple(
        index for index in range(160) if index not in (39, 79, 119, 159)
    )
    # Raw frame 80 is latent cell 24. Its translation from camera zero is
    # -2 x metres; unit-diameter preparation divides by the ring diameter 2.
    torch.testing.assert_close(
        out["camera_rays"][0, :, 24, 0, 0],
        torch.tensor([0, 1, 0, 0, 0, 1], dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert not out["clean"].count_nonzero()


def test_online_clean_rows_fill_native_storage_without_changing_mask(config):
    model = ServingDenoiser(config)
    size = VideoSize(13, image.Config(16, 16), 5)
    condition = _condition(size, clean=False)
    out = _buffers(model.camera_buffers(size))
    model.prepare_camera(size, condition, out=out)
    assert model.input_frame_indices(size, condition, 13) == tuple(range(13))
    rows = torch.arange(4 * 2 * 2 * 4, dtype=torch.bfloat16).reshape(16, 4)
    model.store_clean(size, rows, out=out["clean"])
    for frame in range(4):
        for row in range(2):
            for column in range(2):
                torch.testing.assert_close(
                    out["clean"][0, :, frame, row, column],
                    rows[(frame * 2 + row) * 2 + column],
                    rtol=0,
                    atol=0,
                )
    assert out["clean_mask"].flatten().tolist() == [True, False, False, True]


@torch.inference_mode()
def test_camera_capability_matches_direct_denoiser_four_step_trajectory(config):
    torch.manual_seed(91)
    model = ServingDenoiser(config).bfloat16()
    for name, parameter in model.named_parameters():
        parameter.normal_(std=0.1)
        if "norm" in name and name.endswith("weight"):
            parameter.add_(1)
    direct = Denoiser(config).bfloat16()
    direct.load_state_dict(model.state_dict(), strict=True)
    size = VideoSize(13, image.Config(16, 16), 5)
    direct_size = LatentSize(size.frames, size.height, size.width, 512)
    camera = _buffers(model.camera_buffers(size))
    model.prepare_camera(size, _condition(size), out=camera)
    constants = _buffers(model.constant_buffers(size))
    model.prepare_constants(size, out=constants)
    schedule = model.make_schedules(device="cpu")
    rng = torch.Generator().manual_seed(17)
    native_shape = model.noise_shape("video", size)
    draws = [
        torch.randn(native_shape, generator=rng).bfloat16() for _ in range(4)
    ]
    sample, reference = draws[0].clone(), draws[0].clone()
    text = torch.randn(5, 12).bfloat16()
    text = torch.cat((text, torch.zeros(507, 12, dtype=text.dtype)))
    for index in range(4):
        timestep = schedule["video"].timesteps[index]
        step = schedule["video"].step(index)
        noise = draws[index + 1] if index < 3 else torch.zeros_like(sample)
        actual_inputs = model.bind_camera_inputs(
            latents={"video": (LatentInput(sample, timestep),)},
            sizes=(size,),
            step=step,
            text_features=(text,),
            state=camera,
            noise=(noise,),
        )
        expected_inputs = direct.bind_inputs(
            sample=reference,
            timestep=timestep,
            step=step,
            size=direct_size,
            context=text[None],
            camera_rays=camera["camera_rays"],
            scale_embedding=camera["scale_embedding"],
            noise=noise,
            clean=camera["clean"],
            clean_mask=camera["clean_mask"],
        )
        for denoiser, inputs in (
            (model, actual_inputs),
            (direct, expected_inputs),
        ):
            DenoisingStep(denoiser, inputs, schedule, {}, constants, {})()
        torch.testing.assert_close(sample, reference, rtol=0, atol=0)
        mask = camera["clean_mask"].expand_as(sample)
        torch.testing.assert_close(
            sample[mask], camera["clean"][mask], rtol=0, atol=0
        )
    assert torch.isfinite(sample).all()


def _native_camera_parity():
    """Check actual CUDA preparation against the native ray/scale functions."""
    from research.videogen.infra_fastvideo.models import plucker_camera

    config = Config(
        TransformerConfig(
            num_layers=1,
            num_heads=2,
            head_dim=16,
            ffn_dim=64,
            text_dim=32,
            freq_dim=16,
        )
    )
    model = ServingDenoiser(config)
    size = model.make_size(156, 5, canvas=image.Config(480, 832))
    angle = torch.arange(size.frames, dtype=torch.float32) * 0.13
    poses = torch.eye(4).repeat(size.frames, 1, 1)
    poses[:, 0, 0] = angle.cos()
    poses[:, 0, 2] = angle.sin()
    poses[:, 2, 0] = -angle.sin()
    poses[:, 2, 2] = angle.cos()
    poses[:, :3, 3] = torch.stack((angle * 0.7, angle.sin(), angle.cos()), -1)
    intrinsic = torch.tensor(
        [[701.25, 0, 413.75], [0, 699.5, 241.0], [0, 0, 1]]
    )
    conditions = [
        CameraCondition(
            poses=value,
            intrinsics=intrinsic,
            clean_cells=(0, 47),
            pose_space="prepared_camera_to_world",
            scene_scale=torch.tensor([scale]),
        )
        for value, scale in (
            (poses, 3.75),
            (poses.flip(0), 6.25),
            (poses, 3.75),
        )
    ]
    out = {
        name: torch.empty(field.shape, dtype=field.dtype, device="cuda")
        for name, field in model.camera_buffers(size).items()
    }
    outputs = []
    with torch.inference_mode():
        for condition in conditions:
            model.prepare_camera(size, condition, out=out)
            expected = plucker_camera.compute_plucker_camera_input(
                c2w=condition.poses[None].cuda().float(),
                intrinsics=intrinsic.expand(size.frames, -1, -1)[None]
                .cuda()
                .float(),
                height=480,
                width=832,
                patch_size=16,
            )["cam_emb"].bfloat16()
            scale = plucker_camera.scale_to_sinusoidal(
                condition.scene_scale.cuda(), num_bands=8
            ).bfloat16()
            torch.testing.assert_close(
                out["camera_rays"], expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                out["scale_embedding"], scale, rtol=0, atol=0
            )
            outputs.append(out["camera_rays"].clone())
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    assert not torch.equal(outputs[0], outputs[1])
    print("PASS: CUDA Plucker ray/scale preparation and A/B/A camera isolation")


if __name__ == "__main__":
    _native_camera_parity()
