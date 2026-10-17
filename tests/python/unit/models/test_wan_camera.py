"""Camera conventions and exact numerical parity with native Wan helpers.

The unit cases use analytic fixtures. Tests marked integration require the
native source checkout (IMP_GENAI_ROOT, or a sibling imp_genai checkout) and
einops; missing reference prerequisites fail visibly instead of skipping.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from uniserve_models.wan.camera import (
    CameraPoses,
    normalize_cameras,
    plucker_input,
    prepare_ring_cameras,
    ring_frame_indices,
    ring_latent_indices,
    scale_to_sinusoidal,
    segment_latent_indices,
    ucpe_input,
)


def _poses(slots=4, batch=1):
    angles = torch.linspace(0.17, 1.83, slots, dtype=torch.float64)
    rotations = torch.eye(4, dtype=torch.float64).repeat(batch, slots, 1, 1)
    rotations[..., 0, 0] = angles.cos()
    rotations[..., 0, 2] = angles.sin()
    rotations[..., 2, 0] = -angles.sin()
    rotations[..., 2, 2] = angles.cos()
    rotations[..., 0, 3] = angles.cos() * 3.7
    rotations[..., 1, 3] = angles * 0.13
    rotations[..., 2, 3] = angles.sin() * 2.3
    intrinsics = torch.tensor(
        [[700.5, 0, 410.25], [0, 702.0, 238.75], [0, 0, 1]],
        dtype=torch.float32,
    ).repeat(batch, slots, 1, 1)
    intrinsics[..., 0, 0] += torch.arange(slots) * 0.25
    if batch > 1:
        intrinsics[1:, :, 0, 0] *= 1.5
    return CameraPoses(rotations.float(), intrinsics)


@pytest.mark.unit
def test_frame_and_bundle_order():
    np.testing.assert_array_equal(segment_latent_indices(13), [0, 1, 5, 9])
    np.testing.assert_array_equal(
        segment_latent_indices(26), [0, 1, 5, 9, 13, 14, 18, 22]
    )
    np.testing.assert_array_equal(ring_frame_indices(156), np.arange(156))
    kept = np.delete(np.arange(160), [39, 79, 119, 159])
    np.testing.assert_array_equal(ring_frame_indices(160), kept)
    np.testing.assert_array_equal(
        ring_latent_indices(160)[:12],
        [0, 1, 5, 9, 13, 14, 18, 22, 26, 27, 31, 35],
    )
    assert ring_latent_indices(160)[12] == 40
    assert ring_latent_indices(160)[-1] == 155
    for invalid in (0, 12, 14):
        with pytest.raises(ValueError, match="13-frame"):
            segment_latent_indices(invalid)
    for invalid in (155, 161, 169):
        with pytest.raises(ValueError):
            ring_frame_indices(invalid)


@pytest.mark.unit
def test_raw_world_to_camera_metric_and_unit_diameter():
    # Four equal populations at radius 2, translated away from the world origin.
    centers = np.repeat(
        np.asarray([[2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0]]),
        40,
        axis=0,
    ).astype(np.float32)
    centers += np.asarray([8, 9, 10], dtype=np.float32)
    w2c = np.tile(np.eye(4, dtype=np.float32), (160, 1, 1))
    w2c[:, :3, 3] = -centers
    original = w2c.copy()
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (160, 1, 1))
    intrinsics[:, 0, 0] = np.arange(160) + 1
    selected = ring_latent_indices(160)
    expected = np.tile(np.eye(4, dtype=np.float32), (48, 1, 1))
    expected[:, :3, 3] = centers[selected] - centers[0]
    metric = prepare_ring_cameras(w2c, intrinsics, unit_diameter=False)
    unit = prepare_ring_cameras(w2c, intrinsics, unit_diameter=True)
    torch.testing.assert_close(
        metric.c2w[0], torch.from_numpy(expected), rtol=0, atol=0
    )
    expected[:, :3, 3] /= 4
    torch.testing.assert_close(
        unit.c2w[0], torch.from_numpy(expected), rtol=0, atol=0
    )
    torch.testing.assert_close(
        unit.intrinsics[0],
        torch.from_numpy(intrinsics[selected]),
        rtol=0,
        atol=0,
    )
    assert unit.c2w.dtype == unit.intrinsics.dtype == torch.float32
    np.testing.assert_array_equal(w2c, original)
    coincident = np.tile(np.eye(4, dtype=np.float32), (156, 1, 1))
    with pytest.raises(ValueError, match="nondegenerate"):
        prepare_ring_cameras(coincident, np.eye(3), unit_diameter=True)


@pytest.mark.unit
def test_plucker_transposed_rotation_moment_and_corner_pixel():
    # At pixel (0,0), this 90-degree camera rotation produces direction -X.
    # With origin (0,2,0), the Plücker moment is +2Z.
    c2w = torch.tensor(
        [[0, 0, 1, 0], [0, 1, 0, 2], [-1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )[None, None]
    pixel_k = torch.diag(torch.tensor([16.0, 16.0, 1.0]))[None, None]
    output = plucker_input(
        CameraPoses(c2w, pixel_k),
        height=16,
        width=16,
        token_grid=(1, 1, 1),
        dtype=torch.float32,
        scale_meters=None,
    )
    expected = torch.tensor([0.0, 0.0, 2.0, -1.0, 0.0, 0.0])
    torch.testing.assert_close(output.rays.flatten(), expected, rtol=0, atol=0)
    assert output.scale_embedding is None


@pytest.mark.unit
def test_ucpe_patch_center_is_camera_frame_and_maps_center_to_zero():
    c2w = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]],
        dtype=torch.float32,
    )[None, None]
    pixel_k = torch.tensor([[16.0, 0, 8], [0, 16.0, 8], [0, 0, 1]])[None, None]
    output = ucpe_input(
        CameraPoses(c2w, pixel_k),
        height=16,
        width=16,
        token_grid=(1, 1, 1),
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [[0.0, 0, -1, 3], [0, 1, 0, -2], [1, 0, 0, -1], [0, 0, 0, 1]]
    )
    torch.testing.assert_close(output.viewmats[0, 0], expected, rtol=0, atol=0)
    center = torch.tensor([1.0, 2.0, 3.0, 1.0])
    torch.testing.assert_close(
        output.viewmats[0, 0] @ center,
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.unit
def test_scale_saturation_and_batch_one():
    output = scale_to_sinusoidal(torch.tensor([[0.5]]))
    assert output.shape == (1, 16)
    torch.testing.assert_close(
        output,
        torch.cat([torch.zeros(1, 8), torch.ones(1, 8)], dim=1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scale_to_sinusoidal(-2.0), output[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        scale_to_sinusoidal(80.0), scale_to_sinusoidal(20.0), rtol=0, atol=0
    )


@pytest.mark.unit
@pytest.mark.parametrize("function", (plucker_input, ucpe_input))
def test_camera_grid_rejects_transpose_with_equal_token_count(function):
    kwargs = {"scale_meters": None} if function is plucker_input else {}
    with pytest.raises(ValueError, match="does not match token grid"):
        function(
            _poses(),
            height=32,
            width=48,
            token_grid=(4, 3, 2),
            dtype=torch.bfloat16,
            **kwargs,
        )


@pytest.fixture(scope="module")
def native_camera():
    root = (
        Path(
            os.environ.get(
                "IMP_GENAI_ROOT",
                str(Path(__file__).resolve().parents[5] / "imp_genai"),
            )
        )
        / "research/videogen/infra_fastvideo"
    )
    modules = []
    for path in (
        root / "models/plucker_camera.py",
        root / "models/ucpe_camera.py",
        root / "data/orbit360_dataset.py",
    ):
        if not path.is_file():
            pytest.fail(
                f"native camera reference missing: {path}; set IMP_GENAI_ROOT"
            )
        spec = importlib.util.spec_from_file_location(
            f"native_{path.stem}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


@pytest.mark.integration
@pytest.mark.parametrize(
    "height,width,slots", ((16, 16, 1), (32, 48, 8), (480, 832, 48))
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_plucker_matches_native_payload(
    native_camera, height, width, slots, dtype
):
    native, _, _ = native_camera
    poses = _poses(slots)
    scale = torch.tensor([[3.7]])
    output = plucker_input(
        poses,
        height=height,
        width=width,
        token_grid=(slots, height // 16, width // 16),
        dtype=dtype,
        scale_meters=scale,
    )
    expected = native.compute_plucker_camera_input(
        poses.c2w, poses.intrinsics, height, width, 16
    )["cam_emb"].to(dtype)
    torch.testing.assert_close(output.rays, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.scale_embedding,
        native.scale_to_sinusoidal(scale).to(dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "height,width,slots", ((16, 16, 1), (32, 48, 8), (480, 832, 48))
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_ucpe_matches_native_payload(
    native_camera, height, width, slots, dtype
):
    _, native, _ = native_camera
    poses = _poses(slots)
    output = ucpe_input(
        poses,
        height=height,
        width=width,
        token_grid=(slots, height // 16, width // 16),
        dtype=dtype,
    )
    expected = native.compute_ucpe_camera_input(
        poses.c2w, poses.intrinsics, height, width, 16
    )["viewmats"].to(dtype)
    torch.testing.assert_close(output.viewmats, expected, rtol=0, atol=0)


@pytest.mark.integration
def test_ucpe_batches_preserve_each_scenes_intrinsics(native_camera):
    _, native, _ = native_camera
    poses = _poses(batch=2)
    output = ucpe_input(
        poses, height=32, width=48, token_grid=(4, 2, 3), dtype=torch.float32
    )
    for batch in range(2):
        expected = native.compute_ucpe_camera_input(
            poses.c2w[batch : batch + 1],
            poses.intrinsics[batch : batch + 1],
            32,
            48,
            16,
        )["viewmats"]
        torch.testing.assert_close(
            output.viewmats[batch], expected[0], rtol=0, atol=0
        )


@pytest.mark.integration
def test_numpy_relative_poses_and_scale_match_native(native_camera):
    plucker, _, orbit = native_camera
    c2w = _poses(160).c2w[0].numpy()
    np.testing.assert_array_equal(
        normalize_cameras(c2w), orbit.normalize_to_view0(c2w)
    )
    scales = torch.tensor(
        [[0.1], [0.5], [1.0], [3.7], [20.0], [100.0]], dtype=torch.float64
    )
    torch.testing.assert_close(
        scale_to_sinusoidal(scales),
        plucker.scale_to_sinusoidal(scales),
        rtol=0,
        atol=0,
    )
    raw = np.linalg.inv(c2w).astype(np.float32)
    slots = ring_latent_indices(160)
    result = prepare_ring_cameras(raw, np.eye(3), unit_diameter=True)
    # Reference rounding is exercised via the native normalization helper.
    expected = orbit.normalize_to_view0(
        np.linalg.inv(raw[slots]).astype(np.float32)
    )
    full = orbit.normalize_to_view0(np.linalg.inv(raw).astype(np.float32))
    centers = np.asarray(full, dtype=np.float64)[:, :3, 3]
    radius = float(
        np.median(np.linalg.norm(centers - centers.mean(axis=0), axis=1))
    )
    expected[:, :3, 3] /= 2.0 * radius
    np.testing.assert_array_equal(result.c2w[0].numpy(), expected)
