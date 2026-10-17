"""Camera numerical inputs retain the archive's declared cell domain."""

from dataclasses import replace

import pytest
import torch

from uniserve.model.camera import CameraCondition, CameraGrid
from uniserve_worker.execution.camera import camera_condition
from uniserve_worker.media.camera import (
    CameraArchive,
    dump_camera_archive,
    load_camera_archive,
)
from uniserve_worker.protocol.video import CameraInput, MediaLocator

pytestmark = pytest.mark.unit


def _archive(cells, *, raw_frames=None, times=None):
    poses = torch.eye(4).repeat(raw_frames or cells, 1, 1)
    poses[:, 0, 3] = torch.arange(poses.shape[0]) / 8
    return CameraArchive(
        intrinsics=torch.tensor([[16.0, 0, 8], [0, 16.0, 8], [0, 0, 1]]),
        world_to_camera=poses if raw_frames else None,
        camera_to_world=None if raw_frames else poses,
        clean_latents=torch.arange(2 * cells * 2 * 3)
        .reshape(2, cells, 2, 3)
        .bfloat16(),
        cell_time_index=times,
    )


def _roundtrip(archive, *, views, columns, order, clean):
    payload = dump_camera_archive(
        archive,
        views=views,
        columns=columns,
        order=order,
        clean_cells=clean,
    )
    request = CameraInput(
        MediaLocator("camera.safetensors", len(payload)),
        views,
        columns,
        order,
        clean,
    )
    return camera_condition(load_camera_archive(payload, request), request)


def test_stdm_condition_preserves_full_time_major_grid_and_row_head_slots():
    # In time-major order, the head occupies 0..31; subsequent camera-0
    # cells are 32,64,...,480. These are 47 distinct anchors, not 47 rows.
    clean = tuple(range(32)) + (
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
    )
    archive = _archive(
        512, times=torch.arange(16, dtype=torch.int64).repeat_interleave(32)
    )
    condition = _roundtrip(
        archive, views=32, columns=16, order="time_major", clean=clean
    )
    assert condition.grid == CameraGrid(32, 16, "time_major")
    assert condition.grid.num_cells == 512
    assert condition.clean_cells == clean
    assert len(condition.clean_cells) == 47
    assert condition.cell_time_index[[0, 31, 32, 479, 480, 511]].tolist() == [
        0,
        0,
        1,
        14,
        15,
        15,
    ]
    torch.testing.assert_close(
        condition.poses, archive.camera_to_world, rtol=0, atol=0
    )
    torch.testing.assert_close(
        condition.clean_latents, archive.clean_latents, rtol=0, atol=0
    )
    # A declared smaller grid must not reinterpret these 512 prepared cells.
    with pytest.raises(ValueError, match="pose count"):
        replace(condition, grid=CameraGrid(32, 15, "time_major"))


@pytest.mark.parametrize(
    "order,times",
    [
        ("view_major", [0, 1, 2, 0, 1, 2]),
        ("time_major", [0, 0, 1, 1, 2, 2]),
    ],
)
def test_condition_keeps_declared_order_without_reordering_values(order, times):
    archive = _archive(6, times=torch.tensor(times))
    condition = _roundtrip(
        archive, views=2, columns=3, order=order, clean=(0, 5)
    )
    assert condition.grid == CameraGrid(2, 3, order)
    assert condition.cell_time_index.tolist() == times
    torch.testing.assert_close(
        condition.poses, archive.camera_to_world, rtol=0, atol=0
    )


@pytest.mark.parametrize("raw_frames", [None, 156])
def test_stage1_pose_domain_remains_independent_of_the_48_cell_grid(raw_frames):
    archive = _archive(48, raw_frames=raw_frames)
    condition = _roundtrip(
        archive, views=48, columns=1, order="time_major", clean=(0,)
    )
    assert condition.grid == CameraGrid(48, 1, "time_major")
    assert condition.poses.shape == (raw_frames or 48, 4, 4)
    assert condition.clean_latents.shape == (2, 48, 2, 3)
    assert condition.clean_cells == (0,)
    assert condition.pose_space == (
        "raw_world_to_camera" if raw_frames else "prepared_camera_to_world"
    )
    assert condition.cell_time_index is None
    torch.testing.assert_close(
        condition.poses,
        archive.world_to_camera if raw_frames else archive.camera_to_world,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        condition.clean_latents, archive.clean_latents, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"poses": torch.eye(4).repeat(5, 1, 1)}, "pose count"),
        ({"clean_latents": torch.zeros(2, 5, 2, 3)}, "latent.*cell"),
        ({"cell_time_index": torch.arange(5)}, "timestamp.*cell"),
        ({"clean_cells": (0, 6)}, "clean.*grid"),
        ({"grid": (2, 3, "view_major")}, "CameraGrid"),
    ],
)
def test_condition_rejects_fields_outside_the_declared_grid(changes, match):
    condition = _roundtrip(
        _archive(6), views=2, columns=3, order="view_major", clean=(0,)
    )
    with pytest.raises(ValueError, match=match):
        replace(condition, **changes)


@pytest.mark.parametrize(
    "values",
    [
        (0, 16, "time_major"),
        (32, -1, "time_major"),
        (True, 16, "time_major"),
        (32, 16, "camera_major"),
    ],
)
def test_grid_requires_positive_integer_axes_and_a_supported_order(values):
    with pytest.raises(ValueError):
        CameraGrid(*values)


def test_direct_legacy_condition_does_not_infer_grid_from_raw_frames():
    archive = _archive(48, raw_frames=156)
    condition = CameraCondition(
        poses=archive.world_to_camera,
        intrinsics=archive.intrinsics,
        clean_cells=(0, 47),
        pose_space="raw_world_to_camera",
        clean_latents=archive.clean_latents,
    )
    assert condition.grid is None
    assert condition.clean_latents.shape[1] == 48
