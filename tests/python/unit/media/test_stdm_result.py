"""Complete grid results retain tails and declare per-camera preview timing."""

import io
import json
import zipfile

import pytest
import torch
from safetensors.torch import load

from uniserve.media import image
from uniserve.model.camera import CameraCondition, CameraGrid, CameraVideo
from uniserve_worker.media.camera import CameraArchive
from uniserve_worker.media.camera_result import CameraResult, dump_camera_result
from uniserve_worker.protocol.video import CameraInput, MediaLocator

pytestmark = pytest.mark.unit


def test_latent_only_grid_zip_keeps_native_values_and_explicit_view_timing():
    grid = CameraGrid(32, 16, "time_major")
    layout = CameraVideo(grid, 57, 15)
    poses = torch.eye(4).repeat(512, 1, 1)
    intrinsics = torch.tensor([[16, 0, 8], [0, 16, 8], [0, 0, 1]]).float()
    clean = tuple(range(32)) + tuple(range(32, 512, 32))
    times = torch.arange(16).repeat_interleave(32)
    latents = torch.arange(512).reshape(1, 1, 512, 1, 1).bfloat16()
    source = CameraArchive(
        intrinsics=intrinsics,
        camera_to_world=poses,
        cell_time_index=times,
        clean_latents=latents[0],
    )
    canonical = CameraCondition(
        poses,
        intrinsics,
        clean,
        "prepared_camera_to_world",
        clean_latents=latents[0],
        cell_time_index=times,
        grid=grid,
    )
    request = CameraInput(
        MediaLocator("fixture", 1), 32, 16, "time_major", clean, False
    )
    result = CameraResult(
        latents,
        source,
        canonical,
        request,
        0,
        1824,
        image.Config(16, 16),
        camera_video=layout,
        frame_rate=8,
    )
    with zipfile.ZipFile(io.BytesIO(dump_camera_result(result))) as archive:
        assert set(archive.namelist()) == {
            "result.safetensors",
            "metadata.json",
        }
        tensors = load(archive.read("result.safetensors"))
        metadata = json.loads(archive.read("metadata.json"))
    torch.testing.assert_close(tensors["latents"], latents, rtol=0, atol=0)
    assert tensors["latents"].shape[2] == 512
    assert metadata["grid"] == {
        "views": 32,
        "columns": 16,
        "order": "time_major",
    }
    assert metadata["clean_cells"] == list(clean)
    assert metadata["video"]["included"] is False
    assert metadata["video"]["frames"] == 1824
    assert metadata["video"]["order"] == "camera_major"
    assert metadata["video"]["duration_per_view_seconds"] == 57 / 8
    for view, record in enumerate(metadata["video"]["views"]):
        assert record["view"] == view
        assert record["frame_range"] == [view * 57, (view + 1) * 57]
        assert record["latent_cells"] == list(range(view, 480, 32))
        assert record["timestamps_seconds"] == [
            frame / 8 for frame in range(57)
        ]
