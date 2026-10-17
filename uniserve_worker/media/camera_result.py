"""Native camera results packaged by the runtime for downstream stages."""

import io
import json
import zipfile
from dataclasses import dataclass

import torch
from safetensors.torch import save

from uniserve.media import image
from uniserve.model.camera import CameraCondition, CameraVideo
from uniserve_worker.media.camera import CameraArchive
from uniserve_worker.protocol.video import CameraInput


@dataclass(frozen=True, slots=True)
class CameraResult:
    """Borrow a complete native latent and its original/canonical cameras.

    The host execution owner retains every tensor through serialization.
    ``latents`` keeps the model's full [1,C,cells,H,W] shape and dtype;
    ``canonical`` contains prepared OpenCV camera-to-world poses in exactly
    that cell order. No tensor is converted or normalized by the exporter.
    """

    latents: torch.Tensor
    original: CameraArchive
    canonical: CameraCondition
    camera: CameraInput
    seed: int
    num_frames: int
    canvas: image.Config
    camera_video: CameraVideo | None = None
    frame_rate: int = 24


def dump_camera_result(
    result: CameraResult, *, video: bytes | None = None
) -> bytes:
    """Create a ZIP of native tensors, camera metadata and optional MP4.

    ZIP members use fixed names, stored bytes and a fixed timestamp. The
    safetensors member retains both input poses and canonical latent-cell
    cameras, avoiding any downstream inverse/normalization roundtrip. This
    function owns neither publication nor request lifetime.
    """
    latents, canonical, original, camera = (
        result.latents,
        result.canonical,
        result.original,
        result.camera,
    )
    cells = camera.views * camera.columns
    if (
        latents.device.type != "cpu"
        or latents.dtype not in {torch.bfloat16, torch.float16, torch.float32}
        or latents.ndim != 5
        or latents.shape[0] != 1
        or latents.shape[2] != cells
        or min(latents.shape) <= 0
        or not torch.isfinite(latents).all()
    ):
        raise ValueError(
            "camera result requires finite native CPU [1,C,cells,H,W] latents"
        )
    if (
        canonical.pose_space != "prepared_camera_to_world"
        or canonical.poses.shape != (cells, 4, 4)
        or canonical.poses.dtype != torch.float32
        or canonical.intrinsics.dtype != torch.float32
    ):
        raise ValueError(
            "camera result requires canonical FP32 cameras for every cell"
        )
    if type(result.seed) is not int or not 0 <= result.seed < 2**64:
        raise ValueError(
            "camera result seed must be an unsigned 64-bit integer"
        )
    if (
        result.num_frames < 1
        or type(result.frame_rate) is not int
        or result.frame_rate < 1
        or min(result.canvas.height, result.canvas.width) < 1
    ):
        raise ValueError(
            "camera result requires a positive output timeline and canvas"
        )
    if camera.include_video != (video is not None) or video == b"":
        raise ValueError("camera result preview must match include_video")
    if original.clean_latents is not None:
        clean = original.clean_latents
        if clean.shape != latents.shape[1:]:
            raise ValueError(
                "camera result clean condition has another native shape"
            )
        indices = list(camera.clean_cells)
        if not torch.equal(
            latents[0, :, indices], clean[:, indices].to(latents.dtype)
        ):
            raise ValueError("generated camera result changed a clean cell")

    raw = original.world_to_camera is not None
    input_poses = original.world_to_camera if raw else original.camera_to_world
    if input_poses is None:
        raise ValueError("camera result requires the original input poses")
    tensors = {
        "latents": latents,
        "camera_to_world": canonical.poses,
        "intrinsics": canonical.intrinsics,
        "input_world_to_camera"
        if raw
        else "input_camera_to_world": input_poses,
        "input_intrinsics": original.intrinsics,
    }
    if canonical.scene_scale is not None:
        tensors["scene_scale"] = canonical.scene_scale
    if canonical.cell_time_index is not None:
        tensors["cell_time_index"] = canonical.cell_time_index
    if original.cell_time_index is not None:
        tensors["input_cell_time_index"] = original.cell_time_index
    metadata = {
        "schema": "uniserve.camera.result",
        "version": 1,
        "seed": result.seed,
        "grid": {
            "views": camera.views,
            "columns": camera.columns,
            "order": camera.order,
        },
        "clean_cells": list(camera.clean_cells),
        "input_pose_space": "raw_world_to_camera"
        if raw
        else "prepared_camera_to_world",
        "camera_pose_space": canonical.pose_space,
        "camera_convention": "opencv",
        "intrinsics_units": "pixels",
        "scene_scale_units": "metres",
        "latents": {
            "file": "result.safetensors",
            "tensor": "latents",
            "shape": list(latents.shape),
            "dtype": str(latents.dtype).removeprefix("torch."),
            "axes": ["batch", "channel", "cell", "height", "width"],
        },
        "video": {
            "included": camera.include_video,
            "frames": result.num_frames,
            "height": result.canvas.height,
            "width": result.canvas.width,
        },
    }
    metadata["video"]["frame_rate"] = result.frame_rate
    if result.camera_video is not None:
        layout = result.camera_video
        if layout.grid != camera.grid or layout.num_frames != result.num_frames:
            raise ValueError(
                "camera video metadata must describe its complete grid "
                "and timeline"
            )
        metadata["video"].update(
            {
                "order": "camera_major",
                "frames_per_view": layout.frames_per_view,
                "duration_per_view_seconds": layout.frames_per_view
                / result.frame_rate,
                "views": [
                    {
                        "view": view,
                        "frame_range": [
                            view * layout.frames_per_view,
                            (view + 1) * layout.frames_per_view,
                        ],
                        "latent_cells": list(layout.cells(view)),
                        "timestamps_seconds": [
                            frame / result.frame_rate
                            for frame in range(layout.frames_per_view)
                        ],
                    }
                    for view in range(layout.grid.views)
                ],
            }
        )
    # Independently packed tensors also support borrowed fields that alias
    # each other. Packing changes storage strides, never numerical layout.
    packed = {
        name: value.detach().clone(memory_format=torch.contiguous_format)
        for name, value in tensors.items()
    }
    members = {
        "result.safetensors": save(
            packed, metadata={"schema": metadata["schema"], "version": "1"}
        ),
        "metadata.json": json.dumps(
            metadata, separators=(",", ":"), sort_keys=True
        ).encode(),
    }
    if video is not None:
        members["video.mp4"] = video
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for name, payload in members.items():
            archive.writestr(zipfile.ZipInfo(name), payload)
    return output.getvalue()
