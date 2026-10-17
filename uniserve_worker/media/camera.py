"""Safe serialization of camera conditions at the CPU media boundary."""

import json
from dataclasses import dataclass, fields
from typing import Literal

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from uniserve_worker.protocol.video import CameraInput


@dataclass(frozen=True, slots=True)
class CameraArchive:
    """CPU tensors for one camera-conditioned request.

    Exactly one pose field and one image/latent field must be present.
    OpenCV ``world_to_camera`` contains every raw frame in metres;
    ``camera_to_world`` contains already prepared latent-cell poses whose
    relative frame and scale must be preserved. Intrinsics are in pixels.
    The record borrows tensors: callers must not mutate their storage while
    it is in use. Loading does not promise zero-copy sharing with its bytes.
    """

    intrinsics: torch.Tensor
    world_to_camera: torch.Tensor | None = None
    camera_to_world: torch.Tensor | None = None
    scene_scale: torch.Tensor | None = None
    clean_latents: torch.Tensor | None = None
    frames: torch.Tensor | None = None
    cell_time_index: torch.Tensor | None = None


_TENSOR_NAMES = frozenset(field.name for field in fields(CameraArchive))
_POSE_SPACES = {
    "world_to_camera": "raw_world_to_camera",
    "camera_to_world": "prepared_camera_to_world",
}


def _grid_metadata(
    views: int,
    columns: int,
    order: str,
    clean_cells: tuple[int, ...],
) -> dict[str, str]:
    if (
        type(views) is not int
        or type(columns) is not int
        or views <= 0
        or columns <= 0
        or views * columns > 2**32 - 1
    ):
        raise ValueError("camera grid requires a positive u32 cell count")
    if order not in {"view_major", "time_major"}:
        raise ValueError("camera order must be view_major or time_major")
    if (
        not isinstance(clean_cells, tuple)
        or any(type(cell) is not int for cell in clean_cells)
        or len(set(clean_cells)) != len(clean_cells)
        or any(cell < 0 or cell >= views * columns for cell in clean_cells)
    ):
        raise ValueError(
            "clean camera cells must be distinct and inside the grid"
        )
    return {
        "schema": "uniserve.camera",
        "version": "1",
        "views": str(views),
        "columns": str(columns),
        "order": order,
        "clean_cells": json.dumps(clean_cells, separators=(",", ":")),
    }


def _validate_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    views: int,
    columns: int,
    order: str,
) -> str:
    if set(tensors) - _TENSOR_NAMES or "intrinsics" not in tensors:
        raise ValueError("camera archive has unknown or missing tensor fields")
    pose_keys = set(tensors) & _POSE_SPACES.keys()
    if len(pose_keys) != 1:
        raise ValueError(
            "camera archive requires exactly one pose representation"
        )
    if len(set(tensors) & {"clean_latents", "frames"}) != 1:
        raise ValueError(
            "camera archive requires exactly one of clean_latents or frames"
        )
    for name, tensor in tensors.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
        ):
            raise ValueError(f"{name} must be a dense CPU tensor")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")

    pose_key = next(iter(pose_keys))
    poses = tensors[pose_key]
    if (
        poses.dtype != torch.float32
        or poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or poses.shape[0] == 0
    ):
        raise ValueError("camera poses require FP32 N,4,4")
    cells = views * columns
    if pose_key == "camera_to_world" and poses.shape[0] != cells:
        raise ValueError(
            "prepared camera pose count must match the latent grid"
        )
    homogeneous = poses.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(
        poses.shape[0], 4
    )
    if not torch.equal(poses[:, 3], homogeneous):
        raise ValueError("camera pose last rows must be homogeneous [0,0,0,1]")

    intrinsics = tensors["intrinsics"]
    if intrinsics.dtype != torch.float32 or intrinsics.shape not in (
        (3, 3),
        (poses.shape[0], 3, 3),
    ):
        raise ValueError("intrinsics require FP32 3,3 or per-pose N,3,3")
    if (intrinsics[..., 0, 0] <= 0).any() or (intrinsics[..., 1, 1] <= 0).any():
        raise ValueError("camera focal lengths must be positive")

    if "scene_scale" in tensors:
        scale = tensors["scene_scale"]
        if scale.dtype != torch.float32 or scale.shape != (1,) or scale[0] <= 0:
            raise ValueError(
                "scene_scale requires one positive FP32 value in metres"
            )
    if "clean_latents" in tensors:
        latents = tensors["clean_latents"]
        if (
            latents.dtype not in {torch.bfloat16, torch.float16, torch.float32}
            or latents.ndim != 4
            or min(latents.shape) <= 0
            or latents.shape[1] != cells
        ):
            raise ValueError(
                "clean_latents require floating C,N,H,W with N grid cells"
            )
    if "frames" in tensors:
        frames = tensors["frames"]
        if (
            frames.dtype != torch.uint8
            or frames.ndim != 4
            or min(frames.shape) <= 0
            or frames.shape[0] != 3
        ):
            raise ValueError("frames require uint8 RGB with shape 3,F,H,W")
        if pose_key == "world_to_camera" and frames.shape[1] != poses.shape[0]:
            raise ValueError(
                "raw RGB frame count must match raw camera pose count"
            )
    if "cell_time_index" in tensors:
        times = tensors["cell_time_index"]
        if times.dtype != torch.int64 or times.shape != (cells,):
            raise ValueError(
                "cell_time_index requires int64 with one value per grid cell"
            )
        column = torch.arange(columns, dtype=torch.int64, device=times.device)
        expected = (
            column.repeat(views)
            if order == "view_major"
            else column.repeat_interleave(views)
        )
        if not torch.equal(times, expected):
            raise ValueError(
                "cell_time_index does not match the camera grid order"
            )
    return _POSE_SPACES[pose_key]


def dump_camera_archive(
    archive: CameraArchive,
    *,
    views: int,
    columns: int,
    order: Literal["view_major", "time_major"],
    clean_cells: tuple[int, ...],
) -> bytes:
    """Validate and serialize CPU values without changing dtype or ordering.

    The safetensors header binds these values to their declared grid and
    clean cells. Noncontiguous tensors are packed without numeric conversion.
    Callers must keep the borrowed tensor storage stable for this call.
    Publication and file/network IO remain outside this function.
    """
    metadata = _grid_metadata(views, columns, order, clean_cells)
    tensors = {
        field.name: value
        for field in fields(archive)
        if (value := getattr(archive, field.name)) is not None
    }
    metadata["pose_space"] = _validate_tensors(
        tensors, views=views, columns=columns, order=order
    )
    # Independent packed values also permit fields that borrow overlapping
    # source storage; the archive records values, never tensor aliasing.
    packed = {
        name: tensor.detach().clone(memory_format=torch.contiguous_format)
        for name, tensor in tensors.items()
    }
    return save(packed, metadata=metadata)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate camera archive header key: {key}")
        result[key] = value
    return result


def _header(payload: bytes) -> dict:
    if len(payload) < 8:
        raise ValueError("camera archive is shorter than a safetensors header")
    header_size = int.from_bytes(payload[:8], "little")
    if header_size == 0 or header_size > len(payload) - 8:
        raise ValueError(
            "camera archive has an invalid safetensors header length"
        )
    try:
        header = json.loads(
            payload[8 : 8 + header_size], object_pairs_hook=_unique_object
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("camera archive has an invalid JSON header") from error
    if not isinstance(header, dict):
        raise ValueError("camera archive header must be an object")
    return header


def load_camera_archive(payload: bytes, camera: CameraInput) -> CameraArchive:
    """Read safe tensor bytes and validate their complete request contract.

    Returns CPU tensors whose values are borrowed by downstream preparation.
    Raw poses and RGB frames retain every supplied frame; supported raw clip
    counts and numerical camera normalization are the model's responsibility.
    Prepared poses and clean latents must already have views*columns cells.
    This function performs no IO, device transfers, or model mathematics.

    Raises:
        ValueError: Invalid archive, grid, tensor values, metadata agreement,
            or a byte count differing from the published MediaLocator.
    """
    if not isinstance(payload, bytes):
        raise ValueError("camera archive payload must be bytes")
    if (
        type(camera.source.bytes) is not int
        or len(payload) != camera.source.bytes
    ):
        raise ValueError("camera archive byte count does not match its source")
    expected = _grid_metadata(
        camera.views, camera.columns, camera.order, camera.clean_cells
    )
    header = _header(payload)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError("camera archive requires schema metadata")
    names = set(header) - {"__metadata__"}
    pose_keys = names & _POSE_SPACES.keys()
    if len(pose_keys) != 1:
        raise ValueError(
            "camera archive requires exactly one pose representation"
        )
    if names - _TENSOR_NAMES:
        raise ValueError("camera archive has unknown tensor fields")
    expected["pose_space"] = _POSE_SPACES[next(iter(pose_keys))]
    # Safetensors metadata stores the cell list as JSON text. Whitespace is
    # serialization detail; its integer values and ordering bind the request.
    try:
        clean_cells = json.loads(metadata.get("clean_cells", ""))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(
            "camera archive has invalid clean_cells metadata"
        ) from error
    if not isinstance(clean_cells, list) or any(
        type(cell) is not int for cell in clean_cells
    ):
        raise ValueError(
            "camera archive clean_cells metadata requires integer indices"
        )
    metadata = {
        **metadata,
        "clean_cells": json.dumps(clean_cells, separators=(",", ":")),
    }
    if metadata != expected:
        raise ValueError(
            "camera archive metadata does not match the request contract"
        )
    try:
        tensors = load(payload)
    except (SafetensorError, ValueError, RuntimeError, KeyError) as error:
        raise ValueError(
            "camera archive contains invalid safetensors data"
        ) from error
    _validate_tensors(
        tensors, views=camera.views, columns=camera.columns, order=camera.order
    )
    return CameraArchive(**tensors)
