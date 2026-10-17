"""The camera media boundary preserves values and rejects ambiguous archives."""

import json
from dataclasses import fields, replace

import pytest
import torch
from safetensors.torch import save

from uniserve_worker.media.camera import (
    CameraArchive,
    dump_camera_archive,
    load_camera_archive,
)
from uniserve_worker.protocol.video import CameraInput, MediaLocator

pytestmark = pytest.mark.unit


def _prepared(cells=48):
    poses = torch.eye(4).repeat(cells, 1, 1)
    poses[:, 0, 3] = torch.arange(cells) / 8
    intrinsics = torch.tensor([[700.0, 0, 416], [0, 702.0, 240], [0, 0, 1]])
    latents = (
        torch.arange(16 * cells * 2 * 3)
        .reshape(16, cells, 2, 3)
        .to(torch.bfloat16)
        .transpose(-1, -2)
    )
    return CameraArchive(
        intrinsics=intrinsics,
        camera_to_world=poses,
        scene_scale=torch.tensor([3.5]),
        clean_latents=latents,
    )


def _dump(archive, *, views=12, columns=4, order="view_major", clean=(0,)):
    return dump_camera_archive(
        archive, views=views, columns=columns, order=order, clean_cells=clean
    )


def _request(payload, *, views=12, columns=4, order="view_major", clean=(0,)):
    return CameraInput(
        MediaLocator("camera_archive", len(payload)),
        views,
        columns,
        order,
        clean,
    )


def _assert_values(actual, expected):
    for field in fields(CameraArchive):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if right is None:
            assert left is None
        else:
            assert left.device.type == "cpu"
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def _tensors(archive):
    return {
        field.name: tensor.contiguous().clone()
        for field in fields(CameraArchive)
        if (tensor := getattr(archive, field.name)) is not None
    }


def _metadata(payload):
    size = int.from_bytes(payload[:8], "little")
    return json.loads(payload[8 : 8 + size])["__metadata__"]


def test_prepared_bf16_roundtrip_preserves_values_and_pose_semantics():
    archive = _prepared()
    payload = _dump(archive)
    actual = load_camera_archive(payload, _request(payload))
    _assert_values(actual, archive)
    assert actual.clean_latents.dtype == torch.bfloat16
    assert _metadata(payload)["pose_space"] == "prepared_camera_to_world"


@pytest.mark.parametrize("raw", (False, True))
def test_rgb_roundtrip_keeps_all_frames_in_native_channel_order(raw):
    count = 160 if raw else 48
    poses = torch.eye(4).repeat(count, 1, 1)
    poses[:, 2, 3] = torch.arange(count) / 16
    frames = torch.arange(3 * 160 * 2 * 4).remainder(256).to(torch.uint8)
    archive = CameraArchive(
        intrinsics=torch.eye(3).repeat(count, 1, 1),
        world_to_camera=poses if raw else None,
        camera_to_world=None if raw else poses,
        frames=frames.reshape(3, 160, 2, 4),
    )
    payload = _dump(archive)
    actual = load_camera_archive(payload, _request(payload))
    _assert_values(actual, archive)
    assert actual.frames.shape == (3, 160, 2, 4)
    assert _metadata(payload)["pose_space"] == (
        "raw_world_to_camera" if raw else "prepared_camera_to_world"
    )


@pytest.mark.parametrize("encoding", ("compact", "spaced", "indented"))
def test_stdm_grid_keeps_time_columns_and_row_head_clean_union(encoding):
    clean = tuple(sorted(set(range(32)) | set(range(0, 512, 32))))
    assert len(clean) == 47
    archive = replace(
        _prepared(512),
        cell_time_index=torch.arange(16).repeat_interleave(32),
    )
    payload = _dump(
        archive, views=32, columns=16, order="time_major", clean=clean
    )
    if encoding != "compact":
        metadata = _metadata(payload)
        metadata["clean_cells"] = json.dumps(
            clean, indent=2 if encoding == "indented" else None
        )
        payload = save(_tensors(archive), metadata=metadata)
    actual = load_camera_archive(
        payload,
        _request(
            payload, views=32, columns=16, order="time_major", clean=clean
        ),
    )
    _assert_values(actual, archive)


@pytest.mark.parametrize(
    "cells", ("[0.0,1]", "[false,1]", "null", '{"0":0}', "[", "[1,0]")
)
def test_reader_rejects_invalid_or_different_clean_cell_json(cells):
    archive = _prepared()
    metadata = _metadata(_dump(archive, clean=(0, 1)))
    metadata["clean_cells"] = cells
    payload = save(_tensors(archive), metadata=metadata)
    with pytest.raises(ValueError, match="metadata"):
        load_camera_archive(payload, _request(payload, clean=(0, 1)))


@pytest.mark.parametrize(
    "changes",
    (
        {"views": 6, "columns": 8},
        {"order": "time_major"},
        {"clean_cells": (4,)},
    ),
)
def test_reader_rejects_valid_bytes_for_a_different_request(changes):
    payload = _dump(_prepared())
    request = replace(_request(payload), **changes)
    with pytest.raises(ValueError, match="metadata does not match"):
        load_camera_archive(payload, request)


@pytest.mark.parametrize(
    "changes",
    (
        {"views": 0},
        {"views": True},
        {"views": 2**32},
        {"order": "camera_major"},
        {"clean_cells": (0, 0)},
        {"clean_cells": (-1,)},
        {"clean_cells": (48,)},
        {"clean_cells": (False,)},
    ),
)
def test_reader_and_writer_reject_invalid_grid_contract(changes):
    archive = _prepared()
    payload = _dump(archive)
    request = replace(_request(payload), **changes)
    with pytest.raises(ValueError):
        load_camera_archive(payload, request)
    with pytest.raises(ValueError):
        dump_camera_archive(
            archive,
            views=request.views,
            columns=request.columns,
            order=request.order,
            clean_cells=request.clean_cells,
        )


@pytest.mark.parametrize(
    "defect,match",
    (
        ("missing_pose", "exactly one pose"),
        ("both_poses", "exactly one pose"),
        ("missing_condition", "exactly one of"),
        ("both_conditions", "exactly one of"),
        ("missing_intrinsics", "missing tensor"),
        ("unknown_tensor", "unknown tensor"),
        ("wrong_pose_dtype", "FP32 N,4,4"),
        ("wrong_pose_count", "pose count"),
        ("nonfinite_pose", "finite"),
        ("bad_homogeneous_row", "homogeneous"),
        ("nonfinite_intrinsics", "finite"),
        ("bad_focal", "focal"),
        ("wrong_intrinsics_count", "per-pose"),
        ("bad_scene_scale", "positive FP32"),
        ("nonfinite_latents", "finite"),
        ("wrong_latent_count", "grid cells"),
        ("wrong_latent_dtype", "floating C,N,H,W"),
        ("bad_time_order", "grid order"),
        ("wrong_time_dtype", "int64"),
        ("wrong_rgb_dtype", "uint8 RGB"),
        ("wrong_rgb_layout", "uint8 RGB"),
        ("raw_rgb_count", "raw camera pose count"),
    ),
)
def test_reader_rejects_malformed_tensor_payloads(defect, match):
    archive = _prepared()
    metadata = _metadata(_dump(archive))
    tensors = _tensors(archive)
    if defect == "missing_pose":
        tensors.pop("camera_to_world")
    elif defect == "both_poses":
        tensors["world_to_camera"] = tensors["camera_to_world"].clone()
    elif defect == "missing_condition":
        tensors.pop("clean_latents")
    elif defect == "both_conditions":
        tensors["frames"] = torch.zeros(3, 156, 2, 4, dtype=torch.uint8)
    elif defect == "missing_intrinsics":
        tensors.pop("intrinsics")
    elif defect == "unknown_tensor":
        tensors["unexpected"] = torch.ones(1)
    elif defect == "wrong_pose_dtype":
        tensors["camera_to_world"] = tensors["camera_to_world"].bfloat16()
    elif defect == "wrong_pose_count":
        tensors["camera_to_world"] = tensors["camera_to_world"][
            :-1
        ].contiguous()
    elif defect == "nonfinite_pose":
        tensors["camera_to_world"][0, 0, 3] = float("nan")
    elif defect == "bad_homogeneous_row":
        tensors["camera_to_world"][0, 3, 0] = 1.0
    elif defect == "nonfinite_intrinsics":
        tensors["intrinsics"][0, 0] = float("inf")
    elif defect == "bad_focal":
        tensors["intrinsics"][1, 1] = 0
    elif defect == "wrong_intrinsics_count":
        tensors["intrinsics"] = tensors["intrinsics"].repeat(47, 1, 1)
    elif defect == "bad_scene_scale":
        tensors["scene_scale"][0] = -1
    elif defect == "nonfinite_latents":
        tensors["clean_latents"][0, 0, 0, 0] = float("nan")
    elif defect == "wrong_latent_count":
        tensors["clean_latents"] = tensors["clean_latents"][:, :1].contiguous()
    elif defect == "wrong_latent_dtype":
        tensors["clean_latents"] = tensors["clean_latents"].to(torch.int16)
    elif defect == "bad_time_order":
        tensors["cell_time_index"] = torch.arange(4).repeat_interleave(12)
    elif defect == "wrong_time_dtype":
        tensors["cell_time_index"] = torch.arange(4).repeat(12).float()
    elif defect in {"wrong_rgb_dtype", "wrong_rgb_layout", "raw_rgb_count"}:
        tensors.pop("clean_latents")
        if defect == "wrong_rgb_dtype":
            tensors["frames"] = torch.zeros(3, 156, 2, 4)
        elif defect == "wrong_rgb_layout":
            tensors["frames"] = torch.zeros(156, 2, 4, 3, dtype=torch.uint8)
        else:
            tensors["world_to_camera"] = tensors.pop("camera_to_world")
            tensors["frames"] = torch.zeros(3, 156, 2, 4, dtype=torch.uint8)
            metadata["pose_space"] = "raw_world_to_camera"
    payload = save(tensors, metadata=metadata)
    with pytest.raises(ValueError, match=match):
        load_camera_archive(payload, _request(payload))


@pytest.mark.parametrize("defect", ("missing", "pose_space", "schema", "extra"))
def test_reader_requires_explicit_matching_schema_metadata(defect):
    archive = _prepared()
    metadata = _metadata(_dump(archive))
    if defect == "missing":
        metadata = None
    elif defect == "pose_space":
        metadata["pose_space"] = "raw_world_to_camera"
    elif defect == "schema":
        metadata["schema"] = "another.archive"
    else:
        metadata["undeclared"] = "value"
    payload = save(_tensors(archive), metadata=metadata)
    with pytest.raises(ValueError, match="metadata"):
        load_camera_archive(payload, _request(payload))


def test_reader_rejects_corrupt_serialization_and_wrong_publication_length():
    payload = _dump(_prepared())
    with pytest.raises(ValueError, match="byte count"):
        load_camera_archive(payload[:-1], _request(payload))
    # Even with a matching publication length, truncated tensor storage fails.
    truncated = payload[:-1]
    with pytest.raises(ValueError, match="safetensors data"):
        load_camera_archive(truncated, _request(truncated))
    for malformed in (b"", b"not a tensor archive", b"\x80\x04N."):
        with pytest.raises(ValueError):
            load_camera_archive(malformed, _request(malformed))
    duplicate = b'{"__metadata__":{},"__metadata__":{}}'
    malformed = len(duplicate).to_bytes(8, "little") + duplicate
    with pytest.raises(ValueError, match="JSON header"):
        load_camera_archive(malformed, _request(malformed))
