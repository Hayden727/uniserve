"""Published camera results preserve native tensors and reusable metadata."""

import io
import json
import zipfile
from dataclasses import replace

import av
import numpy as np
import pytest
import torch
from safetensors.torch import load

from uniserve.media import image
from uniserve.model.camera import CameraCondition
from uniserve_models.wan.config import Config, TransformerConfig
from uniserve_models.wan.serving import ServingDenoiser, VideoSize
from uniserve_worker.execution.host import HostLane
from uniserve_worker.media.camera import CameraArchive
from uniserve_worker.media.camera_result import CameraResult, dump_camera_result
from uniserve_worker.media.container import AvMuxConfig, encode_video_unit
from uniserve_worker.media.mux import MediaMux
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.video import CameraInput, MediaLocator

pytestmark = pytest.mark.unit


def _result(*, include_video=False):
    frames = 48
    angle = torch.arange(160, dtype=torch.float32) * (2 * torch.pi / 160)
    poses = torch.eye(4).repeat(160, 1, 1)
    poses[:, 0, 3], poses[:, 2, 3] = angle.cos(), angle.sin()
    intrinsics = torch.tensor(
        [[40.0, 0.0, 16.0], [0.0, 40.0, 8.0], [0.0, 0.0, 1.0]]
    )
    clean = (
        torch.arange(16 * frames * 2 * 4).reshape(16, frames, 2, 4).bfloat16()
    )
    original = CameraArchive(
        intrinsics=intrinsics,
        world_to_camera=poses,
        scene_scale=torch.tensor([2.0]),
        clean_latents=clean,
    )
    condition = CameraCondition(
        poses,
        intrinsics,
        (0, 47),
        "raw_world_to_camera",
        clean_latents=clean,
        scene_scale=original.scene_scale,
    )
    # Host workers retain this weightless description. Canonical metadata
    # must be computed solely from CPU inputs, with no parameter access.
    with torch.device("meta"):
        denoiser = ServingDenoiser(
            Config(
                TransformerConfig(
                    num_layers=1,
                    num_heads=2,
                    head_dim=8,
                    ffn_dim=24,
                    text_dim=12,
                    freq_dim=8,
                    scale_bands=2,
                )
            )
        )
    size = VideoSize(156, image.Config(16, 32), 3)
    canonical = denoiser.output_camera(size, condition)
    latents = clean.unsqueeze(0).clone()
    latents[:, :, 1:47] *= 2
    return CameraResult(
        latents,
        original,
        canonical,
        CameraInput(
            MediaLocator("source", 8),
            12,
            4,
            "view_major",
            (0, 47),
            include_video,
        ),
        2**64 - 1,
        156,
        size.canvas,
    )


def _unpack(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return (
            load(archive.read("result.safetensors")),
            json.loads(archive.read("metadata.json")),
            archive.read("video.mp4")
            if "video.mp4" in archive.namelist()
            else None,
        )


def test_native_bf16_and_raw_and_canonical_camera_metadata_round_trip():
    result = _result()
    tensors, metadata, preview = _unpack(dump_camera_result(result))
    assert preview is None
    for name, expected in (
        ("latents", result.latents),
        ("camera_to_world", result.canonical.poses),
        ("intrinsics", result.canonical.intrinsics),
        ("input_world_to_camera", result.original.world_to_camera),
        ("input_intrinsics", result.original.intrinsics),
        ("scene_scale", result.original.scene_scale),
    ):
        assert tensors[name].dtype == expected.dtype
        torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)
    assert tensors["latents"].shape == (1, 16, 48, 2, 4)
    assert metadata["seed"] == 2**64 - 1
    assert metadata["grid"] == {
        "views": 12,
        "columns": 4,
        "order": "view_major",
    }
    assert metadata["clean_cells"] == [0, 47]
    assert metadata["input_pose_space"] == "raw_world_to_camera"
    assert metadata["camera_pose_space"] == "prepared_camera_to_world"
    assert metadata["latents"]["dtype"] == "bfloat16"
    assert metadata["latents"]["shape"] == list(result.latents.shape)


def test_export_refuses_changed_clean_cells_and_missing_requested_preview():
    result = _result()
    damaged = result.latents.clone()
    damaged[0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="clean cell"):
        dump_camera_result(replace(result, latents=damaged))
    with pytest.raises(ValueError, match="include_video"):
        dump_camera_result(_result(include_video=True))


def test_host_result_component_reserves_the_complete_remote_native_latent():
    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.bootstrap.capacity import local_product_storage_bytes
    from uniserve_worker.config.deployment import ComponentConfig
    from uniserve_worker.model_executor.component_binding import (
        ComponentBinding,
    )
    from uniserve_worker.protocol.call import MediaCall
    from uniserve_worker.protocol.tensor import (
        DType,
        OutputInfo,
        ShapeBound,
        StaticDim,
    )

    native = _result().latents
    outputs = {
        "denoiser": (
            OutputInfo(
                "video_latents",
                DType.BF16,
                ShapeBound(tuple(StaticDim(size) for size in native.shape)),
            ),
        )
    }
    bindings = {
        "result": ComponentBinding(
            "result",
            ComponentConfig((0,)),
            Communicator(),
            None,
            torch.device("cpu"),
        )
    }
    routes = {
        MediaCall.DENOISING: "denoiser",
        MediaCall.MUXING: "result",
    }
    assert (
        local_product_storage_bytes(
            outputs, bindings=bindings, media_components=routes
        )
        == 0
    )
    assert (
        local_product_storage_bytes(
            outputs,
            bindings=bindings,
            media_components=routes,
            camera_outputs=True,
        )
        == native.numel() * native.element_size()
    )


def test_host_worker_advertises_space_for_native_latents_and_preview():
    from tests.python.fixtures.worker_config import cpu_worker_config
    from uniserve.distributed.mesh import Communicator
    from uniserve_models.wan.model import Model
    from uniserve_models.wan.vae import VAEConfig
    from uniserve_worker.bootstrap.report import build_worker_layout
    from uniserve_worker.config.deployment import ComponentConfig
    from uniserve_worker.model_executor.component_binding import (
        ComponentBinding,
    )

    vae = VAEConfig(base_dim=8, num_res_blocks=1)
    with torch.device("meta"):
        model = Model(
            Config(
                TransformerConfig(
                    num_layers=1,
                    num_heads=2,
                    head_dim=8,
                    ffn_dim=24,
                    text_dim=12,
                    freq_dim=8,
                ),
                video_encoder=vae,
                video_decoder=vae,
            )
        )
    held = ("media_reader", "video_codec", "muxer")
    components = tuple((name, ComponentConfig((0,))) for name in held)
    group = Communicator(device=torch.device("cpu"))
    bindings = {
        name: ComponentBinding(name, config, group, None, group.device)
        for name, config in components
    }
    config = cpu_worker_config(
        max_video_seconds=13 / 24,
        max_model_len=512,
        max_request_pool_size=1,
        graph_policy="off",
        deployment_components=(
            "transformer",
            "video_encoder",
            "video_decoder",
            *held,
        ),
    )
    info = build_worker_layout(
        model, config, components=components, bindings=bindings, queue_depth=6
    ).info
    encoded = next(
        output.max_bytes
        for component in info.components
        for output in component.outputs
        if output.name == "encoded_units"
    )
    # The host codec consumes every RGB frame, and the bundle assembler
    # also imports the final native latent. Both coexist with encoded bytes.
    rgb_bytes = 13 * 480 * 832 * 3
    latent_bytes = 16 * 4 * 60 * 104 * 2
    assert info.buffer_pool_bytes >= rgb_bytes + latent_bytes + encoded


def test_host_mux_returns_bundle_with_real_video_before_publication():
    result = _result(include_video=True)
    config = AvMuxConfig(
        width=32,
        height=16,
        frame_count=result.num_frames,
        frame_rate=24,
        audio_rate=0,
        video_unit_frames=(result.num_frames,),
    )
    frames = np.zeros((result.num_frames, 16, 32, 3), dtype=np.uint8)
    frames[..., 1] = 255
    unit = encode_video_unit(config, frames)
    request, call = RequestKey(1, 7, 0), CallId(1, 0)
    lane, mux = HostLane(max_inflight=1, workers=1), MediaMux(rank=0)

    def run(schedule):
        task = schedule(lane.reserve())
        task.submit_if_ready()
        return task.promise.result(timeout=30)

    try:
        mux.open(request, config=config)
        run(
            lambda reservation: mux.append_units(
                request, (unit,), reservation, call
            )
        )
        payload = run(
            lambda reservation: mux.finalize_artifact(
                request,
                reservation,
                call,
                camera=result,
            )
        )
    finally:
        mux.close()
        lane.close()
    # Runtime materialization owns publication. Even a completed host task
    # may be abandoned, and must therefore retain only reclaimable bytes.
    assert isinstance(payload, bytes)
    tensors, metadata, preview = _unpack(payload)
    torch.testing.assert_close(
        tensors["latents"], result.latents, rtol=0, atol=0
    )
    assert metadata["video"]["included"]
    with av.open(io.BytesIO(preview)) as container:
        decoded = list(container.decode(video=0))
    assert len(decoded) == result.num_frames
    assert decoded[0].to_ndarray(format="rgb24").mean((0, 1)).argmax() == 1
