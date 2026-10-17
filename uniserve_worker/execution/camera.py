"""Stage camera archives and encode their RGB through worker-owned entries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.media import video
from uniserve.model import VideoEncoder
from uniserve.model.camera import CameraCondition, CameraDenoiser, CameraGrid
from uniserve.model.denoiser import VideoSize
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.media.camera import CameraArchive, load_camera_archive
from uniserve_worker.media.camera_result import CameraResult
from uniserve_worker.media.reader import read_bytes
from uniserve_worker.model_executor.output import ExecutionOutput
from uniserve_worker.protocol.video import CameraInput

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.media_inputs import MediaBuilder
    from uniserve_worker.protocol.batch import NewRequest


def camera_condition(
    archive: CameraArchive, camera: CameraInput
) -> CameraCondition:
    """Borrow archive tensors in the request's complete latent-cell order.

    Archive loading checks serialized metadata against the request. This
    boundary preserves that grid for numerical preparation and validates
    cell-domain shapes without deriving cells from the raw pose count.
    """
    raw = archive.world_to_camera is not None
    poses = archive.world_to_camera if raw else archive.camera_to_world
    if poses is None:
        raise invalid_descriptor("camera archive has no pose tensor")
    return CameraCondition(
        poses=poses,
        intrinsics=archive.intrinsics,
        clean_cells=camera.clean_cells,
        pose_space=(
            "raw_world_to_camera" if raw else "prepared_camera_to_world"
        ),
        clean_latents=archive.clean_latents,
        scene_scale=archive.scene_scale,
        cell_time_index=archive.cell_time_index,
        grid=CameraGrid(camera.views, camera.columns, camera.order),
    )


def _denoiser(builder: MediaBuilder | None) -> CameraDenoiser:
    if builder is None or not isinstance(builder.denoiser, CameraDenoiser):
        raise invalid_descriptor(
            "camera input requires a camera-conditioned denoiser"
        )
    return builder.denoiser


@torch.inference_mode()
def stage_camera(
    builder: MediaBuilder,
    size: VideoSize,
    tensors: Mapping[str, torch.Tensor],
    camera: CameraInput,
) -> CameraArchive:
    """Read the published archive and fill this request's host sources.

    Called after ordinary state/noise staging, before initialization copies
    host sources into retained device state. Returning the archive keeps its
    RGB input available for optional encoder execution on the service thread.
    Archive reading and camera preparation perform no sampling.
    """
    denoiser = _denoiser(builder)
    archive = load_camera_archive(read_bytes(camera.source), camera)
    denoiser.prepare_camera(
        size,
        camera_condition(archive, camera),
        out={
            name: tensors[f"{name}_source"]
            for name in denoiser.camera_buffers(size)
        },
    )
    return archive


@torch.inference_mode()
def prepare_camera(
    runner: ModelExecutor,
    size: VideoSize,
    camera: CameraInput,
    archive: CameraArchive,
    tensors: Mapping[str, torch.Tensor],
) -> None:
    """Prepare camera math on the device chosen by the request's buffers.

    Called in the worker's input preparation context after initial staging
    and optional RGB encoding. Raw pose normalization remains CPU work, but
    rays and scale features use the destination device's native arithmetic.
    The already staged clean source supplies either archive or online VAE
    latents, so preparing the camera fields cannot erase encoded clean slots.
    """
    denoiser = _denoiser(runner.media_builder)
    condition = camera_condition(archive, camera)
    if condition.clean_latents is None:
        # Online encoding fills complete ring latents. Prepared archives
        # retain their global clean source even when device state is sharded.
        condition = replace(condition, clean_latents=tensors["clean_source"][0])
    denoiser.prepare_camera(
        size,
        condition,
        out={name: tensors[name] for name in denoiser.camera_buffers(size)},
    )


@torch.inference_mode()
def camera_result(
    runner: ModelExecutor, admission: NewRequest, latents: torch.Tensor
) -> CameraResult:
    """Borrow a gathered final latent and canonical camera metadata on the host.

    Host workers retain the weightless model description, whose camera
    metadata capability performs CPU tensor math without inference weights.
    The tensor-store read stays owned by the pending host call until its
    serialization task finishes; this helper neither copies nor publishes it.
    """
    denoiser = _denoiser(runner.media_builder)
    media, video_input = admission.diffusion, admission.video
    if media is None or video_input is None or video_input.camera is None:
        raise invalid_descriptor(
            "camera result requires its admitted camera input"
        )
    camera = video_input.camera
    archive = load_camera_archive(read_bytes(camera.source), camera)
    size = denoiser.make_camera_size(
        CameraGrid(camera.views, camera.columns, camera.order),
        media.num_frames,
        len(admission.prompt_token_ids),
        canvas=media.canvas,
    )
    layout = denoiser.output_layout(size)["video"]
    if latents.shape != layout.shape or latents.dtype != layout.dtype:
        raise invalid_descriptor(
            "camera result must preserve the complete native latent"
        )
    return CameraResult(
        latents=latents,
        original=archive,
        canonical=denoiser.output_camera(
            size, camera_condition(archive, camera)
        ),
        camera=camera,
        seed=media.seed,
        num_frames=media.num_frames,
        canvas=media.canvas,
        camera_video=denoiser.camera_video,
        frame_rate=runner.video_postprocessor.frame_rate,
    )


@torch.inference_mode()
def encode_camera(
    runner: ModelExecutor,
    size: VideoSize,
    camera: CameraInput,
    archive: CameraArchive,
    tensors: Mapping[str, torch.Tensor],
) -> ExecutionOutput | None:
    """Encode selected RGB clips through a resident VideoEncoder entry.

    The encoder declares independent clip boundaries and packs their latent
    rows. A single numerical call receives every clip, preserving its shared
    per-video posterior behavior. The helper draws no noise, changes no RGB
    values, and uses the entry's ordinary prepared context and graph policy.

    Packed rows are copied synchronously to CPU and stored in clean_source
    before MediaBuilder.initialize stages it. Returns the actual execution
    output and its statistics, or None when the archive already has latents.

    Online encoding currently requires one encoder placed on this rank alone.
    A remote or collective encoder needs a scheduled input-product dependency;
    it cannot be invoked implicitly inside this rank's latent preparation.
    """
    if archive.frames is None:
        return None
    denoiser = _denoiser(runner.media_builder)
    encoders = [
        (binding, call)
        for binding in runner.bindings.values()
        for call in binding.calls
        if call.entry_point.method == "encode"
        and isinstance(call.module, VideoEncoder)
    ]
    if len(encoders) != 1:
        raise unsupported_setup(
            "online camera RGB requires exactly one local VideoEncoder; "
            "an encoder on another rank requires scheduled latent inputs"
        )
    binding, call = encoders[0]
    if (
        not binding.owns
        or len(binding.config.ranks) != 1
        or any(group.size != 1 for group in call.groups)
    ):
        raise unsupported_setup(
            "online camera RGB requires a VideoEncoder placed on this rank "
            "alone; distributed encoding requires scheduled latent inputs"
        )
    encoder = call.module
    frames = archive.frames
    if frames.shape[-2:] != (size.canvas.height, size.canvas.width):
        raise invalid_descriptor(
            "camera RGB raster must match the admitted output canvas"
        )
    indices = denoiser.input_frame_indices(
        size, camera_condition(archive, camera), frames.shape[1]
    )
    if not indices or any(
        type(index) is not int or not 0 <= index < frames.shape[1]
        for index in indices
    ):
        raise invalid_descriptor(
            "camera frame selection lies outside its RGB input"
        )
    selected = frames.index_select(
        1, frames.new_tensor(indices, dtype=torch.int64)
    ).permute(1, 2, 3, 0)
    count = len(indices)
    clips = encoder.clips(count)
    # The archive stores native channel-first RGB. VideoEncoder's public
    # boundary uses frame-major RGB24; this permutation changes no values.
    pixels = tuple(
        selected[clip].contiguous().to(binding.device) for clip in clips
    )
    output = runner.run_module(
        binding.name,
        pixels,
        method="encode",
        path=call.path,
        size=tuple(value.shape for value in pixels),
        clips=clips,
        num_frames=(count,) * len(clips),
    )
    if output.stats is None:
        raise RuntimeError("camera encoder returned no execution statistics")
    if len(output.values) != len(clips):
        raise invalid_descriptor(
            "camera encoder must return one row tensor per clip"
        )
    rows = torch.cat(output.values, dim=0)
    expected = encoder.output_layout(video.Config(count, size.canvas))["video"]
    if rows.shape != expected.shape or rows.dtype != expected.dtype:
        raise invalid_descriptor(
            "camera encoder rows do not cover its declared video"
        )
    clean = tensors["clean_source"]
    if clean.device.type != "cpu":
        raise invalid_descriptor(
            "camera clean_source must remain a CPU staging tensor"
        )
    # run_module joins the encoder stream to the caller's stream and returns
    # owned outputs. The blocking CPU copy completes before initialization
    # reads this source; no retained graph output or manual fence is needed.
    denoiser.store_clean(size, rows.cpu(), out=clean)
    return output
