"""Camera archives stage native clean tensors through real worker entries."""

import pytest
import torch

from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.media import image
from uniserve_models.wan.config import Config, TransformerConfig
from uniserve_models.wan.model import Model, entry_points
from uniserve_models.wan.serving import VideoSize
from uniserve_models.wan.vae import VAEConfig
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.camera import (
    encode_camera,
    prepare_camera,
    stage_camera,
)
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.media import reader
from uniserve_worker.media.camera import CameraArchive, dump_camera_archive
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.protocol.video import CameraInput, MediaLocator

pytestmark = pytest.mark.integration

# The numerical bridge accepts the capability's size directly. Small spatial
# dimensions keep the real VAE integration independent of serving admission.
SIZE = VideoSize(26, image.Config(16, 32), 3)
CLEAN_CELLS = (0, 4)


def _runner(*, encoder=True):
    vae = VAEConfig(base_dim=8, num_res_blocks=1)
    config = Config(
        TransformerConfig(
            num_layers=1,
            num_heads=2,
            head_dim=8,
            ffn_dim=24,
            text_dim=12,
            freq_dim=8,
            scale_bands=2,
        ),
        video_encoder=vae if encoder else None,
        video_decoder=vae,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(16)
        model = Model(config)
    if encoder:
        model.video_encoder.encoder.bfloat16()
    entries = dict(entry_points(config))
    if encoder:
        # Component names belong to deployment, independently of module paths.
        entries["condition_frames"] = entries.pop("video_encoder")
    components = {
        name: ComponentConfig(
            (0,),
            distribution="temporal_units" if name == "video_decoder" else None,
            units_per_rank=1,
        )
        for name in entries
    }
    group = Communicator(device=torch.device("cpu"))
    bindings = {
        name: ComponentBinding(
            name,
            component,
            group,
            DeviceMesh(
                ranks=component.ranks,
                rank=0,
                shape=tuple(
                    count for _, count in component.parallel_config.dimensions
                ),
                axes=tuple(
                    axis for axis, _ in component.parallel_config.dimensions
                ),
            ),
            group.device,
        )
        for name, component in components.items()
    }
    return ModelExecutor(
        model,
        cpu_worker_config(
            max_model_len=3,
            max_video_seconds=26 / 24,
            graph_policy="off",
            deployment_components=tuple(components),
        ),
        bindings=bindings,
        entry_points=entries,
    )


@pytest.fixture
def runner():
    value = _runner()
    try:
        yield value
    finally:
        value.close()


def _publish(tmp_path, monkeypatch, *, frames=None, clean_latents=None):
    archive = CameraArchive(
        intrinsics=torch.tensor(
            [[40.0, 0.0, 16.0], [0.0, 40.0, 8.0], [0.0, 0.0, 1.0]]
        ),
        camera_to_world=torch.eye(4).repeat(SIZE.frames, 1, 1),
        scene_scale=torch.tensor([2.0]),
        frames=frames,
        clean_latents=clean_latents,
    )
    payload = dump_camera_archive(
        archive,
        views=2,
        columns=4,
        order="view_major",
        clean_cells=CLEAN_CELLS,
    )
    # Redirect only the external filesystem root. The public reader still
    # loads and validates the bytes advertised by the actual media locator.
    monkeypatch.setattr(reader, "SHARED_MEMORY", tmp_path)
    (tmp_path / "camera.safetensors").write_bytes(payload)
    return CameraInput(
        MediaLocator("camera.safetensors", len(payload)),
        2,
        4,
        "view_major",
        CLEAN_CELLS,
    )


def _sources(runner):
    return {
        f"{name}_source": torch.empty(field.shape, dtype=field.dtype)
        for name, field in runner.media_builder.denoiser.camera_buffers(
            SIZE
        ).items()
    }


def _rgb():
    values = torch.arange(3 * 26 * 16 * 32, dtype=torch.int64)
    return ((values * 17 + 127) % 256).to(torch.uint8).reshape(3, 26, 16, 32)


def test_precomputed_camera_latents_preserve_values_and_clean_cells(
    runner, tmp_path, monkeypatch
):
    clean = torch.arange(16 * 8 * 2 * 4).reshape(16, 8, 2, 4).bfloat16()
    camera = _publish(tmp_path, monkeypatch, clean_latents=clean)
    tensors = _sources(runner)
    before = torch.random.get_rng_state()
    archive = stage_camera(runner.media_builder, SIZE, tensors, camera)
    assert encode_camera(runner, SIZE, camera, archive, tensors) is None
    torch.testing.assert_close(
        tensors["clean_source"], clean.unsqueeze(0), rtol=0, atol=0
    )
    assert tensors["clean_mask_source"].flatten().tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert torch.equal(torch.random.get_rng_state(), before)


def test_rgb_camera_encoding_preserves_independent_clip_order(
    runner, tmp_path, monkeypatch
):
    frames = _rgb()
    camera = _publish(tmp_path, monkeypatch, frames=frames)
    tensors = _sources(runner)
    archive = stage_camera(runner.media_builder, SIZE, tensors, camera)

    # Independently encode each native 13-frame segment through the executor.
    # Concatenating their public row outputs supplies a reference for the
    # camera bridge's full-video native C,F,H,W storage.
    reference = []
    for first in (0, 13):
        pixels = frames[:, first : first + 13].permute(1, 2, 3, 0).contiguous()
        result = runner.run_module(
            "condition_frames",
            (pixels,),
            method="encode",
            size=(pixels.shape,),
            clips=(slice(first, first + 13),),
            num_frames=(26,),
        )
        reference.append(result.values[0])
    expected = (
        torch.cat(reference)
        .reshape(8, 2, 4, 16)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
    )

    before = torch.random.get_rng_state()
    output = encode_camera(runner, SIZE, camera, archive, tensors)
    assert output is not None and output.stats is not None
    torch.testing.assert_close(
        tensors["clean_source"], expected, rtol=0, atol=0
    )
    tensors.update(
        {
            name.removesuffix("_source"): torch.empty_like(value)
            for name, value in tensors.items()
        }
    )
    prepare_camera(runner, SIZE, camera, archive, tensors)
    torch.testing.assert_close(tensors["clean"], expected, rtol=0, atol=0)
    assert torch.equal(tensors["clean_mask"], tensors["clean_mask_source"])
    assert torch.equal(archive.frames, frames)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_rgb_without_resident_encoder_reports_placement_requirement(
    tmp_path, monkeypatch
):
    runner = _runner(encoder=False)
    try:
        camera = _publish(tmp_path, monkeypatch, frames=_rgb())
        tensors = _sources(runner)
        archive = stage_camera(runner.media_builder, SIZE, tensors, camera)
        with pytest.raises(WorkerError, match="local VideoEncoder"):
            encode_camera(runner, SIZE, camera, archive, tensors)
    finally:
        runner.close()
