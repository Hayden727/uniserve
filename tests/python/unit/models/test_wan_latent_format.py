"""Latent storage is part of UCPE's native BF16 projection contract."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from uniserve.diffusion import ReNoiseInput
from uniserve.model.camera import CameraGrid
from uniserve_models.wan.config import Config, TransformerConfig, read_config
from uniserve_models.wan.denoiser import Denoiser
from uniserve_models.wan.inputs import LatentFormat, LatentSize
from uniserve_models.wan.model import Model
from uniserve_models.wan.ucpe import CameraAttention, CameraConfig, UcpeConfig

pytestmark = pytest.mark.unit


def _config(latent_format=None, *, grid=None, plucker=False):
    return Config(
        TransformerConfig(
            num_layers=1,
            num_heads=1,
            head_dim=8,
            ffn_dim=16,
            in_channels=2,
            out_channels=2,
            text_dim=8,
            freq_dim=8,
            use_scale_embedding=plucker,
        ),
        ucpe=None
        if plucker
        else UcpeConfig(CameraConfig(1, 4), latent_format=latent_format),
        grid=grid,
    )


def _camera(device):
    model = CameraAttention(2, CameraConfig(1, 4)).to(
        device=device, dtype=torch.bfloat16
    )
    for parameter in model.parameters():
        parameter.zero_()
    # A BF16 product of 257 rounds to 256 before separate bias addition.
    # Fused bias preserves one. Uniform attention preserves constant rows.
    model.v_proj.weight[0] = torch.tensor([256, 1], device=device)
    model.v_proj.bias[0] = -256
    model.out_proj.weight[0, 0] = 1
    return model


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config(), LatentFormat.CHANNELS_LAST_3D),
        (
            _config(grid=CameraGrid(32, 16, "time_major")),
            LatentFormat.CONTIGUOUS,
        ),
        (_config(LatentFormat.CONTIGUOUS), LatentFormat.CONTIGUOUS),
        (_config(plucker=True), None),
    ],
)
def test_serving_resolves_format_from_explicit_numerical_configuration(
    config, expected
):
    with torch.device("meta"):
        denoiser = Model(config).transformer
    assert denoiser.latent_format is expected


@pytest.mark.parametrize("latent_format", list(LatentFormat))
def test_export_configuration_retains_typed_latent_format(
    tmp_path, latent_format
):
    config = _config(latent_format)
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["WanCameraPipeline"], **asdict(config)})
    )
    assert read_config(tmp_path, None, sources=()) == config
    with pytest.raises(TypeError, match="typed LatentFormat"):
        replace(config.ucpe, latent_format=latent_format.value)
    with pytest.raises(ValueError, match="contiguous latent inputs"):
        _config(
            LatentFormat.CHANNELS_LAST_3D,
            grid=CameraGrid(32, 16, "time_major"),
        )


@torch.inference_mode()
@pytest.mark.parametrize("plucker", [False, True])
def test_direct_denoiser_keeps_caller_layout_without_an_explicit_contract(
    plucker,
):
    denoiser = Denoiser(_config(plucker=plucker))
    sample = torch.ones(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    mask = torch.tensor([True, False]).reshape(1, 1, 2, 1, 1)
    for latent_format in LatentFormat:
        clean = sample.contiguous(memory_format=latent_format.memory_format)
        actual = denoiser._network_input(
            sample, ReNoiseInput(sample, clean, mask)
        )
        expected = torch.where(mask, clean, sample)
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("latent_format", list(LatentFormat))
def test_declared_layout_makes_camera_output_independent_of_caller_strides(
    latent_format,
):
    denoiser = Denoiser(_config(latent_format))
    camera = _camera("cpu")
    sample = torch.ones(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    mask = torch.tensor([True, False]).reshape(1, 1, 2, 1, 1)
    poses = torch.eye(4, dtype=torch.bfloat16).repeat(1, 8, 1, 1)
    # Independent native recipe: pin the native-format clean tensor first,
    # then let camera nn.Linear behavior follow the actual B,S,D layout.
    native_clean = sample.contiguous(memory_format=latent_format.memory_format)
    native = torch.where(mask, native_clean, sample)
    expected = camera(native.flatten(2).transpose(1, 2), poses)
    assert expected[0, 0, 0] == (
        1 if latent_format is LatentFormat.CHANNELS_LAST_3D else 0
    )
    for sample_format in LatentFormat:
        for clean_format in LatentFormat:
            borrowed_sample = sample.contiguous(
                memory_format=sample_format.memory_format
            )
            borrowed_clean = sample.contiguous(
                memory_format=clean_format.memory_format
            )
            before = (borrowed_sample.stride(), borrowed_clean.stride())
            conditions = ReNoiseInput(sample, borrowed_clean, mask)
            current = denoiser._network_input(borrowed_sample, conditions)
            assert current.is_contiguous(
                memory_format=latent_format.memory_format
            )
            actual = camera(current.flatten(2).transpose(1, 2), poses)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert before == (borrowed_sample.stride(), borrowed_clean.stride())
            torch.testing.assert_close(borrowed_clean, sample, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("latent_format", list(LatentFormat))
def test_predict_x0_matches_native_layout_for_equal_request_values(
    latent_format,
):
    torch.manual_seed(253)
    denoiser = Denoiser(_config(latent_format)).bfloat16()
    for parameter in denoiser.parameters():
        parameter.normal_(std=0.1)
    sample = torch.randn(1, 2, 2, 4, 4).bfloat16()
    clean = torch.randn_like(sample)
    noise = torch.randn_like(sample)
    mask = torch.tensor([True, False]).reshape(1, 1, 2, 1, 1)
    context = torch.randn(1, 3, 8).bfloat16()
    viewmats = torch.eye(4, dtype=torch.bfloat16).repeat(1, 8, 1, 1)
    size = LatentSize(2, 4, 4, 3)
    constants = {
        name: torch.empty(field.shape, dtype=field.dtype)
        for name, field in denoiser.constant_buffers(size).items()
    }
    denoiser.prepare_constants(size, out=constants)
    rotary = denoiser.network.rotary_embedding(sample.shape[2:], device="cpu")
    schedule = denoiser.make_schedules(device="cpu")["video"]
    timestep = schedule.timesteps[1]
    native_clean = clean.contiguous(memory_format=latent_format.memory_format)
    velocity = denoiser.network(
        torch.where(mask, native_clean, sample),
        context,
        timestep.reshape(1),
        viewmats,
        rotary,
        time_frequencies=constants["time_frequencies"],
    )
    sigma = (timestep.float() / 1000).bfloat16()
    expected = torch.where(mask, clean, sample - sigma * velocity)
    for sample_format in LatentFormat:
        for clean_format in LatentFormat:
            inputs = denoiser.bind_ucpe_inputs(
                sample=sample.contiguous(
                    memory_format=sample_format.memory_format
                ),
                timestep=timestep,
                step=schedule.step(1),
                size=size,
                context=context,
                viewmats=viewmats,
                rotary=rotary,
                noise=noise,
                clean=clean.contiguous(
                    memory_format=clean_format.memory_format
                ),
                clean_mask=mask,
            )
            output = denoiser(
                inputs, state={}, constants=constants, workspace={}
            )
            torch.testing.assert_close(
                output["video"][0].tensor, expected, rtol=0, atol=0
            )


@torch.inference_mode()
@pytest.mark.parametrize("contiguous_tokens", [False, True])
def test_post_patch_layout_retains_native_single_rank_and_sharded_semantics(
    contiguous_tokens,
):
    # Native FastVideo bypasses sharding at SP1 and makes the selected token
    # slice contiguous at SP>1. Observe the real block input and camera Q input,
    # without replacing either computation or changing projection semantics.
    torch.manual_seed(132)
    network = Denoiser(_config(LatentFormat.CONTIGUOUS)).network.bfloat16()
    for parameter in network.parameters():
        parameter.normal_(std=0.1)
    sample = torch.randn(1, 2, 2, 4, 4).bfloat16()
    patch = network.patch_embedding(sample).flatten(2).transpose(1, 2)
    assert not patch.is_contiguous()
    observed = {}

    def record(name):
        def hook(module, args):
            observed[name] = args[0]

        return hook

    handles = [
        network.blocks[0].register_forward_pre_hook(record("block")),
        network.blocks[0].cam_self_attn.q_proj.register_forward_pre_hook(
            record("camera")
        ),
    ]
    try:
        network(
            sample,
            torch.randn(1, 3, 8).bfloat16(),
            torch.tensor([1000.0]),
            torch.eye(4, dtype=torch.bfloat16).repeat(1, 8, 1, 1),
            network.rotary_embedding(sample.shape[2:], device="cpu"),
            contiguous_tokens=contiguous_tokens,
        )
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(observed["block"], patch, rtol=0, atol=0)
    assert observed["block"].is_contiguous() is contiguous_tokens
    assert observed["camera"].is_contiguous() is contiguous_tokens


@pytest.mark.gpu
@torch.inference_mode()
@pytest.mark.parametrize("latent_format", list(LatentFormat))
def test_layout_conversion_replays_live_clean_state_in_a_cuda_graph(
    latent_format,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph replay requires CUDA")
    denoiser = Denoiser(_config(latent_format))
    sample = torch.ones(1, 2, 2, 2, 2, device="cuda", dtype=torch.bfloat16)
    clean = sample.clone()
    mask = torch.tensor([True, False], device="cuda").reshape(1, 1, 2, 1, 1)
    conditions = ReNoiseInput(sample, clean, mask)
    # The conversion owns only a graph temporary. Both request inputs stay
    # borrowed and change between replays, including the clean-cell mask.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            denoiser._network_input(sample, conditions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = denoiser._network_input(sample, conditions)
    for value in (2, 3, 2):
        clean.fill_(value)
        sample.fill_(-value)
        mask.logical_not_()
        graph.replay()
        expected = torch.where(mask, clean, sample)
        assert output.is_contiguous(memory_format=latent_format.memory_format)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
