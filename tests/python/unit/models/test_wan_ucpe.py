"""UCPE relative-pose semantics and invocation-local camera conditioning.

Run this file directly under a configured native FastVideo environment for
the complete small-DiT comparison. That diagnostic explicitly selects a
common SDPA backend and is separate from optimized-checkpoint acceptance.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from uniserve_models.wan.config import TransformerConfig
from uniserve_models.wan.ucpe import (
    CameraAttention,
    CameraConfig,
    UcpeTransformer,
)

pytestmark = pytest.mark.unit


def _poses(tokens, *, device="cpu", dtype=torch.float64):
    angles = torch.arange(tokens, device=device, dtype=torch.float64) * 0.2
    poses = torch.eye(4, device=device, dtype=torch.float64).repeat(
        1, tokens, 1, 1
    )
    poses[0, :, 0, 0] = angles.cos()
    poses[0, :, 0, 2] = angles.sin()
    poses[0, :, 2, 0] = -angles.sin()
    poses[0, :, 2, 2] = angles.cos()
    poses[0, :, :3, 3] = torch.stack((angles, -angles * 0.7, angles * 0.4), -1)
    return poses.to(dtype)


def _camera_weights(width, heads, head_dim):
    generator = torch.Generator().manual_seed(871)
    compressed = heads * head_dim
    values = {}
    for name, rows, columns in (
        ("q_proj", compressed, width),
        ("k_proj", compressed, width),
        ("v_proj", compressed, width),
        ("out_proj", width, compressed),
    ):
        values[name + ".weight"] = (
            torch.randn(rows, columns, generator=generator, dtype=torch.float64)
            * 0.2
        )
        values[name + ".bias"] = (
            torch.randn(rows, generator=generator, dtype=torch.float64) * 0.1
        )
    return values


def _relative_attention(hidden, poses, weights, heads, head_dim):
    """Independent oracle: attend through each query/key relative transform.

    Instead of encoding Q/K/V then decoding output, transport key and value
    vectors directly into each query ray's frame using P_i @ inverse(P_j).
    """
    tokens = hidden.shape[1]
    projected = [
        F.linear(
            hidden, weights[name + ".weight"], weights[name + ".bias"]
        ).reshape(tokens, heads, head_dim // 4, 4)
        for name in ("q_proj", "k_proj", "v_proj")
    ]
    q, k, v = projected
    rows = []
    for query in range(tokens):
        scores, values = [], []
        for key in range(tokens):
            relative = poses[0, query] @ torch.linalg.inv(poses[0, key])
            key_in_query = k[key] @ relative.T
            scores.append((q[query] * key_in_query).sum((-1, -2)))
            values.append(v[key] @ relative.T)
        probabilities = (
            torch.stack(scores, dim=1).div(math.sqrt(head_dim)).softmax(-1)
        )
        transported = torch.stack(values, dim=1)
        rows.append((probabilities[..., None, None] * transported).sum(1))
    attended = torch.stack(rows).reshape(1, tokens, heads * head_dim)
    return F.linear(
        attended, weights["out_proj.weight"], weights["out_proj.bias"]
    )


@torch.inference_mode()
def test_camera_attention_matches_relative_pose_transport():
    width, heads, head_dim, tokens = 12, 2, 8, 5
    weights = _camera_weights(width, heads, head_dim)
    model = CameraAttention(width, CameraConfig(heads, head_dim)).double()
    model.load_state_dict(weights, strict=True)
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(
        1, tokens, width, generator=generator, dtype=torch.float64
    )
    poses = _poses(tokens)
    expected = _relative_attention(hidden, poses, weights, heads, head_dim)
    actual = model(hidden, poses)
    # FP64 arithmetic over five keys and groups of four: this checks the
    # mathematical transform, not the lower-precision model quality gate.
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@torch.inference_mode()
def test_camera_projection_bias_rounding_follows_native_input_layout():
    model = CameraAttention(2, CameraConfig(1, 4)).to(torch.bfloat16)
    for parameter in model.parameters():
        parameter.zero_()
    # Zero Q/K make attention uniform. Equal value rows preserve the exact
    # projected values, so the camera output exposes both rounding points.
    model.v_proj.weight[0] = torch.tensor([256, 1])
    model.v_proj.bias[0] = -256
    model.v_proj.bias[1:3] = 1
    model.out_proj.weight[0, 0] = 1
    model.out_proj.weight[1, 1:3] = torch.tensor([256, 1])
    model.out_proj.bias[1] = -256
    poses = torch.eye(4, dtype=torch.bfloat16).repeat(1, 2, 1, 1)
    hidden = torch.ones(1, 2, 2, dtype=torch.bfloat16).transpose(1, 2)

    # Native nn.Linear separates bias for noncontiguous B,S,D input. Its
    # contiguous camera output projection fuses bias and retains the unit.
    expected = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.bfloat16)
    torch.testing.assert_close(model(hidden, poses), expected, rtol=0, atol=0)
    expected[..., 0] = 1
    torch.testing.assert_close(
        model(hidden.contiguous(), poses), expected, rtol=0, atol=0
    )


def test_camera_attention_rejects_misaligned_tokens_and_missing_rope():
    model = CameraAttention(16, CameraConfig(2, 8))
    hidden = torch.zeros(1, 3, 16)
    with pytest.raises(ValueError, match="align"):
        model(hidden, _poses(2, dtype=torch.float32))
    spatial = CameraAttention(16, CameraConfig(2, 8, use_2d_rope=True))
    with pytest.raises(ValueError, match="spatial RoPE"):
        spatial(hidden, _poses(3, dtype=torch.float32))


@torch.inference_mode()
def test_transformer_camera_requests_are_reentrant():
    config = TransformerConfig(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        ffn_dim=32,
        in_channels=4,
        out_channels=4,
        text_dim=12,
        freq_dim=8,
        use_scale_embedding=False,
    )
    torch.manual_seed(217)
    model = UcpeTransformer(config, CameraConfig(1, 8))
    for name, parameter in model.named_parameters():
        parameter.normal_(std=0.15)
        if "norm" in name and name.endswith("weight"):
            parameter.add_(1)
    latents = torch.randn(1, 4, 2, 4, 4)
    context = torch.randn(1, 5, 12)
    timestep = torch.tensor([625.0])
    rotary = model.rotary_embedding(latents.shape[2:], device="cpu")
    first = _poses(8, dtype=torch.float32)
    second = first.flip(1).contiguous()
    outputs = [
        model(latents, context, timestep, poses, rotary)
        for poses in (first, second, first)
    ]
    assert outputs[0].shape == latents.shape
    assert all(torch.isfinite(value).all() for value in outputs)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    assert not torch.equal(outputs[0], outputs[1])


def test_cell_timestamps_preserve_simultaneous_camera_positions():
    model = UcpeTransformer(
        TransformerConfig(num_layers=1, num_heads=2, head_dim=8, ffn_dim=16),
        CameraConfig(1, 8),
    )
    sequential = model.rotary_embedding((3, 4, 4), device="cpu")
    grid = model.rotary_embedding(
        (3, 4, 4), device="cpu", cell_time_index=torch.tensor([2, 0, 2])
    )
    for actual, expected in zip(grid, sequential, strict=True):
        torch.testing.assert_close(
            actual.reshape(3, 4, 8),
            expected.reshape(3, 4, 8)[[2, 0, 2]],
            rtol=0,
            atol=0,
        )


def _comparison_taps(model, *, native):
    """Retain only small diagnostic outputs to locate the first divergence."""
    values = {}
    handles = []
    paths = {
        "patch_embedding.proj",
        "condition_embedder.time_embedder.mlp.fc_in",
        "condition_embedder.time_embedder.mlp.fc_out",
        "condition_embedder.time_modulation.linear",
        "condition_embedder.text_embedder.fc_in",
        "condition_embedder.text_embedder.fc_out",
        "proj_out",
    }
    block_paths = (
        "norm1",
        "to_q",
        "norm_q",
        "to_k",
        "norm_k",
        "to_v",
        "attn1",
        "to_out",
        "cam_self_attn.q_proj",
        "cam_self_attn.k_proj",
        "cam_self_attn.v_proj",
        "cam_self_attn.out_proj",
        "cam_self_attn",
        "self_attn_residual_norm.norm",
        "attn2",
        "cross_attn_residual_norm.norm",
        "ffn.fc_in",
        "ffn.fc_out",
    )
    paths.update(
        f"blocks.{index}.{name}" for index in range(2) for name in block_paths
    )

    def capture(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if native and name.endswith(".attn1"):
                tensor = tensor.transpose(1, 2)
            values[name] = tensor.detach().clone()

        return hook

    def capture_input(name):
        def hook(module, inputs):
            # Preserve the native strides: F.linear uses them to choose
            # between fused addmm and a separately rounded matmul + bias.
            values[name + ".input"] = inputs[0].detach().clone()

        return hook

    for name, module in model.named_modules():
        if name in paths:
            if ".cam_self_attn." in name:
                handles.append(
                    module.register_forward_pre_hook(capture_input(name))
                )
            handles.append(module.register_forward_hook(capture(name)))
    return values, handles


def _report_first_difference(expected, actual):
    import json

    for name, value in expected.items():
        if name not in actual:
            continue
        candidate = actual[name]
        if torch.equal(value, candidate):
            continue
        report = {
            "first_differing_module": name,
            "native_shape": list(value.shape),
            "candidate_shape": list(candidate.shape),
            "native_stride": list(value.stride()),
            "candidate_stride": list(candidate.stride()),
        }
        if value.shape == candidate.shape:
            difference = (value.float() - candidate.float()).abs()
            report.update(
                max_abs=difference.max().item(),
                mean_abs=difference.mean().item(),
                mismatched=(value != candidate).sum().item(),
            )
        print(json.dumps(report), flush=True)
        return name
    print(
        "No difference at recorded modules; inspect rotary, residual "
        "and output modulation",
        flush=True,
    )


def _report_camera_projection(name, native, candidate, expected, actual):
    """Distinguish differing inputs, GEMM paths and bias-rounding semantics."""
    import json

    if name is None or ".cam_self_attn." not in name:
        return
    name = name.removesuffix(".input")
    if name + ".input" not in expected:
        return
    reference = native.get_submodule(name)
    projection = candidate.get_submodule(name)
    native_input = expected[name + ".input"]
    candidate_input = actual[name + ".input"]
    weight, bias = reference.weight, reference.bias

    def difference(left, right):
        delta = (left.float() - right.float()).abs()
        return {
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "mismatched": (left != right).sum().item(),
        }

    matrix = native_input.reshape(-1, native_input.shape[-1])
    fused = torch.addmm(bias, matrix, weight.T).reshape(expected[name].shape)
    separate = (native_input @ weight.T).add_(bias)
    native_linear = F.linear(native_input, weight, bias)
    candidate_linear = F.linear(candidate_input, weight, bias)
    report = {
        "camera_projection": name,
        "native_input_stride": list(native_input.stride()),
        "candidate_input_stride": list(candidate_input.stride()),
        "input_difference": difference(native_input, candidate_input),
        "weight_difference": difference(weight, projection.weight),
        "bias_difference": difference(bias, projection.bias),
        "native_vs_native_linear": difference(expected[name], native_linear),
        "native_vs_fused": difference(expected[name], fused),
        "native_vs_separate": difference(expected[name], separate),
        "candidate_vs_native_linear": difference(
            actual[name], candidate_linear
        ),
        "candidate_vs_fused": difference(actual[name], fused),
        "candidate_vs_separate": difference(actual[name], separate),
    }
    print(json.dumps(report), flush=True)


def _native_parity():
    """Compare complete BF16 DiTs, including nonzero learned camera outputs."""
    import os

    from fastvideo.configs.models.dits.wanvideo import (
        WanVideoArchConfig,
        WanVideoConfig,
    )
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel
    from research.videogen.infra_fastvideo.models import prope
    from research.videogen.infra_fastvideo.models.ucpe_dit import (
        upgrade_to_ucpe,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from uniserve.model import TextSize
    from uniserve.runtime.execution import ExecutionContext

    if os.environ.get("IMP_PROPE_PROJMAT", "einsum") != "einsum":
        raise ValueError(
            "the common-backend diagnostic requires IMP_PROPE_PROJMAT=einsum"
        )
    if os.environ.get("ATTN_INFERENCE_BACKEND", "").lower() in {
        "sage2",
        "sage3",
    }:
        raise ValueError("the common-backend diagnostic requires dense SDPA")
    # The native camera helper otherwise selects FA4 independently of the main
    # attention backend. This declared diagnostic fixes both to torch SDPA.
    prope.FLASH_ATTN_4_AVAILABLE = False
    prope.FLASH_ATTN_3_AVAILABLE = False
    prope.FLASH_ATTN_2_AVAILABLE = False
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        for spatial_rope in (False, True):
            torch.manual_seed(2026)
            arch = WanVideoArchConfig(
                num_layers=2,
                num_attention_heads=4,
                attention_head_dim=32,
                ffn_dim=192,
                text_dim=32,
                freq_dim=16,
            )
            native = WanTransformer3DModel(
                WanVideoConfig(arch_config=arch), {}
            ).to(device="cuda", dtype=torch.bfloat16)
            upgrade_to_ucpe(
                native,
                attn_compress=2,
                patches_x=4,
                patches_y=2,
                image_width=64,
                image_height=32,
                use_2d_rope=spatial_rope,
            )
            config = TransformerConfig(
                num_layers=2,
                num_heads=4,
                head_dim=32,
                ffn_dim=192,
                text_dim=32,
                freq_dim=16,
                use_scale_embedding=False,
                fused_modulation=os.environ.get("IMP_FUSED_MODULATE", "0")
                == "1",
            )
            candidate = UcpeTransformer(
                config, CameraConfig(2, 32, spatial_rope)
            ).to(device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                # FastVideo inference layers allocate empty weights. Initialize
                # every native parameter, including the camera output weights.
                for name, parameter in native.named_parameters():
                    parameter.normal_(std=0.1)
                    if "norm" in name and name.endswith("weight"):
                        parameter.add_(1)
            candidate.load_state_dict(native.state_dict(), strict=True)
            native_taps, native_handles = _comparison_taps(native, native=True)
            candidate_taps, candidate_handles = _comparison_taps(
                candidate, native=False
            )
            latents = torch.randn(
                1, 16, 2, 4, 8, device="cuda", dtype=torch.bfloat16
            )
            context = torch.randn(1, 9, 32, device="cuda", dtype=torch.bfloat16)
            first = _poses(16, device="cuda", dtype=torch.bfloat16)
            cameras = (first, first.flip(1).contiguous(), first)
            camera_rotary = candidate.camera_rotary_embedding(
                latents.shape[2:], device="cuda", dtype=torch.bfloat16
            )
            for cell_times in (None, torch.tensor([0, 0], device="cuda")):
                rotary = candidate.rotary_embedding(
                    latents.shape[2:], device="cuda", cell_time_index=cell_times
                )
                outputs = []
                with (
                    torch.inference_mode(),
                    set_forward_context(
                        current_timestep=0,
                        attn_metadata=None,
                        forward_batch=None,
                    ),
                    ExecutionContext(
                        candidate, attention="torch", matmul="torch"
                    ) as execution,
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                        ]
                    ),
                ):
                    execution.prepare(TextSize(rotary[0].shape[0], 1))
                    for poses in cameras:
                        for time in (1000.0, 937.5, 833.3333333333334, 625.0):
                            timestep = torch.tensor([time], device="cuda")
                            payload = {"viewmats": poses}
                            if cell_times is not None:
                                payload["cell_time_index"] = cell_times
                            expected = native(
                                latents,
                                context,
                                timestep,
                                control_camera_dit_input=payload,
                            )
                            actual = candidate(
                                latents,
                                context,
                                timestep,
                                poses,
                                rotary,
                                camera_rotary,
                            )
                            assert (
                                torch.isfinite(expected).all()
                                and torch.isfinite(actual).all()
                            )
                            try:
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                            except AssertionError:
                                name = _report_first_difference(
                                    native_taps, candidate_taps
                                )
                                _report_camera_projection(
                                    name,
                                    native,
                                    candidate,
                                    native_taps,
                                    candidate_taps,
                                )
                                raise
                            if time == 625.0:
                                outputs.append(actual.clone())
                torch.testing.assert_close(
                    outputs[0], outputs[2], rtol=0, atol=0
                )
                assert not torch.equal(outputs[0], outputs[1])
            print(
                f"PASS: UCPE common-SDPA small DiT, spatial_rope={spatial_rope}"
            )
            for handle in (*native_handles, *candidate_handles):
                handle.remove()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _native_parity()
