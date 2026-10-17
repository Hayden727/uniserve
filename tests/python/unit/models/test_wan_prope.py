"""Persisted PRoPE arithmetic controls the public camera computation."""

import json
import struct
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from tests.python.unit.models.test_wan_ucpe import _camera_weights, _poses
from uniserve_models.wan.config import Config, TransformerConfig, read_config
from uniserve_models.wan.ucpe import CameraAttention, CameraConfig, UcpeConfig

pytestmark = pytest.mark.unit


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _scalar_projection(features, matrix):
    """Independent scalar oracle for four rounded FP32 products and sums."""
    result = torch.empty_like(features)
    for batch in range(features.shape[0]):
        for head in range(features.shape[1]):
            for token in range(features.shape[2]):
                for offset in range(0, features.shape[3], 4):
                    for row in range(4):
                        products = [
                            _float32(
                                _float32(matrix[batch, token, row, col].item())
                                * _float32(
                                    features[
                                        batch, head, token, offset + col
                                    ].item()
                                )
                            )
                            for col in range(4)
                        ]
                        total = products[0]
                        for product in products[1:]:
                            total = _float32(total + product)
                        result[batch, head, token, offset + row] = total
    return result


@torch.inference_mode()
@pytest.mark.parametrize("spatial_rope", [False, True])
def test_elementwise_camera_uses_sequential_fp32_for_all_pose_transforms(
    spatial_rope,
):
    width, heads, head_dim, tokens = 12, 2, 8, 3
    model = CameraAttention(
        width,
        CameraConfig(heads, head_dim, spatial_rope, projection="elementwise"),
    ).double()
    weights = _camera_weights(width, heads, head_dim)
    model.load_state_dict(weights, strict=True)
    hidden = torch.randn(
        1,
        tokens,
        width,
        generator=torch.Generator().manual_seed(23),
        dtype=torch.float64,
    )
    poses = _poses(tokens)
    inverse = torch.zeros_like(poses)
    inverse[..., :3, :3] = poses[..., :3, :3].transpose(-1, -2)
    inverse[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", inverse[..., :3, :3], poses[..., :3, 3]
    )
    inverse[..., 3, 3] = 1
    rotary = None
    if spatial_rope:
        angle = torch.arange(tokens, dtype=torch.float64).reshape(1, 1, -1, 1)
        rotary = tuple((a.cos(), a.sin()) for a in (angle * 0.2, angle * 0.3))

    def transform(value, matrix, *, output=False):
        if rotary is None:
            return _scalar_projection(value, matrix)
        pose, horizontal, vertical = value.split((4, 2, 2), dim=-1)
        parts = [_scalar_projection(pose, matrix)]
        for axis, (cosine, sine) in zip(
            (horizontal, vertical), rotary, strict=True
        ):
            left, right = axis.chunk(2, dim=-1)
            if output:
                parts.append(
                    torch.cat(
                        (
                            cosine * left - sine * right,
                            sine * left + cosine * right,
                        ),
                        -1,
                    )
                )
            else:
                parts.append(
                    torch.cat(
                        (
                            cosine * left + sine * right,
                            -sine * left + cosine * right,
                        ),
                        -1,
                    )
                )
        return torch.cat(parts, -1)

    q, k, v = (
        F.linear(hidden, weights[name + ".weight"], weights[name + ".bias"])
        .reshape(1, tokens, heads, head_dim)
        .transpose(1, 2)
        for name in ("q_proj", "k_proj", "v_proj")
    )
    q = transform(q, poses.transpose(-1, -2))
    k, v = (transform(value, inverse) for value in (k, v))
    attended = F.scaled_dot_product_attention(q, k, v)
    attended = (
        transform(attended, poses, output=True).transpose(1, 2).flatten(2)
    )
    expected = F.linear(
        attended, weights["out_proj.weight"], weights["out_proj.bias"]
    )
    # Spatial concatenation can select a different FP64 SDPA layout. The
    # existing relative-pose test's FP64 tolerance is far below FP32 rounding.
    tolerance = 1e-12 if spatial_rope else 0
    torch.testing.assert_close(
        model(hidden, poses, rotary), expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("projection", ["einsum", "elementwise"])
def test_pipeline_round_trip_preserves_camera_projection(tmp_path, projection):
    config = Config(
        TransformerConfig(use_scale_embedding=False),
        ucpe=UcpeConfig(CameraConfig(projection=projection)),
    )
    values = {"architectures": ["WanCameraPipeline"], **asdict(config)}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    assert read_config(tmp_path, None, sources=()) == config
    # Existing STDM and diagnostic pipeline configs omit the new field.
    del values["ucpe"]["camera"]["projection"]
    path.write_text(json.dumps(values))
    assert (
        read_config(tmp_path, None, sources=()).ucpe.camera.projection
        == "einsum"
    )


@pytest.mark.parametrize("projection", ["triton", "", None, 1])
def test_camera_config_rejects_unknown_projection(projection):
    with pytest.raises(ValueError, match="projection"):
        CameraConfig(projection=projection)
