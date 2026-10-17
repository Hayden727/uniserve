"""Wan UCPE computation with explicit per-token world-to-ray transforms.

The separate camera residual follows imp_genai's ``UcpeWanBlock``. PRoPE
uses column-vector SE(3) transforms on each consecutive group of four head
features; camera and rotary inputs belong to the invocation, not the model.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from uniserve.nn.attention import Attention, AttentionBatch, DenseInput
from uniserve.nn.attention.grid import GridAttention, GridLayout
from uniserve.nn.attention.vsa.layer import BlockAttention
from uniserve.nn.attention.vsa.tiles import TileAttention, TileLayout
from uniserve.nn.convolution import Conv3d
from uniserve.nn.linear import Linear
from uniserve.nn.mlp import MLP
from uniserve.nn.norm import LayerNorm, RMSNorm
from uniserve.quantization.nvfp4 import fold_gate

from . import fusion
from .config import TransformerConfig, VsaConfig
from .inputs import LatentFormat
from .transformer import (
    ConditionEmbedding,
    CrossAttention,
    _norm_container,
    rotate,
)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Compressed camera-head dimensions, resolved from checkpoint metadata.

    Native UCPE uses ``main_heads // attn_compress`` camera heads and keeps
    the main head dimension. The 14B Quality checkpoint has 10 heads of 128
    features, with no spatial RoPE in this separate branch.
    Projection selects either a GEMM contraction or native sequential FP32
    products and additions, rounded back to the camera feature dtype.
    """

    num_heads: int = 10
    head_dim: int = 128
    use_2d_rope: bool = False
    compress_gate: bool = False
    vsa: VsaConfig | None = None
    projection: Literal["einsum", "elementwise"] = "einsum"

    def __post_init__(self):
        if self.projection not in ("einsum", "elementwise"):
            raise ValueError("UCPE projection must be einsum or elementwise")
        if self.vsa is not None:
            if not isinstance(self.vsa, VsaConfig):
                raise TypeError(
                    "UCPE sparse attention requires a typed VsaConfig"
                )
            if not self.compress_gate:
                raise ValueError(
                    "UCPE VSA requires the trained camera compression gate"
                )
        if any(
            type(value) is not int or value < 1
            for value in (self.num_heads, self.head_dim)
        ):
            raise ValueError("UCPE camera dimensions must be positive integers")
        divisor = 8 if self.use_2d_rope else 4
        if self.head_dim % divisor:
            raise ValueError(
                f"UCPE camera head dimension must divide by {divisor}"
            )

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class UcpeConfig:
    """UCPE camera architecture and retained main compression-gate weights.

    Presence of trained compression gates describes the checkpoint. Quality
    evaluates dense attention and leaves these gate projections inactive.
    A declared latent format fixes patch-input storage and its downstream
    projection rounding. None preserves direct inputs; serving resolves it
    from the orbit or rectangular-grid data contract.
    """

    camera: CameraConfig = CameraConfig()
    main_compress_gate: bool = False
    latent_format: LatentFormat | None = None

    def __post_init__(self):
        if self.latent_format is not None and not isinstance(
            self.latent_format, LatentFormat
        ):
            raise TypeError("UCPE latent format requires a typed LatentFormat")


def _inverse_se3(viewmats):
    rotation = viewmats[..., :3, :3].transpose(-1, -2)
    inverse = torch.zeros_like(viewmats)
    inverse[..., :3, :3] = rotation
    inverse[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", rotation, viewmats[..., :3, 3]
    )
    inverse[..., 3, 3] = 1
    return inverse


def _project(features, matrix):
    grouped = features.unflatten(-1, (-1, 4))
    result = torch.einsum("btij,bntpj->bntpi", matrix, grouped)
    return result.reshape(features.shape)


def _project_elementwise(features, matrix):
    # Native PRoPE rounds each product and sequential addition in FP32,
    # then casts the complete four-term contraction to the feature dtype.
    # Reassociation or a fused GEMM changes the checkpoint's numerical recipe.
    batch, heads, tokens, width = features.shape
    grouped = features.reshape(batch, heads, tokens, width // 4, 4).float()
    matrix = matrix.float()
    rows = []
    for row in range(4):
        value = (
            matrix[:, :, row, 0].reshape(batch, 1, tokens, 1) * grouped[..., 0]
        )
        for column in range(1, 4):
            value = (
                value
                + matrix[:, :, row, column].reshape(batch, 1, tokens, 1)
                * grouped[..., column]
            )
        rows.append(value)
    return torch.stack(rows, dim=-1).reshape(features.shape).to(features.dtype)


def _spatial_rotate(features, rotary, *, inverse=False):
    # PRoPE uses split halves and the opposite sign to main interleaved RoPE.
    cosine, sine = rotary
    left, right = features.chunk(2, dim=-1)
    if inverse:
        return torch.cat(
            (cosine * left - sine * right, sine * left + cosine * right), -1
        )
    return torch.cat(
        (cosine * left + sine * right, -sine * left + cosine * right), -1
    )


def _transform(features, matrix, rotary, *, inverse=False, projection="einsum"):
    project = _project_elementwise if projection == "elementwise" else _project
    if rotary is None:
        # Native block-diagonal concatenation also makes the result contiguous.
        return project(features, matrix).contiguous()
    width = features.shape[-1]
    pose, horizontal, vertical = features.split(
        (width // 2, width // 4, width // 4), dim=-1
    )
    return torch.cat(
        (
            project(pose, matrix),
            _spatial_rotate(horizontal, rotary[0], inverse=inverse),
            _spatial_rotate(vertical, rotary[1], inverse=inverse),
        ),
        dim=-1,
    )


def _camera_projection(layer, value):
    # Native nn.Linear changes its BF16 bias rounding for noncontiguous
    # B,S,D input. An encoded projection has its own explicit epilogue and
    # must not inherit this dense-only hint.
    if type(layer) is Linear:
        return layer(value, round_before_bias=not value.is_contiguous())
    return layer(value)


def bias_project(product, bias, matrix, heads):
    """Unbiased BF16 product, FP32 bias and PRoPE, one BF16 store."""
    value = (
        (product.float() + bias.float())
        .unflatten(-1, (heads, -1))
        .transpose(1, 2)
    )
    return _project_elementwise(value, matrix).to(product.dtype).contiguous()


class CameraAttention(nn.Module):
    """Compressed Q/K/V attention with per-ray PRoPE and output unprojection.

    ``viewmats`` is B,S,4,4 in exactly the same token order as B,S,D hidden
    states. Each matrix maps world coordinates into its ray's local frame.
    Optional spatial rotary coefficients are two (cos, sin) pairs, each
    shaped 1,1,S,head_dim/8 and rounded to the projection dtype beforehand.
    """

    def __init__(
        self,
        width: int,
        config: CameraConfig,
        *,
        grid_attention=False,
        fuse_projection=False,
    ):
        super().__init__()
        self.config = config
        if fuse_projection and (
            grid_attention
            or config.use_2d_rope
            or config.vsa is not None
            or config.projection != "elementwise"
        ):
            raise ValueError(
                "fused camera projections require "
                "dense elementwise orbit cameras"
            )
        self.fuse_projection = fuse_projection
        self.q_proj = Linear(width, config.hidden_size)
        self.k_proj = Linear(width, config.hidden_size)
        self.v_proj = Linear(width, config.hidden_size)
        self.out_proj = Linear(config.hidden_size, width)
        # The joint Quality/Fast student contains trained sparse-attention
        # gates. Dense Quality retains their weights without applying them.
        self.to_gate_compress = (
            Linear(width, config.hidden_size) if config.compress_gate else None
        )
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.attn = (
            GridAttention(config.num_heads, config.head_dim)
            if grid_attention
            else Attention(config.num_heads, config.num_heads, config.head_dim)
            if config.vsa is None
            else TileAttention(
                BlockAttention(config.head_dim**-0.5, recipe=config.vsa.recipe),
                num_heads=config.num_heads,
                head_dim=config.head_dim,
            )
        )

    def forward(
        self,
        hidden,
        viewmats,
        rotary=None,
        *,
        attention=None,
        tiles: TileLayout | None = None,
        grid: GridLayout | None = None,
    ):
        if viewmats.shape != (*hidden.shape[:2], 4, 4):
            raise ValueError(
                "UCPE view matrices must align with every latent token"
            )
        if self.config.use_2d_rope != (rotary is not None):
            raise ValueError(
                "UCPE spatial RoPE must match the camera configuration"
            )
        if rotary is not None:
            shape = (1, 1, hidden.shape[1], self.config.head_dim // 8)
            if (
                len(rotary) != 2
                or any(len(axis) != 2 for axis in rotary)
                or any(
                    value.shape != shape
                    or value.dtype != hidden.dtype
                    or value.device != hidden.device
                    for axis in rotary
                    for value in axis
                )
            ):
                raise ValueError(
                    "UCPE spatial RoPE must match token count and dtype"
                )
        head_shape = (self.config.num_heads, self.config.head_dim)
        # Native camera projections are nn.Linear: contiguous B,S,D input
        # fuses bias, while a noncontiguous residual rounds GEMM first. Keep
        # this choice before shared Linear flattens the logical row axes.
        if self.fuse_projection:
            pose = viewmats.to(hidden.dtype)
            inverse = _inverse_se3(pose)
            q, k, v = (
                fusion.compiled(bias_project)(
                    layer(hidden, include_bias=False),
                    layer.bias,
                    matrix,
                    self.config.num_heads,
                )
                for layer, matrix in (
                    (self.q_proj, pose.transpose(-1, -2)),
                    (self.k_proj, inverse),
                    (self.v_proj, inverse),
                )
            )
        else:
            q, k, v = self._project(hidden, viewmats, rotary)
            pose = viewmats.to(q.dtype)
        if isinstance(self.attn, GridAttention):
            if grid is None or hidden.shape[0] != 1 or attention is not None:
                raise ValueError(
                    "grid camera attention requires one explicit global grid"
                )
            attended = (
                self.attn(
                    q[0].transpose(0, 1),
                    k[0].transpose(0, 1),
                    v[0].transpose(0, 1),
                    grid,
                )
                .transpose(0, 1)
                .unsqueeze(0)
            )
        elif self.config.vsa is None:
            if tiles is not None:
                raise ValueError(
                    "dense UCPE attention does not consume VSA tiles"
                )
            if attention is None:
                attention = AttentionBatch.single(DenseInput(False, None))
            attended = self.attn(q, k, v, attention)
        else:
            if attention is not None:
                raise ValueError(
                    "UCPE VSA requires tile connectivity, not a dense mask"
                )
            if tiles is None or tiles.tile_shape != self.config.vsa.tile_shape:
                raise ValueError("UCPE VSA requires its configured tile layout")
            gate = _camera_projection(self.to_gate_compress, hidden).unflatten(
                -1, head_shape
            )
            attended = torch.stack(
                [
                    self.attn(
                        qi,
                        ki,
                        vi,
                        gi,
                        tiles,
                        keep=tiles.keep(self.config.vsa.sparsity),
                    )
                    for qi, ki, vi, gi in zip(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        gate,
                        strict=True,
                    )
                ]
            ).transpose(1, 2)
        attended = _transform(
            attended,
            pose,
            rotary,
            inverse=True,
            projection=self.config.projection,
        )
        attended = attended.transpose(1, 2).flatten(2)
        return _camera_projection(self.out_proj, attended)

    def _project(self, hidden, viewmats, rotary):
        head_shape = (self.config.num_heads, self.config.head_dim)
        q = (
            _camera_projection(self.q_proj, hidden)
            .unflatten(-1, head_shape)
            .transpose(1, 2)
        )
        k = (
            _camera_projection(self.k_proj, hidden)
            .unflatten(-1, head_shape)
            .transpose(1, 2)
        )
        v = (
            _camera_projection(self.v_proj, hidden)
            .unflatten(-1, head_shape)
            .transpose(1, 2)
        )
        # Native inversion happens after rounding the poses to projection dtype.
        pose = viewmats.to(q.dtype)
        inverse = _inverse_se3(pose)
        projection = self.config.projection
        q = _transform(q, pose.transpose(-1, -2), rotary, projection=projection)
        k = _transform(k, inverse, rotary, projection=projection)
        v = _transform(v, inverse, rotary, projection=projection)
        return q, k, v


class UcpeBlock(nn.Module):
    """Gated Wan self-attention, camera residual, text attention, then MLP."""

    def __init__(
        self,
        config: TransformerConfig,
        camera: CameraConfig,
        *,
        main_compress_gate: bool = False,
        grid_attention: bool = False,
    ):
        super().__init__()
        if config.vsa is not None and not main_compress_gate:
            raise ValueError(
                "UCPE VSA requires the trained main compression gate"
            )
        self.config = config
        self.grid_attention = grid_attention
        # Weight-derived constants set once after encoding, never request state.
        self.output_alpha = 1.0
        self.ffn_alpha = 1.0
        width = config.hidden_size
        self.norm1 = LayerNorm(width, eps=config.eps, elementwise_affine=False)
        self.to_q, self.to_k = Linear(width, width), Linear(width, width)
        self.to_v, self.to_out = Linear(width, width), Linear(width, width)
        self.to_gate_compress = (
            Linear(width, width) if main_compress_gate else None
        )
        self.norm_q = RMSNorm(
            width,
            config.eps,
            round_before_scale=True,
            reduction_order=config.rms_reduction_order,
        )
        self.norm_k = RMSNorm(
            width,
            config.eps,
            round_before_scale=True,
            reduction_order=config.rms_reduction_order,
        )
        self.attn1 = (
            GridAttention(config.num_heads, config.head_dim)
            if grid_attention and config.vsa is None
            else Attention(config.num_heads, config.num_heads, config.head_dim)
            if config.vsa is None
            else TileAttention(
                BlockAttention(config.head_dim**-0.5, recipe=config.vsa.recipe),
                num_heads=config.num_heads,
                head_dim=config.head_dim,
            )
        )
        self.cam_self_attn = CameraAttention(
            width,
            camera,
            grid_attention=grid_attention,
            fuse_projection=config.fusion_recipe == "ucpe_prefix",
        )
        self.self_attn_residual_norm = _norm_container(
            width, config.eps, affine=True
        )
        self.attn2 = CrossAttention(config)
        self.cross_attn_residual_norm = _norm_container(
            width, config.eps, affine=False
        )
        self.ffn = MLP(
            width, config.ffn_dim, width, activation=nn.GELU(approximate="tanh")
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, width) / width**0.5, requires_grad=False
        )

    def _norm_modulate(self, norm, hidden, scale, shift):
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.layer_norm_modulate)(
                hidden, scale, shift, norm.eps, self.config.fused_modulation
            )
        value = norm(hidden)
        if self.config.fused_modulation:
            return torch.addcmul(shift, value, 1 + scale)
        return value * (1 + scale) + shift

    def _residual(self, hidden, gate, value):
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.gated_residual)(
                hidden, gate, value, self.config.fused_modulation
            )
        if self.config.fused_modulation:
            return torch.addcmul(hidden, gate, value)
        return hidden + gate * value

    def _qk(self, norm, projected, rotary):
        """Full-width RMSNorm, head split and RoPE: (B,S,W) -> (B,S,H,D)."""
        if self.config.fusion_recipe == "ucpe_prefix":
            # The original UCPE RMS callable returns BF16 before main RoPE.
            normalized = fusion.compiled(fusion.rms_scale)(
                projected, norm.weight, norm.eps
            )
            return rotate(
                normalized.unflatten(-1, (self.config.num_heads, -1)), rotary
            )
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.rms_rotate)(
                projected, norm.weight, norm.eps, self.config.num_heads, *rotary
            )
        head_shape = (self.config.num_heads, self.config.head_dim)
        return rotate(norm(projected).unflatten(-1, head_shape), rotary)

    def _affine_norm(self, container, hidden):
        norm = container.norm
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.layer_norm_affine)(
                hidden, norm.weight, norm.bias, norm.eps
            )
        return container(hidden)

    def forward(
        self,
        hidden,
        context,
        temb,
        rotary,
        viewmats,
        camera_rotary=None,
        *,
        attention=None,
        camera_attention=None,
        tiles: TileLayout | None = None,
        camera_tiles: TileLayout | None = None,
        grid: GridLayout | None = None,
        tile_allowed: torch.Tensor | None = None,
        tile_keep: int | None = None,
    ):
        if self.grid_attention and (
            grid is None
            or (
                self.config.vsa is not None
                and (tile_allowed is None or tile_keep is None)
            )
        ):
            raise ValueError(
                "grid attention requires explicit global connectivity "
                "and its structural VSA budget"
            )
        modulation = (self.scale_shift_table + temb).to(hidden.dtype)
        shift, scale, gate, ff_shift, ff_scale, ff_gate = modulation.chunk(
            6, dim=1
        )
        value = self._norm_modulate(self.norm1, hidden, scale, shift)
        head_shape = (self.config.num_heads, self.config.head_dim)
        q = self._qk(self.norm_q, self.to_q(value), rotary)
        k = self._qk(self.norm_k, self.to_k(value), rotary)
        v = self.to_v(value).unflatten(-1, head_shape)
        text_attention = AttentionBatch.single(DenseInput(False, None))
        if isinstance(self.attn1, GridAttention):
            if grid is None or hidden.shape[0] != 1 or attention is not None:
                raise ValueError(
                    "grid main attention requires one explicit global grid"
                )
            attended = self.attn1(q[0], k[0], v[0], grid).unsqueeze(0)
        elif self.config.vsa is None:
            if tiles is not None:
                raise ValueError(
                    "dense Wan attention does not consume VSA tiles"
                )
            if attention is None:
                attention = text_attention
            attended = self.attn1(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attention,
            ).transpose(1, 2)
        else:
            if attention is not None:
                raise ValueError(
                    "Wan VSA requires tile connectivity, not a dense mask"
                )
            if tiles is None or tiles.tile_shape != self.config.vsa.tile_shape:
                raise ValueError("Wan VSA requires its configured tile layout")
            gate_compress = self.to_gate_compress(value).unflatten(
                -1, head_shape
            )
            attended = torch.stack(
                [
                    self.attn1(
                        qi,
                        ki,
                        vi,
                        gi,
                        tiles,
                        keep=tiles.keep(self.config.vsa.sparsity)
                        if tile_keep is None
                        else tile_keep,
                        allowed=tile_allowed,
                    )
                    for qi, ki, vi, gi in zip(
                        q, k, v, gate_compress, strict=True
                    )
                ]
            )
        hidden = self._residual(
            hidden,
            fold_gate(gate, self.output_alpha),
            self.to_out(attended.flatten(2)),
        )
        # Camera QKV read the updated residual, rather than the modulated
        # input that produced the main attention QKV.
        hidden = hidden + self.cam_self_attn(
            hidden,
            viewmats,
            camera_rotary,
            attention=camera_attention,
            tiles=camera_tiles,
            grid=grid,
        )
        hidden = hidden + self.attn2(
            self._affine_norm(self.self_attn_residual_norm, hidden),
            context,
            text_attention,
        )
        value = self._norm_modulate(
            self.cross_attn_residual_norm.norm, hidden, ff_scale, ff_shift
        )
        return self._residual(
            hidden, fold_gate(ff_gate, self.ffn_alpha), self.ffn(value)
        )


class UcpeTransformer(nn.Module):
    """Wan velocity prediction with a distinct UCPE camera-attention branch.

    Latents use B,C,T,H,W order. Camera matrices follow the patchified T,H,W
    token order. Network timesteps are one FP32 value per sample (1000*sigma).
    Rotary tensors are prepared outside graph capture and borrowed per call.
    Dense attention is the default; explicit masks may independently define
    the main and camera attended sets without changing text cross-attention.
    """

    def __init__(
        self,
        config: TransformerConfig,
        camera: CameraConfig,
        *,
        main_compress_gate: bool = False,
        grid_attention: bool = False,
    ):
        super().__init__()
        self.config, self.camera_config = config, camera
        width = config.hidden_size
        self.patch_embedding = nn.Sequential(
            OrderedDict(
                proj=Conv3d(
                    config.in_channels,
                    width,
                    kernel_size=config.patch_size,
                    stride=config.patch_size,
                    backend=config.patch_convolution,
                )
            )
        )
        self.condition_embedder = ConditionEmbedding(config)
        self.blocks = nn.ModuleList(
            UcpeBlock(
                config,
                camera,
                main_compress_gate=main_compress_gate,
                grid_attention=grid_attention,
            )
            for _ in range(config.num_layers)
        )
        self.norm_out = _norm_container(width, config.eps, affine=False)
        self.proj_out = Linear(
            width, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, width) / width**0.5, requires_grad=False
        )

    def rotary_embedding(self, latent_size, *, device, cell_time_index=None):
        """Prepare native FP64 3D RoPE with optional explicit cell timestamps.

        ``cell_time_index`` has one nonnegative integer per patchified frame.
        Equal timestamps give simultaneous camera cells equal temporal phases.
        """
        sizes = tuple(
            size // patch
            for size, patch in zip(
                latent_size, self.config.patch_size, strict=True
            )
        )
        axes = [
            torch.arange(size, dtype=torch.float32, device="cpu")
            for size in sizes
        ]
        if cell_time_index is not None:
            if (
                cell_time_index.shape != (sizes[0],)
                or cell_time_index.dtype not in (torch.int32, torch.int64)
                or bool((cell_time_index < 0).any())
            ):
                raise ValueError(
                    "cell timestamps must be nonnegative integers per frame"
                )
            axes[0] = cell_time_index.to(device="cpu", dtype=torch.float32)
        positions = torch.meshgrid(*axes, indexing="ij")
        head = self.config.head_dim
        dimensions = (head - 4 * (head // 6), 2 * (head // 6), 2 * (head // 6))
        cosines, sines = [], []
        for position, dim in zip(positions, dimensions, strict=True):
            frequencies = 1.0 / (
                10000 ** (torch.arange(0, dim, 2, device="cpu").double() / dim)
            )
            angles = torch.outer(position.flatten(), frequencies)
            cosines.append(angles.cos().repeat_interleave(2, dim=-1))
            sines.append(angles.sin().repeat_interleave(2, dim=-1))
        return tuple(
            torch.cat(values, dim=-1).to(device=device, dtype=torch.float32)
            for values in (cosines, sines)
        )

    def camera_rotary_embedding(self, latent_size, *, device, dtype):
        """Prepare optional native spatial PRoPE coefficients in token order."""
        if not self.camera_config.use_2d_rope:
            return None
        frames, height, width = (
            size // patch
            for size, patch in zip(
                latent_size, self.config.patch_size, strict=True
            )
        )
        x = torch.arange(width, device="cpu").repeat(height * frames)
        y = (
            torch.arange(height, device="cpu")
            .repeat_interleave(width)
            .repeat(frames)
        )
        frequencies = self.camera_config.head_dim // 8
        coefficients = 100.0 ** (
            -torch.arange(frequencies, device="cpu") / frequencies
        )
        result = []
        for positions in (x, y):
            angles = (
                positions[None, None, :, None] * coefficients[None, None, None]
            )
            result.append(
                tuple(
                    value.to(device=device, dtype=dtype)
                    for value in (angles.cos(), angles.sin())
                )
            )
        return tuple(result)

    def forward(
        self,
        latents,
        context,
        timestep,
        viewmats,
        rotary,
        camera_rotary=None,
        *,
        time_frequencies=None,
        attention=None,
        camera_attention=None,
        tiles: TileLayout | None = None,
        camera_tiles: TileLayout | None = None,
        grid: GridLayout | None = None,
        tile_allowed: torch.Tensor | None = None,
        tile_keep: int | None = None,
        contiguous_tokens: bool = False,
    ):
        batch, channels, frames, height, width = latents.shape
        pt, ph, pw = self.config.patch_size
        if channels != self.config.in_channels or any(
            size % patch
            for size, patch in zip(
                latents.shape[2:], self.config.patch_size, strict=True
            )
        ):
            raise ValueError(
                "Wan latents must match channels and patch divisibility"
            )
        if timestep.shape != (batch,):
            raise ValueError("Wan requires one network timestep per sample")
        hidden = self.patch_embedding(latents).flatten(2).transpose(1, 2)
        if contiguous_tokens:
            # Native SP>1 materializes the post-patch token shard; SP1 returns
            # the original view. Local-cell patching must retain that boundary
            # because camera nn.Linear bias fusion depends on these strides.
            hidden = hidden.contiguous()
        time, modulation, context = self.condition_embedder(
            timestep, context, time_frequencies
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                context,
                modulation,
                rotary,
                viewmats,
                camera_rotary,
                attention=attention,
                camera_attention=camera_attention,
                tiles=tiles,
                camera_tiles=camera_tiles,
                grid=grid,
                tile_allowed=tile_allowed,
                tile_keep=tile_keep,
            )
        shift, scale = (self.scale_shift_table + time.unsqueeze(1)).chunk(
            2, dim=1
        )
        hidden = (self.norm_out(hidden).float() * (1 + scale) + shift).to(
            hidden.dtype
        )
        hidden = self.proj_out(hidden).reshape(
            batch,
            frames // pt,
            height // ph,
            width // pw,
            pt,
            ph,
            pw,
            self.config.out_channels,
        )
        return (
            hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )
