"""Wan transformer computation with explicit per-invocation camera tensors.

Arithmetic and checkpoint names follow imp_genai's FastVideo Wan/Plücker
modules. Attention and projections use UniServe's shared computation layers;
resource bindings and graph ownership stay with the caller.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.nn.attention import Attention, AttentionBatch, DenseInput
from uniserve.nn.attention.vsa.layer import BlockAttention
from uniserve.nn.attention.vsa.tiles import TileAttention, TileLayout
from uniserve.nn.convolution import Conv3d
from uniserve.nn.linear import Linear
from uniserve.nn.mlp import MLP
from uniserve.nn.norm import LayerNorm, RMSNorm

from . import fusion, residual
from .config import TransformerConfig


def rotate(x, rotary):
    """Apply interleaved-pair RoPE in FP32, then round to projection dtype."""
    cos, sin = rotary
    real, imaginary = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-imaginary, real), dim=-1).flatten(-2)
    return (x.float() * cos[:, None, :] + rotated * sin[:, None, :]).to(x.dtype)


def _norm_container(width, eps, *, affine):
    # Retain the native checkpoint path ending in `.norm` without carrying
    # the training runtime's residual/attention state.
    return nn.Sequential(
        OrderedDict(norm=LayerNorm(width, eps=eps, elementwise_affine=affine))
    )


class CrossAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        width = config.hidden_size
        self.heads, self.head_dim = config.num_heads, config.head_dim
        self.compile_elementwise = config.compile_elementwise
        self.fuse_projection_bias = config.fusion_recipe in {
            "plucker_prefix",
            "ucpe_prefix",
        }
        self.to_q, self.to_k = Linear(width, width), Linear(width, width)
        self.to_v, self.to_out = Linear(width, width), Linear(width, width)
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
        self.attn = Attention(
            config.num_heads,
            config.num_heads,
            config.head_dim,
            replicated_kv=True,
        )

    def _rms(self, norm, x):
        if self.compile_elementwise:
            return fusion.compiled(fusion.rms_scale)(x, norm.weight, norm.eps)
        return norm(x)

    def forward(self, x, context, attention, *, include_bias=True):
        q = self._project_qk(self.to_q, self.norm_q, x).unflatten(
            -1, (self.heads, self.head_dim)
        )
        k = self._project_qk(self.to_k, self.norm_k, context).unflatten(
            -1, (self.heads, self.head_dim)
        )
        v = self.to_v(context).unflatten(-1, (self.heads, self.head_dim))
        result = self.attn(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attention
        )
        value = result.transpose(1, 2).flatten(2)
        # Quantized projections retain their configured fused epilogue.
        # Only the dense tail requests a separately accumulated bias.
        if include_bias:
            return self.to_out(value)
        return self.to_out(value, include_bias=False)

    def _project_qk(self, layer, norm, value):
        if self.fuse_projection_bias:
            return fusion.compiled(fusion.bias_rms)(
                layer(value, include_bias=False),
                layer.bias,
                norm.weight,
                norm.eps,
            )
        return self._rms(norm, layer(value))


class PluckerBlock(nn.Module):
    """AdaLN → Plücker features → QKV, followed by text attention and MLP."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.norm1 = LayerNorm(width, eps=config.eps, elementwise_affine=False)
        self.to_q, self.to_k = Linear(width, width), Linear(width, width)
        self.to_v, self.to_out = Linear(width, width), Linear(width, width)
        self.to_gate_compress = Linear(width, width) if config.vsa else None
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
            Attention(config.num_heads, config.num_heads, config.head_dim)
            if config.vsa is None
            else TileAttention(
                BlockAttention(config.head_dim**-0.5, recipe=config.vsa.recipe),
                num_heads=config.num_heads,
                head_dim=config.head_dim,
            )
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
            torch.empty(1, 6, width), requires_grad=False
        )
        self.cam_encoder = nn.Linear(6, width)
        self.scale_mlp = (
            nn.Sequential(
                nn.Linear(2 * config.scale_bands, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            if config.use_scale_embedding
            else None
        )

    def _modulate(self, x, norm, scale, shift):
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.layer_norm_modulate)(
                x, scale, shift, norm.eps, self.config.fused_modulation
            )
        value = norm(x)
        if self.config.fused_modulation:
            return torch.addcmul(shift, value, 1 + scale)
        return value * (1 + scale) + shift

    def _residual(self, x, gate, value):
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.gated_residual)(
                x, gate, value, self.config.fused_modulation
            )
        if self.config.fused_modulation:
            return torch.addcmul(x, gate, value)
        return x + gate * value

    def _qk(self, norm, projected, rotary):
        """Full-width RMSNorm, head split and RoPE: (B,S,W) -> (B,S,H,D)."""
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.rms_rotate)(
                projected, norm.weight, norm.eps, self.config.num_heads, *rotary
            )
        head_shape = (self.config.num_heads, self.config.head_dim)
        return rotate(norm(projected).unflatten(-1, head_shape), rotary)

    def _affine_norm(self, container, x):
        norm = container.norm
        if self.config.compile_elementwise:
            return fusion.compiled(fusion.layer_norm_affine)(
                x, norm.weight, norm.bias, norm.eps
            )
        return container(x)

    def _projection_input(
        self, hidden, scale, shift, camera_rays, scale_embedding
    ):
        """Apply the configured camera/modulation store boundary before QKV."""
        if self.config.fusion_recipe == "plucker_prefix":
            camera = F.linear(
                camera_rays.to(hidden.dtype), self.cam_encoder.weight
            )
            first, _, last = self.scale_mlp
            scale_product = F.linear(
                scale_embedding.to(hidden.dtype), first.weight
            )
            activated = fusion.compiled(fusion.bias_silu)(
                scale_product, first.bias
            )
            scale_product = F.linear(activated, last.weight)
            return fusion.compiled(fusion.plucker_modulate)(
                hidden,
                scale,
                shift,
                camera,
                self.cam_encoder.bias,
                scale_product,
                last.bias,
                self.norm1.eps,
            )
        value = self._modulate(hidden, self.norm1, scale, shift)
        camera = self.cam_encoder(camera_rays.to(value.dtype))
        if self.scale_mlp is not None:
            camera = camera + self.scale_mlp(
                scale_embedding.to(value.dtype)
            ).unsqueeze(1)
        return value + camera

    def _project_qk(self, projection, norm, value, rotary):
        """Keep native compiled projection bias/RMS separate from eager RoPE."""
        if self.config.fusion_recipe == "plucker_prefix":
            product = projection(value, include_bias=False)
            normalized = fusion.compiled(fusion.bias_rms)(
                product, projection.bias, norm.weight, norm.eps
            )
            return rotate(
                normalized.unflatten(
                    -1, (self.config.num_heads, self.config.head_dim)
                ),
                rotary,
            )
        return self._qk(norm, projection(value), rotary)

    def _projection_tail(
        self,
        hidden,
        gate,
        attended,
        context,
        attention,
        ff_scale,
        ff_shift,
        ff_gate,
    ):
        """Preserve compiled native stores through text attention and FFN."""
        main_product = self.to_out(attended.flatten(2), include_bias=False)
        norm = self.self_attn_residual_norm.norm
        value = fusion.compiled(residual.projection_residual_affine)(
            hidden,
            gate,
            main_product,
            self.to_out.bias,
            norm.weight,
            norm.bias,
            norm.eps,
        )
        text_product = self.attn2(value, context, attention, include_bias=False)
        mean, variance, hidden = fusion.compiled(
            residual.attention_residual_statistics
        )(
            hidden,
            gate,
            main_product,
            self.to_out.bias,
            text_product,
            self.attn2.to_out.bias,
        )
        value = fusion.compiled(residual.modulate_from_statistics)(
            hidden, mean, variance, ff_scale, ff_shift, self.config.eps
        )
        product = self.ffn.fc_in(value, include_bias=False)
        value = fusion.compiled(residual.projection_gelu)(
            product, self.ffn.fc_in.bias
        )
        product = self.ffn.fc_out(value, include_bias=False)
        return fusion.compiled(residual.projection_residual)(
            hidden, ff_gate, product, self.ffn.fc_out.bias
        )

    def forward(
        self,
        hidden,
        context,
        temb,
        rotary,
        camera_rays,
        scale_embedding,
        *,
        tiles: TileLayout | None = None,
    ):
        """Evaluate a block without retaining request tensors on the model."""
        if camera_rays.shape != (*hidden.shape[:2], 6):
            raise ValueError("Plücker rays must align with the latent tokens")
        if (scale_embedding is None) != (self.scale_mlp is None):
            raise ValueError(
                "Plücker scale input must match the configured scale branch"
            )
        modulation = (self.scale_shift_table + temb).to(hidden.dtype)
        shift, scale, gate, ff_shift, ff_scale, ff_gate = modulation.chunk(
            6, dim=1
        )
        value = self._projection_input(
            hidden, scale, shift, camera_rays, scale_embedding
        )
        head_shape = (self.config.num_heads, self.config.head_dim)
        tile_rotary = None
        if self.config.vsa is not None and not self.config.compile_elementwise:
            # Preserve ordered RMS and its BF16 store. Shared tile preparation
            # consumes these normalized rows and owns RoPE before pooling.
            q = self.norm_q(self.to_q(value)).unflatten(-1, head_shape)
            k = self.norm_k(self.to_k(value)).unflatten(-1, head_shape)
            tile_rotary = rotary
        else:
            # Compiled RMS/RoPE has its own store boundary; dense attention
            # and that explicit composition retain their existing arithmetic.
            q = self._project_qk(self.to_q, self.norm_q, value, rotary)
            k = self._project_qk(self.to_k, self.norm_k, value, rotary)
        v = self.to_v(value).unflatten(-1, head_shape)
        attention = AttentionBatch({0: DenseInput(False, None)}, None)
        if self.config.vsa is None:
            if tiles is not None:
                raise ValueError(
                    "dense Wan attention does not consume VSA tiles"
                )
            attended = self.attn1(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attention,
            ).transpose(1, 2)
        else:
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
                        keep=tiles.keep(self.config.vsa.sparsity),
                        rotary=tile_rotary,
                    )
                    for qi, ki, vi, gi in zip(
                        q, k, v, gate_compress, strict=True
                    )
                ]
            )
        if self.config.fusion_recipe == "plucker_prefix":
            return self._projection_tail(
                hidden,
                gate,
                attended,
                context,
                attention,
                ff_scale,
                ff_shift,
                ff_gate,
            )
        value = self.to_out(attended.flatten(2))
        hidden = self._residual(hidden, gate, value)
        hidden = hidden + self.attn2(
            self._affine_norm(self.self_attn_residual_norm, hidden),
            context,
            attention,
        )
        value = self._modulate(
            hidden, self.cross_attn_residual_norm.norm, ff_scale, ff_shift
        )
        return self._residual(hidden, ff_gate, self.ffn(value))


class TimeEmbedding(nn.Module):
    """Native Wan cosine/sine time frequencies and two learned projections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.frequency_dim = config.freq_dim
        self.mlp = MLP(
            config.freq_dim,
            config.hidden_size,
            config.hidden_size,
            activation=nn.SiLU(),
        )

    def frequencies(self, *, device) -> torch.Tensor:
        """Prepare the immutable FP32 time frequencies outside graph capture."""
        half = self.frequency_dim // 2
        # CPU FP32 construction is part of the reference rounding contract.
        return torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device="cpu")
            / half
        ).to(device)

    def forward(self, timestep: torch.Tensor, frequencies=None) -> torch.Tensor:
        if frequencies is None:
            frequencies = self.frequencies(device=timestep.device)
        angles = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
        return self.mlp(embedding.to(self.mlp.fc_in.weight.dtype))


class Modulation(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = Linear(width, 6 * width)

    def forward(self, time_embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.nn.functional.silu(time_embedding))


class ConditionEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.time_embedder = TimeEmbedding(config)
        self.time_modulation = Modulation(config.hidden_size)
        self.text_embedder = MLP(
            config.text_dim,
            config.hidden_size,
            config.hidden_size,
            activation=nn.GELU(approximate="tanh"),
        )

    def forward(self, timestep, context, time_frequencies=None):
        time = self.time_embedder(timestep, time_frequencies)
        return (
            time,
            self.time_modulation(time).unflatten(-1, (6, -1)),
            self.text_embedder(context),
        )


class PluckerTransformer(nn.Module):
    """Wan velocity prediction with explicit Plücker and rotary inputs.

    Latents have B,C,T,H,W order; rays accept B,6,T/pt,H/ph,W/pw or
    B,S,6 in the matching T,H,W token order. Timesteps are network times
    (1000*sigma), not solver coordinates. The caller prepares rotary tensors
    outside graph capture and owns their storage lifetime.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
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
            PluckerBlock(config) for _ in range(config.num_layers)
        )
        self.norm_out = _norm_container(width, config.eps, affine=False)
        self.proj_out = Linear(
            width, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, width) / width**0.5, requires_grad=False
        )

    def rotary_embedding(
        self, latent_size: tuple[int, int, int], *, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build native CPU FP64 trigonometry, then transfer FP32 results."""
        sizes = tuple(
            size // patch
            for size, patch in zip(
                latent_size, self.config.patch_size, strict=True
            )
        )
        head = self.config.head_dim
        dimensions = (head - 4 * (head // 6), 2 * (head // 6), 2 * (head // 6))
        positions = torch.meshgrid(
            *(
                torch.arange(size, dtype=torch.float32, device="cpu")
                for size in sizes
            ),
            indexing="ij",
        )
        cosines, sines = [], []
        for position, dim in zip(positions, dimensions, strict=True):
            frequencies = 1.0 / (
                10000 ** (torch.arange(0, dim, 2, device="cpu").double() / dim)
            )
            angles = torch.outer(position.flatten(), frequencies)
            cosines.append(angles.cos().repeat_interleave(2, dim=-1))
            sines.append(angles.sin().repeat_interleave(2, dim=-1))
        return (
            torch.cat(cosines, dim=-1).to(device=device, dtype=torch.float32),
            torch.cat(sines, dim=-1).to(device=device, dtype=torch.float32),
        )

    def forward(
        self,
        latents,
        context,
        timestep,
        camera_rays,
        scale_embedding,
        rotary,
        *,
        time_frequencies=None,
        tiles: TileLayout | None = None,
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
        if camera_rays.ndim == 5:
            camera_rays = camera_rays.flatten(2).transpose(1, 2)
        hidden = self.patch_embedding(latents).flatten(2).transpose(1, 2)
        time, modulation, context = self.condition_embedder(
            timestep, context, time_frequencies
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                context,
                modulation,
                rotary,
                camera_rays,
                scale_embedding,
                tiles=tiles,
            )
        shift, scale = (self.scale_shift_table + time.unsqueeze(1)).chunk(
            2, dim=1
        )
        # The native final norm rounds before FP32 scale/shift arithmetic.
        hidden = (self.norm_out(hidden).float() * (1 + scale) + shift).to(
            hidden.dtype
        )
        hidden = self.proj_out(hidden)
        hidden = hidden.reshape(
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
