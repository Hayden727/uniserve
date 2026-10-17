# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
"""Wan 2.1 VAE computation with causal caches scoped to each invocation.

Decoder equations and parameter names follow FastVideo's Apache-2.0 Wan VAE.
Only the standard Wan 2.1 architecture participates here. The numerical
cache contains borrowed activations of one segment and is never stored on
the resident model.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.nn.convolution import Conv2d, Conv3d
from uniserve.nn.vae import LatentNormalization


@dataclass(frozen=True, slots=True)
class VAEConfig:
    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    encoder_norm_order: Literal["default", "ascending"] = "default"
    latent_mean: tuple[float, ...] = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    latent_std: tuple[float, ...] = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )

    def __post_init__(self):
        if self.encoder_norm_order not in {"default", "ascending"}:
            raise ValueError("unknown Wan VAE encoder norm reduction order")
        if any(
            type(n) is not int or n < 1
            for n in (
                self.base_dim,
                self.z_dim,
                self.num_res_blocks,
                *self.dim_mult,
            )
        ):
            raise ValueError("Wan VAE dimensions must be positive integers")
        if len(self.dim_mult) != 4 or self.base_dim % 2:
            raise ValueError(
                "Wan VAE requires four scales and an even base width"
            )
        if self.temporal_downsample != (False, True, True):
            raise ValueError(
                "Wan 2.1 requires temporal downsampling at its last two scales"
            )
        if (
            len(self.latent_mean) != self.z_dim
            or len(self.latent_std) != self.z_dim
        ):
            raise ValueError(
                "Wan VAE statistics must cover every latent channel"
            )
        if any(
            not math.isfinite(value)
            for value in (*self.latent_mean, *self.latent_std)
        ) or any(value <= 0 for value in self.latent_std):
            raise ValueError(
                "Wan VAE statistics must be finite with positive deviations"
            )


class Normalization(LatentNormalization):
    """Shift/scale in latent dtype; decode divides by the FP32 reciprocal."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        mean = torch.tensor(
            config.latent_mean, dtype=torch.float32, device="cpu"
        )
        scale = 1.0 / torch.tensor(
            config.latent_std, dtype=torch.float32, device="cpu"
        )
        self.register_buffer(
            "mean", mean.view(1, -1, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "scale", scale.view(1, -1, 1, 1, 1), persistent=False
        )

    def normalize(self, latents):
        return (latents - self.mean.to(latents.dtype)) * self.scale.to(
            latents.dtype
        )

    def denormalize(self, latents):
        return latents.float() / self.scale + self.mean


class CausalConv3d(Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)
        self.channels_last = False

    def forward(self, x, previous=None):
        padding = list(self.causal_padding)
        if previous is not None and padding[4] > 0:
            x = torch.cat((previous.to(x.device), x), dim=2)
            padding[4] -= previous.shape[2]
        x = F.pad(x, padding)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last_3d)
        return super().forward(x)


class Cache:
    """Local causal activations consumed in numerical layer order."""

    def __init__(self, slots: int):
        self.values: list[torch.Tensor | str | None] = [None] * slots
        self.index = 0

    def convolution(self, layer: CausalConv3d, x: torch.Tensor):
        previous = self.values[self.index]
        saved = x[:, :, -2:].clone()
        if saved.shape[2] < 2 and previous is not None:
            saved = torch.cat(
                (previous[:, :, -1:].to(saved.device), saved), dim=2
            )
        result = layer(x, previous)
        self.values[self.index] = saved
        self.index += 1
        return result


class ChannelNorm(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        images: bool = False,
        reduction_order: Literal["default", "ascending"] = "default",
    ):
        super().__init__()
        self.reduction_order = reduction_order
        self.scale = channels**0.5
        self.gamma = nn.Parameter(
            torch.ones(channels, *((1, 1) if images else (1, 1, 1)))
        )

    def forward(self, x):
        if self.reduction_order == "ascending" and x.stride(1) == 1:
            from uniserve_kernels.norm.l2 import channel_l2_ascending

            # The L2 norm rounds to activation dtype before division. The
            # non-channel-contiguous reduction uses the same tree in both
            # reference and candidate stacks, so it keeps the tensor path.
            normalized = x / channel_l2_ascending(x).clamp_min(1e-12)
            return normalized * self.scale * self.gamma + 0.0
        return F.normalize(x, dim=1) * self.scale * self.gamma + 0.0


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, reduction_order="default"):
        super().__init__()
        self.norm1 = ChannelNorm(in_dim, reduction_order=reduction_order)
        self.norm2 = ChannelNorm(out_dim, reduction_order=reduction_order)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x, cache: Cache):
        shortcut = self.conv_shortcut(x)
        x = cache.convolution(self.conv1, F.silu(self.norm1(x)))
        x = cache.convolution(self.conv2, F.silu(self.norm2(x)))
        return x + shortcut


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, *, reduction_order="default"):
        super().__init__()
        self.norm = ChannelNorm(
            channels, images=True, reduction_order=reduction_order
        )
        self.to_qkv = Conv2d(channels, channels * 3, 1)
        self.proj = Conv2d(channels, channels, 1)

    def forward(self, x):
        identity = x
        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        qkv = self.to_qkv(self.norm(x)).reshape(
            batch * time, 1, channels * 3, -1
        )
        q, k, v = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = (
            x.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch * time, channels, height, width)
        )
        x = self.proj(x).view(batch, time, channels, height, width)
        return x.permute(0, 2, 1, 3, 4) + identity


class MidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attentions = nn.ModuleList((AttentionBlock(channels),))
        self.resnets = nn.ModuleList(
            (
                ResidualBlock(channels, channels),
                ResidualBlock(channels, channels),
            )
        )

    def forward(self, x, cache):
        return self.resnets[1](
            self.attentions[0](self.resnets[0](x, cache)), cache
        )


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class Resample(nn.Module):
    def __init__(self, channels: int, *, temporal: bool):
        super().__init__()
        self.temporal = temporal
        self.resample = nn.Sequential(
            Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
            Conv2d(channels, channels // 2, 3, padding=1),
        )
        if temporal:
            self.time_conv = CausalConv3d(
                channels, 2 * channels, (3, 1, 1), padding=(1, 0, 0)
            )

    def forward(self, x, cache: Cache):
        batch, channels, time, height, width = x.shape
        if self.temporal:
            previous = cache.values[cache.index]
            if previous is None:
                cache.values[cache.index] = "Rep"
            else:
                saved = x[:, :, -2:].clone()
                if saved.shape[2] < 2:
                    prefix = (
                        torch.zeros_like(saved)
                        if isinstance(previous, str)
                        else previous[:, :, -1:].to(saved.device)
                    )
                    saved = torch.cat((prefix, saved), dim=2)
                x = (
                    self.time_conv(x)
                    if isinstance(previous, str)
                    else self.time_conv(x, previous)
                )
                cache.values[cache.index] = saved
                x = x.reshape(batch, 2, channels, time, height, width)
                x = torch.stack((x[:, 0], x[:, 1]), dim=3).reshape(
                    batch, channels, time * 2, height, width
                )
            cache.index += 1
        time = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        x = self.resample(x)
        return x.view(batch, time, *x.shape[1:]).permute(0, 2, 1, 3, 4)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_res_blocks, *, upsample, temporal):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResidualBlock(in_dim if index == 0 else out_dim, out_dim)
            for index in range(num_res_blocks + 1)
        )
        self.upsamplers = (
            nn.ModuleList((Resample(out_dim, temporal=temporal),))
            if upsample
            else None
        )

    def forward(self, x, cache):
        for layer in self.resnets:
            x = layer(x, cache)
        return x if self.upsamplers is None else self.upsamplers[0](x, cache)


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        dims = [
            config.base_dim * multiplier
            for multiplier in (config.dim_mult[-1], *reversed(config.dim_mult))
        ]
        temporal = tuple(reversed(config.temporal_downsample))
        self.conv_in = CausalConv3d(config.z_dim, dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        self.up_blocks = nn.ModuleList(
            UpBlock(
                in_dim if index == 0 else in_dim // 2,
                out_dim,
                config.num_res_blocks,
                upsample=index < 3,
                temporal=temporal[index] if index < 3 else False,
            )
            for index, (in_dim, out_dim) in enumerate(
                zip(dims[:-1], dims[1:], strict=True)
            )
        )
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], 3, 3, padding=1)

    def forward(self, x, cache):
        x = self.mid_block(cache.convolution(self.conv_in, x), cache)
        for layer in self.up_blocks:
            x = layer(x, cache)
        return cache.convolution(self.conv_out, F.silu(self.norm_out(x)))


class DecoderNetwork(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.post_quant_conv = CausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = Decoder(config)
        self.cache_slots = sum(
            isinstance(layer, CausalConv3d) for layer in self.decoder.modules()
        )

    def forward(self, z):
        cache = Cache(self.cache_slots)
        x = self.post_quant_conv(z)
        frames = []
        for index in range(z.shape[2]):
            cache.index = 0
            frames.append(self.decoder(x[:, :, index : index + 1], cache))
        return torch.cat(frames, dim=2).float().clamp(-1.0, 1.0)


class Downsample(nn.Module):
    def __init__(self, channels: int, *, temporal: bool):
        super().__init__()
        self.temporal = temporal
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            Conv2d(channels, channels, 3, stride=2),
        )
        if temporal:
            self.time_conv = CausalConv3d(
                channels, channels, (3, 1, 1), stride=(2, 1, 1)
            )

    def forward(self, x, cache: Cache):
        batch, channels, time, height, width = x.shape
        x = self.resample(
            x.permute(0, 2, 1, 3, 4).reshape(
                batch * time, channels, height, width
            )
        )
        x = x.view(batch, time, *x.shape[1:]).permute(0, 2, 1, 3, 4)
        if self.temporal:
            previous = cache.values[cache.index]
            if previous is None:
                cache.values[cache.index] = x.clone()
            else:
                saved = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat((previous[:, :, -1:], x), dim=2))
                cache.values[cache.index] = saved
            cache.index += 1
        return x


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        dims = [
            config.base_dim * multiplier for multiplier in (1, *config.dim_mult)
        ]
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        layers = []
        for index, (in_dim, out_dim) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            for block in range(config.num_res_blocks):
                layers.append(
                    ResidualBlock(
                        in_dim if block == 0 else out_dim,
                        out_dim,
                        reduction_order=config.encoder_norm_order,
                    )
                )
            if index < 3:
                layers.append(
                    Downsample(
                        out_dim, temporal=config.temporal_downsample[index]
                    )
                )
        self.downsamples = nn.ModuleList(layers)
        self.middle = nn.ModuleList(
            (
                ResidualBlock(
                    dims[-1],
                    dims[-1],
                    reduction_order=config.encoder_norm_order,
                ),
                AttentionBlock(
                    dims[-1], reduction_order=config.encoder_norm_order
                ),
                ResidualBlock(
                    dims[-1],
                    dims[-1],
                    reduction_order=config.encoder_norm_order,
                ),
            )
        )
        self.head = nn.Sequential(
            ChannelNorm(dims[-1], reduction_order=config.encoder_norm_order),
            nn.SiLU(),
            CausalConv3d(dims[-1], 2 * config.z_dim, 3, padding=1),
        )

    def forward(self, x, cache: Cache):
        x = cache.convolution(self.conv1, x)
        for layer in self.downsamples:
            x = layer(x, cache)
        x = self.middle[0](x, cache)
        x = self.middle[1](x)
        x = self.middle[2](x, cache)
        return cache.convolution(self.head[2], self.head[1](self.head[0](x)))


class EncoderNetwork(nn.Module):
    """Encode independent 1+4k frame clips in channels-last BF16."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.conv1 = CausalConv3d(2 * config.z_dim, 2 * config.z_dim, 1)
        self.cache_slots = sum(
            isinstance(layer, CausalConv3d) for layer in self.encoder.modules()
        )
        for layer in self.modules():
            if isinstance(layer, CausalConv3d):
                layer.channels_last = True
                layer.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] < 1 or (x.shape[2] - 1) % 4:
            raise ValueError("Wan encoder clips require 1+4k frames in NCTHW")
        cache = Cache(self.cache_slots)
        frames = []
        for index in range(1 + (x.shape[2] - 1) // 4):
            cache.index = 0
            chunk = (
                x[:, :, :1]
                if index == 0
                else x[:, :, 1 + 4 * (index - 1) : 1 + 4 * index]
            )
            frames.append(self.encoder(chunk, cache))
        return self.conv1(torch.cat(frames, dim=2)).chunk(2, dim=1)[0]
