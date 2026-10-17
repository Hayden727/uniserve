"""Bidirectional UMT5 text computation for Wan's FP32 conditioning.

Numerics and checkpoint names follow FastVideo's Apache-2.0 T5 encoder,
derived from Hugging Face Transformers and the original T5 implementation.
Each UMT5 layer owns its relative position embedding; request masks and
attention biases are local tensors, never retained on the network.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from uniserve.model import TextEncoder
from uniserve.nn.gemv import GemvReduction, validate_reduction
from uniserve.nn.linear import Linear
from uniserve.nn.norm import RMSNorm


@dataclass(frozen=True, slots=True)
class TextConfig:
    vocab_size: int = 256384
    hidden_size: int = 4096
    head_dim: int = 64
    ffn_dim: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_buckets: int = 32
    max_distance: int = 128
    eps: float = 1e-6
    rms_reduction_order: Literal["default", "ascending"] = "default"
    attention_gemv_reduction: GemvReduction = "default"
    ffn_output_gemv_reduction: GemvReduction = "default"

    def __post_init__(self):
        validate_reduction(self.attention_gemv_reduction)
        validate_reduction(self.ffn_output_gemv_reduction)
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.head_dim,
            self.ffn_dim,
            self.num_layers,
            self.num_heads,
            self.relative_buckets,
            self.max_distance,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("UMT5 dimensions must be positive integers")
        if self.relative_buckets < 4 or self.relative_buckets % 4:
            raise ValueError(
                "bidirectional relative buckets must divide by four"
            )
        if self.max_distance <= self.relative_buckets // 4:
            raise ValueError("relative distance must exceed exact buckets")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("UMT5 epsilon must be finite and positive")
        if self.rms_reduction_order not in {"default", "ascending"}:
            raise ValueError("unknown UMT5 RMS reduction order")


def _norm(config: TextConfig):
    return RMSNorm(
        config.hidden_size,
        config.eps,
        round_before_scale=True,
        reduction_order=config.rms_reduction_order,
    )


def relative_buckets(length: int, config: TextConfig, *, device):
    """Bucket signed key-minus-query distances with T5's logarithmic bins."""
    position = torch.arange(length, dtype=torch.long, device=device)
    relative = position[None, :] - position[:, None]
    half = config.relative_buckets // 2
    buckets = (relative > 0).to(torch.long) * half
    distance = relative.abs()
    exact = half // 2
    large = exact + (
        torch.log(distance.float() / exact)
        / math.log(config.max_distance / exact)
        * (half - exact)
    ).to(torch.long)
    large = torch.minimum(large, torch.full_like(large, half - 1))
    return buckets + torch.where(distance < exact, distance, large)


class SelfAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.heads, self.head_dim = config.num_heads, config.head_dim
        width = config.num_heads * config.head_dim
        self.qkv_proj = Linear(
            config.hidden_size,
            3 * width,
            bias=False,
            gemv_reduction=config.attention_gemv_reduction,
        )
        self.o = Linear(
            width,
            config.hidden_size,
            bias=False,
            gemv_reduction=config.attention_gemv_reduction,
        )
        self.relative_attention_bias = nn.Embedding(
            config.relative_buckets, config.num_heads
        )

    def forward(self, hidden, mask, buckets):
        batch, length, _ = hidden.shape
        q, k, v = (
            value.reshape(batch, length, self.heads, self.head_dim)
            for value in self.qkv_proj(hidden).chunk(3, dim=-1)
        )
        bias = self.relative_attention_bias(buckets).permute(2, 0, 1)
        bias = bias.unsqueeze(0).repeat(batch, 1, 1, 1)
        if mask is not None:
            bias.masked_fill_(
                mask[:, None, None, :] == 0, torch.finfo(q.dtype).min
            )
        # T5 does not apply the usual 1/sqrt(head_dim) attention scaling.
        scores = torch.einsum("binc,bjnc->bnij", q, k)
        scores += bias
        probabilities = scores.float().softmax(dim=-1).to(scores.dtype)
        result = torch.einsum("bnij,bjnc->binc", probabilities, v)
        return self.o(result.reshape(batch, length, -1))


class AttentionLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.SelfAttention = SelfAttention(config)
        self.layer_norm = _norm(config)

    def forward(self, hidden, mask, buckets):
        return hidden + self.SelfAttention(
            self.layer_norm(hidden), mask, buckets
        )


class GatedGELU(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.wi_0 = Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.wi_1 = Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.wo = Linear(
            config.ffn_dim,
            config.hidden_size,
            bias=False,
            gemv_reduction=config.ffn_output_gemv_reduction,
        )

    def forward(self, hidden):
        value = self.wi_0(hidden)
        # Preserve gelu_new's operation order, including the FP32 cube.
        activated = (
            0.5
            * value
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (value + 0.044715 * torch.pow(value, 3.0))
                )
            )
        )
        return self.wo(activated * self.wi_1(hidden))


class FeedForwardLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.DenseReluDense = GatedGELU(config)
        self.layer_norm = _norm(config)

    def forward(self, hidden):
        return hidden + self.DenseReluDense(self.layer_norm(hidden))


class Block(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            (AttentionLayer(config), FeedForwardLayer(config))
        )

    def forward(self, hidden, mask, buckets):
        return self.layer[1](self.layer[0](hidden, mask, buckets))


class Stack(nn.Module):
    def __init__(self, config: TextConfig, embedding: nn.Embedding):
        super().__init__()
        self.embed_tokens = embedding
        self.block = nn.ModuleList(
            Block(config) for _ in range(config.num_layers)
        )
        self.final_layer_norm = _norm(config)

    def forward(self, ids, mask, buckets):
        hidden = self.embed_tokens(ids)
        for block in self.block:
            hidden = block(hidden, mask, buckets)
        return self.final_layer_norm(hidden)


class UMT5(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = Stack(config, self.shared)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError(
                "UMT5 requires a nonempty batch-by-token ID matrix"
            )
        if (
            attention_mask is not None
            and attention_mask.shape != input_ids.shape
        ):
            raise ValueError("UMT5 attention mask must match the input IDs")
        buckets = relative_buckets(
            input_ids.shape[1], self.config, device=input_ids.device
        )
        return self.encoder(input_ids, attention_mask, buckets)


class Encoder(TextEncoder):
    """Encode each actual prompt length before diffusion feature padding."""

    def __init__(self, config: TextConfig):
        super().__init__(UMT5(config))

    def encode(
        self, tokens: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        if any(
            value.ndim != 1 or not 1 <= value.numel() <= 512 for value in tokens
        ):
            raise ValueError("Wan text requires sequences of 1..512 tokens")
        # Combining equal-length requests changes FP32 GEMM selection. Preserve
        # the reference batch of one even when the worker admits several texts.
        return tuple(
            self.network(value.unsqueeze(0)).squeeze(0) for value in tokens
        )

    def output_layout(self, num_tokens: int, dtype: torch.dtype):
        return super().output_layout(num_tokens, torch.float32)
