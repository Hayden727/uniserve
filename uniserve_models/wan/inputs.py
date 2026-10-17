"""Wan latent dimensions and borrowed per-request conditioning."""

from dataclasses import dataclass
from enum import StrEnum

import torch

from uniserve.model import DenoiserInput as BaseInput

from .camera import PluckerInput, UCPEInput


class LatentFormat(StrEnum):
    """Physical layout of logical N,C,F,H,W inputs to patch embedding."""

    CONTIGUOUS = "contiguous"
    CHANNELS_LAST_3D = "channels_last_3d"

    @property
    def memory_format(self):
        return (
            torch.channels_last_3d
            if self is LatentFormat.CHANNELS_LAST_3D
            else torch.contiguous_format
        )


@dataclass(frozen=True, slots=True)
class LatentSize:
    frames: int
    height: int
    width: int
    text_tokens: int

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.frames,
                self.height,
                self.width,
                self.text_tokens,
            )
        ):
            raise ValueError(
                "Wan latent and context dimensions must be positive integers"
            )


@dataclass(frozen=True)
class DenoiserInput(BaseInput[LatentSize]):
    context: torch.Tensor
    camera: PluckerInput | UCPEInput
    rotary: tuple[torch.Tensor, torch.Tensor] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size != 1 or set(self.latents) != {"video"}:
            raise ValueError(
                "Wan evaluates one conditioned latent grid per call"
            )
