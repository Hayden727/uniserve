"""Euler integration with explicit prediction and sample mutation behavior."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch


@dataclass(frozen=True, slots=True)
class ReNoiseInput:
    """Borrow this request's independent noise and fixed clean condition.

    Noise and clean tensors have the sample shape and precision. The boolean
    mask may broadcast within that shape. Random draws happen outside this
    numerical call, before capture, in the native generator's draw order.
    """

    noise: torch.Tensor
    clean: torch.Tensor
    clean_mask: torch.Tensor

    def __post_init__(self):
        if self.noise.shape != self.clean.shape:
            raise ValueError("re-noising noise and clean shape must match")
        if self.noise.dtype != self.clean.dtype:
            raise ValueError("re-noising noise and clean dtype must match")
        if self.clean_mask.dtype != torch.bool:
            raise ValueError("clean mask must be boolean")
        if (
            len({self.noise.device, self.clean.device, self.clean_mask.device})
            != 1
        ):
            raise ValueError("re-noising inputs must share one device")
        try:
            shape = torch.broadcast_shapes(
                self.clean_mask.shape, self.clean.shape
            )
        except RuntimeError as error:
            raise ValueError(
                "clean mask shape cannot broadcast to sample"
            ) from error
        if shape != self.clean.shape:
            raise ValueError("clean mask shape cannot expand sample")

    def shard(self, slices):
        """Borrow the same local slice as a sharded prediction."""
        return ReNoiseInput(
            self.noise[slices],
            self.clean[slices],
            self.clean_mask.expand_as(self.clean)[slices],
        )


def _output(value, out):
    if out is None:
        return value
    if (
        out.shape != value.shape
        or out.dtype != value.dtype
        or out.device != value.device
    ):
        raise ValueError(
            "solver output must match the computed shape, dtype and device"
        )
    return out.copy_(value)


def euler_step(sample, velocity, timestep, next_timestep, *, out=None):
    """Cast the time difference to sample precision before the Euler update."""
    delta = (next_timestep - timestep).to(
        dtype=sample.dtype, device=sample.device
    )
    return _output(sample + delta * velocity, out)


def clean_sample_to_velocity(prediction, sample, timestep, *, out=None):
    """Convert an ascending clean-time prediction with a 1e-6 endpoint floor."""
    denominator = (1.0 - timestep).clamp_min(1e-6)
    while denominator.ndim < sample.ndim:
        denominator = denominator.unsqueeze(-1)
    result = (prediction - sample) / denominator.to(
        dtype=sample.dtype, device=sample.device
    )
    return _output(result, out)


class EulerSolver:
    """First-order Euler update from velocity or clean-sample predictions."""

    def __init__(
        self, prediction_type: Literal["velocity", "sample"] = "velocity"
    ):
        if prediction_type not in {"velocity", "sample"}:
            raise ValueError("Euler prediction must be velocity or sample")
        self.prediction_type = prediction_type

    def step_(
        self, prediction, sample, timestep, next_timestep, *, sigma, next_sigma
    ) -> None:
        velocity = (
            clean_sample_to_velocity(prediction, sample, timestep)
            if self.prediction_type == "sample"
            else prediction
        )
        # Prediction precision can exceed the retained sample precision (for
        # example an FP32 image velocity with a BF16 trajectory). Preserve the
        # promoted Euler arithmetic, then round once when committing the sample.
        sample.copy_(euler_step(sample, velocity, timestep, next_timestep))


class CleanSampleEulerSolver:
    """Consume velocity as clean-prediction scratch.

    Updates the sample in place.
    """

    prediction_type = "velocity"

    def step_(
        self, prediction, sample, timestep, next_timestep, *, sigma, next_sigma
    ) -> None:
        # Clean time is rounded in sample precision before subtraction, whereas
        # the sigma ratio retains FP32. Their order is part of the solver math.
        clean_time = 1.0 - timestep.to(device=sample.device, dtype=sample.dtype)
        prediction.mul_(clean_time).add_(sample)
        ratio = next_sigma.float() / sigma.float()
        sample.mul_(ratio)
        sample.addcmul_(prediction, 1.0 - ratio)


class ReNoiseSolver:
    """Predict-x0 DMD: independently re-noise, then restore clean slots.

    The final zero-sigma step returns x0 exactly. No RNG or mutable request
    state is stored on the solver, and endpoint selection stays on-device.
    """

    prediction_type = "sample"

    def step_(
        self,
        prediction,
        sample,
        timestep,
        next_timestep,
        *,
        sigma,
        next_sigma,
        inputs: ReNoiseInput | None = None,
    ) -> None:
        if inputs is None:
            raise ValueError("DMD requires explicit re-noising inputs")
        if (
            inputs.clean.shape != sample.shape
            or inputs.clean.dtype != sample.dtype
            or inputs.clean.device != sample.device
            or prediction.shape != sample.shape
            or prediction.dtype != sample.dtype
        ):
            raise ValueError(
                "re-noising inputs and prediction must match sample"
            )
        # Native Wan rounds sigma to latent precision before both products.
        level = next_sigma.to(device=sample.device, dtype=sample.dtype)
        while level.ndim < sample.ndim:
            level = level.unsqueeze(-1)
        renoised = (1.0 - level) * prediction + level * inputs.noise
        result = torch.where(level == 0, prediction, renoised)
        sample.copy_(torch.where(inputs.clean_mask, inputs.clean, result))


Solver: TypeAlias = EulerSolver | CleanSampleEulerSolver | ReNoiseSolver
