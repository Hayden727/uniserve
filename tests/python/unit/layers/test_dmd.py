"""Re-noising uses explicit request noise and retains clean conditions."""

import pytest
import torch

pytestmark = pytest.mark.unit


def test_rungs_use_wan_network_time_and_native_rounding():
    from uniserve.diffusion import RungGrid

    grid = RungGrid(
        (1000, 750, 500, 250), 5.0, 1000.0, time_parameterization="sigma"
    )
    schedule = grid.schedule("cpu")
    # Native constructs Python float warped times, rounds them to FP32,
    # then divides by 1000 at the numerical call boundary.
    expected = torch.tensor([1000.0, 937.5, 2500.0 / 3.0, 625.0, 0.0])
    torch.testing.assert_close(schedule.timesteps, expected, rtol=0, atol=0)
    torch.testing.assert_close(schedule.sigmas, expected / 1000, rtol=0, atol=0)
    with pytest.raises(ValueError, match="parameterization"):
        RungGrid((1000,), 5.0, 1000.0, time_parameterization="unknown")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_renoising_preserves_native_rounding_and_clean_slots(dtype):
    from uniserve.diffusion import ReNoiseInput, ReNoiseSolver

    clean = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    prediction = torch.tensor([[0.2, 0.7, -1.0, 0.35]], dtype=dtype)
    noise = torch.tensor([[-0.5, 0.2, 1.5, -0.1]], dtype=dtype)
    mask = torch.tensor([[True, False, True, False]])
    sample = torch.zeros_like(clean)
    next_time = torch.tensor([2500.0 / 3.0])
    level = (next_time.float() / 1000).to(dtype).reshape(1, 1)
    expected = torch.where(
        mask, clean, (1 - level) * prediction + level * noise
    )
    ReNoiseSolver().step_(
        prediction,
        sample,
        torch.tensor([1000.0]),
        next_time,
        sigma=torch.tensor(1.0),
        next_sigma=next_time[0] / 1000,
        inputs=ReNoiseInput(noise, clean, mask),
    )
    torch.testing.assert_close(sample, expected, rtol=0, atol=0)
    ReNoiseSolver().step_(
        prediction,
        sample,
        next_time,
        torch.tensor([0.0]),
        sigma=next_time[0] / 1000,
        next_sigma=torch.tensor(0.0),
        inputs=ReNoiseInput(noise, clean, mask),
    )
    torch.testing.assert_close(
        sample, torch.where(mask, clean, prediction), rtol=0, atol=0
    )


def test_missing_or_mismatched_request_noise_is_an_error():
    from uniserve.diffusion import ReNoiseInput, ReNoiseSolver

    sample = torch.zeros(2, 3)
    kwargs = {"sigma": torch.tensor(1.0), "next_sigma": torch.tensor(0.5)}
    with pytest.raises(ValueError, match="re-noising inputs"):
        ReNoiseSolver().step_(sample, sample, 1.0, 0.5, **kwargs)
    with pytest.raises(ValueError, match="shape"):
        ReNoiseInput(
            torch.zeros(3, 2), sample, torch.ones(1, 3, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="boolean"):
        ReNoiseInput(sample, sample, torch.ones_like(sample))


def test_step_uses_request_specific_noise_and_clean_condition():
    from uniserve.diffusion import (
        ReNoiseInput,
        ReNoiseSolver,
        RungGrid,
        advance_,
    )
    from uniserve.model import Denoiser, LatentInput

    class Predictor(Denoiser):
        def __init__(self):
            super().__init__(
                modalities=("video",),
                prediction_dtype=torch.float32,
                solver=ReNoiseSolver(),
                grids={
                    "video": RungGrid(
                        (1000, 750, 500, 250),
                        5,
                        1000,
                        time_parameterization="sigma",
                    )
                },
            )

        def latent_shape(self, modality, size):
            return size

        def noise_shape(self, modality, size):
            return size

        def prepare_latents(self, sizes, *, noise, state, constants, workspace):
            state["video"].copy_(noise["video"])

        def forward(self, inputs, *, state, constants, workspace):
            return {
                "video": tuple(x.tensor * 0.25 for x in inputs.latents["video"])
            }

    denoiser = Predictor()
    schedule = denoiser.grids["video"].schedule("cpu")
    clean = [torch.full((1, 3), float(i + 1)) for i in range(2)]
    mask = torch.tensor([[True, False, False]])
    samples = [x.clone() for x in clean]
    expected = [x.clone() for x in clean]
    generators = [torch.Generator().manual_seed(i) for i in range(2)]
    for step in range(4):
        for request in (0, 1):
            # The final step consumes no RNG; a zero view is safe graph input.
            noise = (
                torch.randn((1, 3), generator=generators[request])
                if step < 3
                else torch.zeros(1, 3)
            )
            predicted = expected[request] * 0.25
            level = schedule.sigmas[step + 1]
            expected[request] = torch.where(
                mask, clean[request], (1 - level) * predicted + level * noise
            )
            latent = LatentInput(
                samples[request],
                schedule.timesteps[step],
                ReNoiseInput(noise, clean[request], mask),
            )
            advance_(
                denoiser,
                {"video": (latent,)},
                {"video": (samples[request] * 0.25,)},
                {"video": schedule},
                schedule.step(step),
            )
            torch.testing.assert_close(
                samples[request], expected[request], rtol=0, atol=0
            )
