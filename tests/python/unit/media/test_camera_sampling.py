"""Camera requests preserve native RNG draws and post-encoding text padding."""

import torch

from uniserve_models.wan.config import Config, TransformerConfig
from uniserve_models.wan.serving import ServingDenoiser
from uniserve_worker.model_executor.media_inputs import MediaBuilder


@torch.inference_mode()
def test_camera_request_draws_and_stages_each_native_noise_in_order():
    denoiser = ServingDenoiser(
        Config(
            TransformerConfig(
                num_layers=1,
                num_heads=2,
                head_dim=8,
                ffn_dim=32,
                text_dim=12,
                freq_dim=8,
            )
        )
    )
    builder = MediaBuilder(denoiser, max_frames=13, max_text_tokens=512)
    size = builder.size(13, 7, next(iter(denoiser.canvases.values())))
    assert builder.layout(size).num_text_tokens == 512
    assert builder.layouts() == (builder.layout(size),)
    views = {
        name: torch.empty(field.shape, dtype=field.dtype)
        for name, field in builder.buffers(size).items()
    }
    # Native UMT5 evaluates only real tokens, then its Wan postprocessor adds
    # zero feature rows to 512 before the DiT's biased text projection.
    assert views["text_condition"].shape == (512, 12)
    sample = torch.empty(
        denoiser.latent_shape("video", size), dtype=torch.bfloat16
    )
    schedules = denoiser.make_schedules(device="cpu")
    for seed in (31, 89, 31):
        builder.stage_request(size, views, seed=seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        expected = tuple(
            torch.randn(
                denoiser.noise_shape("video", size),
                generator=generator,
                dtype=torch.float32,
            )
            for _ in range(4)
        )
        torch.testing.assert_close(
            views["video_noise"], expected[0], rtol=0, atol=0
        )
        for value, reference in zip(
            builder.sampling_noise(size, views, host=True),
            expected[1:],
            strict=True,
        ):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for target, source in builder.initialize(
            size, views, {"video": sample}, constants={}, workspace={}
        ):
            assert target.shape == source.shape
            assert target.dtype == source.dtype
            target.copy_(source)
        torch.testing.assert_close(
            sample, expected[0].bfloat16(), rtol=0, atol=0
        )
        for index in range(4):
            bound = builder.bind(
                size, views, {"video": sample}, schedules, index
            )
            torch.testing.assert_close(
                bound.latents["video"][0].solver_inputs.noise,
                expected[min(index + 1, 3)].bfloat16(),
                rtol=0,
                atol=0,
            )
    for length in (7, 3, 7):
        request = builder.size(13, length, size.canvas)
        features = torch.full((length, 12), float(length))
        builder.store_conditioning(request, views, features)
        expected_context = torch.zeros(512, 12, dtype=torch.bfloat16)
        expected_context[:length] = features
        bound = builder.bind(request, views, {"video": sample}, schedules, 0)
        torch.testing.assert_close(
            bound.context, expected_context[None], rtol=0, atol=0
        )
