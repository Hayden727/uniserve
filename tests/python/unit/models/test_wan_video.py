"""Wan video clips preserve segment order and checkpoint values."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from uniserve.media import image, video
from uniserve.tensors import OutputLayout, TensorOutput
from uniserve_models.wan.vae import VAEConfig
from uniserve_models.wan.video import Decoder, Encoder, Postprocessor

pytestmark = pytest.mark.unit


def test_independent_clip_layouts_and_native_pixel_rounding():
    config = VAEConfig(base_dim=8, num_res_blocks=1)
    encoder, decoder = Encoder(config), Decoder(config)
    size = video.Config(26, image.Config(16, 24))
    assert encoder.clips(26) == (slice(0, 13), slice(13, 26))
    assert encoder.latent_slices(26) == (slice(0, 4), slice(4, 8))
    assert encoder.output_layout(size)["video"].shape == (48, 16)
    assert decoder.output_layout(size)["video"].shape == (2, 1, 3, 13, 16, 24)
    native = torch.arange(1 * 16 * 8 * 2 * 3).reshape(1, 16, 8, 2, 3).bfloat16()
    out = torch.empty((1, 16, 4, 2, 3))
    decoder.unpack_latents(native, slice(13, 26), size, out=out)
    torch.testing.assert_close(out, native[:, :, 4:].float(), rtol=0, atol=0)
    # Byte 127 checks division order at the BF16 midpoint.
    pixels = torch.full((13, 16, 24, 3), 127, dtype=torch.uint8)
    actual = encoder.unpack_pixels(pixels, slice(0, 13), 26)
    expected = (torch.tensor(127.0) / torch.tensor(127.5) - 1).bfloat16()
    assert actual.shape == (1, 3, 13, 16, 24)
    assert torch.all(actual == expected)
    for count in (0, 1, 14, 25):
        with pytest.raises(ValueError, match="13-frame"):
            encoder.clips(count)


def test_postprocessor_publishes_current_independent_clip_and_truncates():
    model = Postprocessor(frame_rate=24)
    size = video.Config(26, image.Config(8, 8))
    workspace = {"rgb_frames": torch.empty((13, 8, 8, 3), dtype=torch.uint8)}
    saved = []
    for number, value in ((0, 0.0), (1, 1.0), (0, 0.0)):
        data = torch.full((1, 3, 13, 8, 8), value)
        output = model(
            (
                TensorOutput(
                    data,
                    OutputLayout(
                        data.shape,
                        data.dtype,
                        tuple(slice(0, n) for n in data.shape),
                    ),
                ),
            ),
            clips=(slice(number * 13, (number + 1) * 13),),
            sizes=(size,),
            state={},
            constants={},
            workspace=workspace,
        )[0]
        assert output.layout.shape == (26, 8, 8, 3)
        assert output.tensor.data_ptr() == workspace["rgb_frames"].data_ptr()
        assert torch.all(output.tensor == (127 if value == 0 else 255))
        saved.append(output.tensor.clone())
    torch.testing.assert_close(saved[0], saved[2], rtol=0, atol=0)
    assert model.state_buffers(size) == model.constant_buffers(size) == {}


def test_catalog_loads_both_vae_checkpoints_with_distinct_precision(tmp_path):
    from uniserve.model import VideoDecoder, VideoEncoder, VideoPostprocessor
    from uniserve_models import loading
    from uniserve_models.wan.config import TransformerConfig
    from uniserve_models.wan.loading import encoder_parameter_sources
    from uniserve_models.wan.transformer import PluckerTransformer
    from uniserve_models.wan.vae import DecoderNetwork, EncoderNetwork
    from uniserve_worker.bootstrap.components import (
        describe_components,
        media_components,
    )
    from uniserve_worker.protocol.call import MediaCall

    vae = VAEConfig(base_dim=8, num_res_blocks=1)
    transformer = TransformerConfig(
        num_layers=1,
        num_heads=2,
        head_dim=8,
        ffn_dim=24,
        text_dim=12,
        freq_dim=8,
        scale_bands=2,
    )
    save_file(
        PluckerTransformer(transformer).bfloat16().state_dict(),
        tmp_path / "model.safetensors",
    )
    source_encoder = EncoderNetwork(vae).bfloat16()
    source_decoder = DecoderNetwork(vae).float()
    (tmp_path / "vae_encoder").mkdir()
    (tmp_path / "vae").mkdir()
    torch.save(
        {
            original: source_encoder.state_dict()[name]
            for name, original in encoder_parameter_sources(
                source_encoder
            ).items()
        },
        tmp_path / "vae_encoder" / "model.pth",
    )
    save_file(
        source_decoder.state_dict(), tmp_path / "vae" / "model.safetensors"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WanCameraPipeline"],
                "transformer": asdict(transformer),
                "video_encoder": asdict(vae),
                "video_decoder": asdict(vae),
            }
        )
    )
    resolved = loading.read_config(tmp_path)
    loaded = loading.load_model(resolved, device="cpu").model
    assert isinstance(loaded.video_encoder, VideoEncoder)
    assert isinstance(loaded.video_decoder, VideoDecoder)
    assert isinstance(loaded.postprocessor, VideoPostprocessor)
    for target, source in (
        (loaded.video_encoder.encoder.encoder, source_encoder),
        (loaded.video_decoder.decoder.decoder, source_decoder),
    ):
        for name, value in target.state_dict().items():
            torch.testing.assert_close(
                value, source.state_dict()[name], rtol=0, atol=0
            )
    assert next(loaded.video_encoder.parameters()).dtype == torch.bfloat16
    assert next(loaded.video_decoder.parameters()).dtype == torch.float32
    calls = describe_components(loaded)
    assert calls["video_encoder"][0].entry_point.method == "encode"
    assert calls["video_decoder"][0].entry_point.method == "decode"
    assert calls["video_decoder"][1].entry_point.method == "forward"
    assert calls["video_decoder"][1].module is loaded.postprocessor
    assert media_components(loaded)[MediaCall.VIDEO_DECODING] == "video_decoder"
