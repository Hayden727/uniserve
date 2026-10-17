"""Independent Wan camera-ring segments through shared video capabilities."""

import torch

from uniserve.media import image, video
from uniserve.model import VideoDecoder, VideoEncoder, VideoPostprocessor
from uniserve.nn.vae import LatentDecoder, LatentEncoder
from uniserve.tensors import BufferConfig, OutputLayout

from .vae import DecoderNetwork, EncoderNetwork, Normalization, VAEConfig


def clips(num_frames: int) -> tuple[slice, ...]:
    if type(num_frames) is not int or num_frames < 1 or num_frames % 13:
        raise ValueError(
            "Wan camera videos require independent 13-frame segments"
        )
    return tuple(slice(start, start + 13) for start in range(0, num_frames, 13))


def latent_size(frame: image.Config):
    if frame.height % 8 or frame.width % 8:
        raise ValueError("Wan VAE frame dimensions must be divisible by eight")
    return frame.height // 8, frame.width // 8


class Encoder(VideoEncoder):
    def __init__(self, config: VAEConfig):
        super().__init__(
            LatentEncoder(
                EncoderNetwork(config), normalization=Normalization(config)
            )
        )
        self.channels = config.z_dim
        self.register_buffer(
            "pixel_divisor",
            torch.tensor(127.5, device="cpu", dtype=torch.float32),
            persistent=False,
        )

    def clips(self, num_frames):
        return clips(num_frames)

    def latent_slices(self, num_frames):
        return tuple(
            slice(index * 4, (index + 1) * 4)
            for index, _ in enumerate(clips(num_frames))
        )

    def latent_size(self, frame):
        return latent_size(frame)

    def output_layout(self, size: video.Config):
        height, width = latent_size(size.frame)
        shape = (
            len(clips(size.num_frames)) * 4 * height * width,
            self.channels,
        )
        return {
            "video": OutputLayout(
                shape,
                torch.bfloat16,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }

    def unpack_pixels(self, pixels, clip, num_frames):
        if clip not in clips(num_frames):
            raise ValueError(
                "Wan encoding requires a complete independent segment"
            )
        value = pixels.permute(3, 0, 1, 2).unsqueeze(0).float()
        # Device-tensor division preserves native rounding at byte value 127.
        return (
            value.div_(self.pixel_divisor)
            .sub_(1.0)
            .bfloat16()
            .contiguous(memory_format=torch.channels_last_3d)
        )

    def pack_latents(self, latents):
        return latents.permute(0, 2, 3, 4, 1).reshape(-1, self.channels)


class Decoder(VideoDecoder):
    def __init__(self, config: VAEConfig):
        super().__init__(
            LatentDecoder(
                DecoderNetwork(config),
                normalization=Normalization(config),
                latent_shape=(1, config.z_dim, 4, None, None),
            )
        )
        self.channels = config.z_dim

    def clips(self, num_frames):
        return clips(num_frames)

    def output_layout(self, size: video.Config):
        latent_size(size.frame)
        shape = (
            len(clips(size.num_frames)),
            1,
            3,
            13,
            size.frame.height,
            size.frame.width,
        )
        return {
            "video": OutputLayout(
                shape,
                torch.float32,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
                value_range=(-1, 1),
            )
        }

    def clip_output(self, size: video.Config, clip: slice):
        if clip not in clips(size.num_frames):
            raise ValueError(
                "Wan decoding requires a complete independent segment"
            )
        return video.Config(13, size.frame)

    def clip_input(self, output: video.Config):
        if output.num_frames != 13:
            raise ValueError("one Wan segment decodes to thirteen frames")
        height, width = latent_size(output.frame)
        return BufferConfig((1, self.channels, 4, height, width), torch.float32)

    def unpack_latents(self, latent, clip, size, *, out):
        self.clip_output(size, clip)
        height, width = latent_size(size.frame)
        frames = len(clips(size.num_frames)) * 4
        if latent.shape != (1, self.channels, frames, height, width):
            raise ValueError(
                "Wan denoising latent must preserve native NCTHW layout"
            )
        if (
            out.shape != self.clip_input(video.Config(13, size.frame)).shape
            or out.dtype != torch.float32
        ):
            raise ValueError(
                "Wan decoder output buffer must match its clip input"
            )
        start = clip.start // 13 * 4
        out.copy_(latent[:, :, start : start + 4])


class Postprocessor(VideoPostprocessor):
    """Preserve native clipping and truncation for independent Wan segments."""

    def output_layout(self, size):
        clips(size.num_frames)
        return super().output_layout(size)

    def workspace_buffers(self, size):
        self.output_layout(size)
        return {
            "rgb_frames": BufferConfig(
                (13, size.frame.height, size.frame.width, 3), torch.uint8
            )
        }

    def rgb(self, decoded):
        return ((decoded.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
