"""Decode each grid camera's head and middle cells as one causal trajectory."""

import torch

from uniserve.media import video
from uniserve.model import VideoDecoder, VideoPostprocessor
from uniserve.model.camera import CameraVideo
from uniserve.nn.vae import LatentDecoder
from uniserve.tensors import BufferConfig, OutputLayout

from .vae import DecoderNetwork, Normalization
from .video import Postprocessor as RingPostprocessor
from .video import latent_size


class Decoder(VideoDecoder):
    def __init__(self, config, layout: CameraVideo):
        super().__init__(
            LatentDecoder(
                DecoderNetwork(config),
                normalization=Normalization(config),
                latent_shape=(
                    1,
                    config.z_dim,
                    layout.latent_frames_per_view,
                    None,
                    None,
                ),
            )
        )
        self.channels, self.layout = config.z_dim, layout

    def clips(self, num_frames):
        frames = self.layout.frames_per_view
        if type(num_frames) is not int or num_frames < 1 or num_frames % frames:
            raise ValueError(
                "camera video requires complete temporal trajectories"
            )
        return tuple(
            slice(start, start + frames)
            for start in range(0, num_frames, frames)
        )

    def output_layout(self, size):
        latent_size(size.frame)
        shape = (
            len(self.clips(size.num_frames)),
            1,
            3,
            self.layout.frames_per_view,
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

    def clip_output(self, size, clip):
        if clip not in self.clips(size.num_frames):
            raise ValueError("camera decode clip must cover one complete view")
        return video.Config(self.layout.frames_per_view, size.frame)

    def clip_input(self, output):
        if output.num_frames != self.layout.frames_per_view:
            raise ValueError(
                "camera decoder requires its configured temporal trajectory"
            )
        height, width = latent_size(output.frame)
        return BufferConfig(
            (
                1,
                self.channels,
                self.layout.latent_frames_per_view,
                height,
                width,
            ),
            torch.float32,
        )

    def unpack_latents(self, latent, clip, size, *, out):
        expected = self.clip_input(self.clip_output(size, clip))
        height, width = latent_size(size.frame)
        if size.num_frames != self.layout.num_frames or latent.shape != (
            1,
            self.channels,
            self.layout.grid.num_cells,
            height,
            width,
        ):
            raise ValueError(
                "camera decoder must read the complete native latent grid"
            )
        if out.shape != expected.shape or out.dtype != expected.dtype:
            raise ValueError(
                "camera decoder destination must match its clip input"
            )
        view = clip.start // self.layout.frames_per_view
        indices = torch.tensor(self.layout.cells(view), device=latent.device)
        out.copy_(latent.index_select(2, indices))


class Postprocessor(RingPostprocessor):
    def __init__(self, layout: CameraVideo):
        super().__init__(frame_rate=8)
        self.layout = layout

    def output_layout(self, size):
        if size.num_frames % self.layout.frames_per_view:
            raise ValueError(
                "camera pixels must contain complete per-view trajectories"
            )
        return VideoPostprocessor.output_layout(self, size)

    def workspace_buffers(self, size):
        self.output_layout(size)
        return {
            "rgb_frames": BufferConfig(
                (
                    self.layout.frames_per_view,
                    size.frame.height,
                    size.frame.width,
                    3,
                ),
                torch.uint8,
            )
        }
