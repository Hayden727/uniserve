"""Wan capability composition and public component entry points."""

from torch import nn

from uniserve.model import ComponentEntry, EntryPoint

from .config import Config
from .serving import ServingDenoiser
from .text import Encoder
from .video import Decoder as VideoDecoder
from .video import Encoder as VideoEncoder
from .video import Postprocessor


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        if config.grid is None:
            self.transformer = ServingDenoiser(config)
        else:
            from .stdm import GridDenoiser

            self.transformer = GridDenoiser(config)
        if config.text_encoder is not None:
            self.text_encoder = Encoder(config.text_encoder)
        if config.video_encoder is not None:
            self.video_encoder = VideoEncoder(config.video_encoder)
        if config.video_decoder is not None:
            if config.grid is None:
                self.video_decoder = VideoDecoder(config.video_decoder)
                self.postprocessor = Postprocessor(frame_rate=24)
            else:
                from .stdm_video import (
                    Decoder,
                )
                from .stdm_video import (
                    Postprocessor as GridPostprocessor,
                )

                self.video_decoder = Decoder(
                    config.video_decoder, self.transformer.camera_video
                )
                self.postprocessor = GridPostprocessor(
                    self.transformer.camera_video
                )


def entry_points(config: Config):
    result = {
        "transformer": ComponentEntry("transformer", (EntryPoint("forward"),))
    }
    if config.text_encoder is not None:
        result["text_encoder"] = ComponentEntry(
            "text_encoder", (EntryPoint("encode"),)
        )
    if config.video_encoder is not None:
        result["video_encoder"] = ComponentEntry(
            "video_encoder", (EntryPoint("encode"),)
        )
    if config.video_decoder is not None:
        # One decoding call reconstructs and converts each independent clip.
        # Both numerical modules must share its single placement owner.
        result["video_decoder"] = ComponentEntry(
            "",
            (
                EntryPoint("video_decoder.decode"),
                EntryPoint("postprocessor.forward"),
            ),
        )
    return result
