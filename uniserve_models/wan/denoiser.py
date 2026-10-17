"""Wan predict-x0 computation for the shared diffusion step and runner."""

import math

import torch

from uniserve.diffusion import ReNoiseInput, ReNoiseSolver, RungGrid
from uniserve.model import Denoiser as BaseDenoiser
from uniserve.model import LatentInput
from uniserve.nn.attention.vsa.tiles import TileLayout
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput

from .camera import PluckerInput, UCPEInput
from .config import Config
from .inputs import DenoiserInput, LatentSize
from .transformer import PluckerTransformer
from .ucpe import UcpeTransformer


class Denoiser(BaseDenoiser[DenoiserInput, LatentSize]):
    def __init__(self, config: Config):
        super().__init__(
            modalities=("video",),
            prediction_dtype=torch.bfloat16,
            solver=ReNoiseSolver(),
            grids={
                "video": RungGrid(
                    config.rungs,
                    config.flow_shift,
                    config.train_timesteps,
                    time_parameterization="sigma",
                )
            },
        )
        self.config = config
        self.network = (
            PluckerTransformer(config.transformer)
            if config.ucpe is None
            else UcpeTransformer(
                config.transformer,
                config.ucpe.camera,
                main_compress_gate=config.ucpe.main_compress_gate,
                grid_attention=config.grid is not None,
            )
        )

    def latent_shape(self, modality: str, size: LatentSize) -> tuple[int, ...]:
        if modality != "video":
            raise ValueError("Wan predicts video latents")
        extents = size.frames, size.height, size.width
        if any(
            a % b
            for a, b in zip(
                extents, self.config.transformer.patch_size, strict=True
            )
        ):
            raise ValueError("latent dimensions must be divisible by patches")
        return (1, self.config.transformer.in_channels, *extents)

    def noise_shape(self, modality: str, size: LatentSize) -> tuple[int, ...]:
        return self.latent_shape(modality, size)

    @property
    def latent_format(self):
        """Declared layout, or preservation for direct diagnostic inputs."""
        return (
            None if self.config.ucpe is None else self.config.ucpe.latent_format
        )

    def _network_input(self, sample, conditions):
        current = torch.where(conditions.clean_mask, conditions.clean, sample)
        if self.latent_format is not None:
            # Clean pinning inherits the clean tensor's strides. Patch embedding
            # propagates them to the camera residual, where native nn.Linear's
            # bias fusion depends on contiguity. Canonicalize the numerical
            # input without changing the runtime's borrowed request storage.
            current = current.contiguous(
                memory_format=self.latent_format.memory_format
            )
        return current

    def make_schedules(self, *, device):
        return {"video": self.grids["video"].schedule(device)}

    def constant_buffers(self, size: LatentSize):
        self.latent_shape("video", size)
        tokens = math.prod((size.frames, size.height, size.width)) // math.prod(
            self.config.transformer.patch_size
        )
        result = {
            "time_frequencies": BufferConfig(
                (self.config.transformer.freq_dim // 2,), torch.float32
            ),
        }
        if self.config.ucpe is None:
            result.update(
                {
                    name: BufferConfig(
                        (tokens, self.config.transformer.head_dim),
                        torch.float32,
                    )
                    for name in ("cos", "sin")
                }
            )
        elif self.config.ucpe.camera.use_2d_rope:
            result.update(
                {
                    f"camera_{axis}_{name}": BufferConfig(
                        (1, 1, tokens, self.config.ucpe.camera.head_dim // 8),
                        self.prediction_dtype,
                    )
                    for axis in ("x", "y")
                    for name in ("cos", "sin")
                }
            )
        grid = self._token_grid(size)
        for branch, config in self._tile_configs().items():
            tiles = math.prod(
                (extent + tile - 1) // tile
                for extent, tile in zip(grid, config.tile_shape, strict=True)
            )
            result.update(
                {
                    f"{branch}_vsa_{name}": BufferConfig((tokens,), torch.int64)
                    for name in ("permutation", "slots", "restore")
                }
            )
            result[f"{branch}_vsa_valid_counts"] = BufferConfig(
                (tiles,), torch.int32
            )
        return result

    def _token_grid(self, size):
        return tuple(
            extent // patch
            for extent, patch in zip(
                (size.frames, size.height, size.width),
                self.config.transformer.patch_size,
                strict=True,
            )
        )

    def _tile_configs(self):
        return {
            branch: config
            for branch, config in (
                ("main", self.config.transformer.vsa),
                (
                    "camera",
                    None
                    if self.config.ucpe is None
                    else self.config.ucpe.camera.vsa,
                ),
            )
            if config is not None
        }

    def _tile_inputs(self, size, constants):
        # These records borrow precomputed runtime-owned tables. Rebuilding
        # the immutable Python record does not allocate tensors, read device
        # values, synchronize, or cache request state during graph capture.
        return {
            branch: TileLayout(
                self._token_grid(size),
                config.tile_shape,
                *(
                    constants[f"{branch}_vsa_{name}"]
                    for name in (
                        "permutation",
                        "slots",
                        "restore",
                        "valid_counts",
                    )
                ),
            )
            for branch, config in self._tile_configs().items()
        }

    def prepare_constants(self, size: LatentSize, *, out):
        device = out["time_frequencies"].device
        extents = size.frames, size.height, size.width
        if self.config.ucpe is None:
            cos, sin = self.network.rotary_embedding(extents, device=device)
            out["cos"].copy_(cos)
            out["sin"].copy_(sin)
        elif self.config.ucpe.camera.use_2d_rope:
            rotary = self.network.camera_rotary_embedding(
                extents, device=device, dtype=self.prediction_dtype
            )
            for axis, values in zip(("x", "y"), rotary, strict=True):
                for name, value in zip(("cos", "sin"), values, strict=True):
                    out[f"camera_{axis}_{name}"].copy_(value)
        out["time_frequencies"].copy_(
            self.network.condition_embedder.time_embedder.frequencies(
                device=device
            )
        )
        for branch, config in self._tile_configs().items():
            layout = TileLayout.from_grid(
                self._token_grid(size), config.tile_shape, device=device
            )
            for name in ("permutation", "slots", "restore", "valid_counts"):
                out[f"{branch}_vsa_{name}"].copy_(getattr(layout, name))

    def prepare_latents(self, sizes, *, noise, state, constants, workspace):
        for index, size in enumerate(sizes):
            shape = self.noise_shape("video", size)
            if tuple(noise["video"][index].shape) != shape:
                raise ValueError(
                    "Wan noise must retain the complete native draw shape"
                )
            state["video"][index].copy_(noise["video"][index])

    def bind_inputs(
        self,
        *,
        sample,
        timestep,
        step,
        size,
        context,
        camera_rays,
        scale_embedding,
        noise,
        clean,
        clean_mask,
    ):
        """Borrow request tensors; the caller owns all buffers and RNG draws."""
        return DenoiserInput(
            {
                "video": (
                    LatentInput(
                        sample, timestep, ReNoiseInput(noise, clean, clean_mask)
                    ),
                )
            },
            (size,),
            step,
            context,
            PluckerInput(camera_rays, scale_embedding),
        )

    def bind_ucpe_inputs(
        self,
        *,
        sample,
        timestep,
        step,
        size,
        context,
        viewmats,
        rotary,
        noise,
        clean,
        clean_mask,
    ):
        """Borrow UCPE ray frames and request-specific temporal phases."""
        return DenoiserInput(
            {
                "video": (
                    LatentInput(
                        sample, timestep, ReNoiseInput(noise, clean, clean_mask)
                    ),
                )
            },
            (size,),
            step,
            context,
            UCPEInput(viewmats),
            rotary,
        )

    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        size = inputs.sizes[0]
        latent = inputs.latents["video"][0]
        sample, conditions = latent.tensor, latent.solver_inputs
        shape = self.latent_shape("video", size)
        if tuple(sample.shape) != shape or conditions is None:
            raise ValueError(
                "Wan requires native sample shape and explicit "
                "clean/noise inputs"
            )
        current = self._network_input(sample, conditions)
        tiles = self._tile_inputs(size, constants)
        if self.config.ucpe is None:
            if not isinstance(inputs.camera, PluckerInput):
                raise ValueError("Plucker requires ray and scale inputs")
            velocity = self.network(
                current,
                inputs.context,
                latent.timestep.reshape(1),
                inputs.camera.rays,
                inputs.camera.scale_embedding,
                (constants["cos"], constants["sin"]),
                time_frequencies=constants["time_frequencies"],
                tiles=tiles.get("main"),
            )
        else:
            if (
                not isinstance(inputs.camera, UCPEInput)
                or inputs.rotary is None
            ):
                raise ValueError("UCPE requires ray frames and temporal phases")
            camera_rotary = (
                tuple(
                    tuple(
                        constants[f"camera_{axis}_{name}"]
                        for name in ("cos", "sin")
                    )
                    for axis in ("x", "y")
                )
                if self.config.ucpe.camera.use_2d_rope
                else None
            )
            velocity = self.network(
                current,
                inputs.context,
                latent.timestep.reshape(1),
                inputs.camera.viewmats,
                inputs.rotary,
                camera_rotary,
                time_frequencies=constants["time_frequencies"],
                tiles=tiles.get("main"),
                camera_tiles=tiles.get("camera"),
            )
        sigma = (latent.timestep.float() / self.config.train_timesteps).to(
            sample.dtype
        )
        x0 = torch.where(
            conditions.clean_mask,
            conditions.clean,
            sample - sigma * velocity.to(sample.dtype),
        )
        layout = OutputLayout(
            shape, x0.dtype, tuple(slice(0, extent) for extent in shape)
        )
        return {"video": (TensorOutput(x0, layout),)}
