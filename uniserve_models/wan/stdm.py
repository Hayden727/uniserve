"""Rectangular Wan grid computation with local cells and global attention."""

import math
from dataclasses import dataclass

import torch

from uniserve.distributed.tokens import TokenShard
from uniserve.media import image
from uniserve.model.camera import CameraCondition, CameraGrid, CameraVideo
from uniserve.model.denoiser import Condition
from uniserve.nn.attention.grid import GridLayout
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput

from .camera import CameraPoses, ucpe_input
from .serving import ServingDenoiser


@dataclass(frozen=True, slots=True)
class GridSize:
    """Complete latent grid, independent camera-video timeline and text size."""

    num_frames: int
    canvas: image.Config
    num_text_tokens: int
    grid: CameraGrid
    conditions: tuple[Condition, ...] = ()
    vision_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if not isinstance(self.grid, CameraGrid):
            raise TypeError("grid size requires a typed CameraGrid")
        if (
            type(self.num_text_tokens) is not int
            or not 1 <= self.num_text_tokens <= 512
            or type(self.num_frames) is not int
            or self.num_frames < 1
        ):
            raise ValueError("grid output and text lengths must be positive")
        if self.canvas.height % 8 or self.canvas.width % 8:
            raise ValueError("Wan grid raster must divide by eight")
        if self.conditions or self.vision_spans:
            raise ValueError(
                "grid conditioning uses explicit camera and latent fields"
            )

    @property
    def frames(self):
        return self.grid.num_cells

    @property
    def height(self):
        return self.canvas.height // 8

    @property
    def width(self):
        return self.canvas.width // 8

    @property
    def text_tokens(self):
        return self.num_text_tokens

    @property
    def condition_rows(self):
        return 0


def cell_tables(grid: CameraGrid):
    """Camera/column labels in the declared complete latent storage order."""
    if grid.order == "time_major":
        return torch.arange(grid.views).repeat(grid.columns), torch.arange(
            grid.columns
        ).repeat_interleave(grid.views)
    return torch.arange(grid.views).repeat_interleave(
        grid.columns
    ), torch.arange(grid.columns).repeat(grid.views)


class GridDenoiser(ServingDenoiser):
    """Native row/head-conditioned STDM through ordinary diffusion execution.

    Noise is drawn in full native order, then sliced. State and projections
    retain this rank's whole-cell interval; OutputLayout describes the complete
    native tensor and lets existing product transport gather the final result.
    """

    def __init__(self, config):
        if config.grid is None or config.ucpe is None:
            raise ValueError(
                "grid denoising requires explicit UCPE/grid configuration"
            )
        super().__init__(config)

    @property
    def camera_video(self):
        grid = self.config.grid
        latent_frames = grid.columns - 1
        return CameraVideo(grid, 4 * latent_frames - 3, latent_frames)

    def legal_frame_count(self, requested):
        total = self.camera_video.num_frames
        if type(requested) is not int or not 1 <= requested <= total:
            raise ValueError(
                "requested timeline exceeds the configured camera grid"
            )
        return total

    def make_size(
        self,
        num_frames,
        num_text_tokens,
        *,
        canvas,
        conditions=(),
        vision_spans=(),
    ):
        return self.make_camera_size(
            self.config.grid,
            num_frames,
            num_text_tokens,
            canvas=canvas,
            conditions=conditions,
            vision_spans=vision_spans,
        )

    def make_camera_size(
        self,
        grid,
        num_frames,
        num_text_tokens,
        *,
        canvas,
        conditions=(),
        vision_spans=(),
    ):
        if (
            grid != self.config.grid
            or num_frames != self.camera_video.num_frames
        ):
            raise ValueError(
                "camera grid and output timeline must match the deployment"
            )
        if canvas not in self.canvases.values():
            raise ValueError("Wan grid serving uses a 480 by 832 canvas")
        size = GridSize(
            num_frames, canvas, num_text_tokens, grid, conditions, vision_spans
        )
        self.latent_shape("video", size)
        return size

    def _cells(self, size):
        group = self.network.blocks[0].cam_self_attn.attn.exchange.group
        if size.frames % group.size:
            raise ValueError(
                "grid sequence ranks must own complete equal cell intervals"
            )
        if self.mesh.size("tp" if "tp" in self.mesh.axes else ()) != 1:
            raise ValueError("Wan grid projections currently require TP1")
        return TokenShard(size.frames, group)

    def output_layout(self, size):
        shape = self.latent_shape("video", size)
        return {
            "video": OutputLayout(
                shape,
                self.prediction_dtype,
                (
                    slice(0, 1),
                    slice(0, shape[1]),
                    self._cells(size).token_slice,
                    slice(0, shape[3]),
                    slice(0, shape[4]),
                ),
            )
        }

    def camera_buffers(self, size):
        local = self._cells(size).count
        _, ph, pw = self.config.transformer.patch_size
        tokens = local * (size.height // ph) * (size.width // pw)
        return {
            "clean": BufferConfig(
                (
                    1,
                    self.config.transformer.in_channels,
                    local,
                    size.height,
                    size.width,
                ),
                self.prediction_dtype,
            ),
            "clean_mask": BufferConfig((1, 1, local, 1, 1), torch.bool),
            "viewmats": BufferConfig((1, tokens, 4, 4), self.prediction_dtype),
            "camera_cos": BufferConfig(
                (tokens, self.config.transformer.head_dim), torch.float32
            ),
            "camera_sin": BufferConfig(
                (tokens, self.config.transformer.head_dim), torch.float32
            ),
        }

    def state_buffers(self, size):
        return {
            "video": BufferConfig(
                self.camera_buffers(size)["clean"].shape, self.prediction_dtype
            ),
            **self.camera_buffers(size),
        }

    def prepare_latents(self, sizes, *, noise, state, constants, workspace):
        for index, size in enumerate(sizes):
            draw = noise["video"][index]
            if draw.shape != self.noise_shape("video", size):
                raise ValueError(
                    "grid noise must preserve the complete native CPU draw"
                )
            state["video"][index].copy_(self._cells(size).local(draw, dim=2))

    def output_camera(self, size, condition: CameraCondition):
        if (
            condition.grid != size.grid
            or condition.pose_space != "prepared_camera_to_world"
        ):
            raise ValueError(
                "grid serving requires prepared cameras and an explicit "
                "complete grid"
            )
        camera, column = cell_tables(size.grid)
        clean = tuple(torch.where((camera == 0) | (column == 0))[0].tolist())
        if set(condition.clean_cells) != set(clean):
            raise ValueError(
                "grid conditioning must preserve the reference row "
                "and head column"
            )
        if condition.cell_time_index is None or not torch.equal(
            condition.cell_time_index.long(), column
        ):
            raise ValueError(
                "grid timestamps must equal their explicit temporal columns"
            )
        if condition.clean_latents is None:
            raise ValueError(
                "grid serving requires prepared clean latent cells"
            )
        return super().output_camera(size, condition)

    def prepare_camera(self, size, condition, *, out):
        self._check_camera_outputs(size, out)
        canonical = self.output_camera(size, condition)
        cells = self._cells(size)
        device = out["clean"].device
        _, ph, pw = self.config.transformer.patch_size
        if ph != pw:
            raise ValueError("UCPE grid rays require square spatial patches")
        poses = CameraPoses(
            cells.local(canonical.poses).unsqueeze(0).to(device, torch.float32),
            cells.local(canonical.intrinsics)
            .unsqueeze(0)
            .to(device, torch.float32),
        )
        camera = ucpe_input(
            poses,
            height=size.height * 8,
            width=size.width * 8,
            token_grid=(cells.count, size.height // ph, size.width // pw),
            dtype=self.prediction_dtype,
            patch_size=ph * 8,
        )
        out["viewmats"].copy_(camera.viewmats)
        rotary = self.network.rotary_embedding(
            (cells.count, size.height, size.width),
            device=device,
            cell_time_index=cells.local(canonical.cell_time_index),
        )
        for name, value in zip(
            ("camera_cos", "camera_sin"), rotary, strict=True
        ):
            out[name].copy_(value)
        out["clean"].copy_(
            cells.local(canonical.clean_latents.unsqueeze(0), dim=2)
        )
        mask = torch.zeros(size.frames, dtype=torch.bool)
        mask[list(canonical.clean_cells)] = True
        out["clean_mask"].copy_(cells.local(mask).reshape(1, 1, -1, 1, 1))

    def input_frame_indices(self, size, condition, frame_count):
        raise ValueError(
            "grid RGB must be prepared through the native temporal encoder "
            "before admission"
        )

    def constant_buffers(self, size):
        fields = super().constant_buffers(size)
        cell_tokens = math.prod(self._token_grid(size)[1:])
        tokens = size.frames * cell_tokens
        for name, count in (
            ("column_queries", tokens),
            ("column_keys", tokens - size.grid.columns * cell_tokens),
            ("column_restore", tokens),
        ):
            fields[f"grid_{name}"] = BufferConfig((count,), torch.int64)
        if size.grid.order == "time_major":
            fields["grid_canonical"] = BufferConfig((tokens,), torch.int64)
        if self.config.transformer.vsa is not None:
            tile = self.config.transformer.vsa.tile_shape
            count = math.prod(
                (n + t - 1) // t
                for n, t in zip(self._token_grid(size), tile, strict=True)
            )
            fields["grid_tile_allowed"] = BufferConfig(
                (count, count), torch.bool
            )
        # Spatial PRoPE repeats the same per-cell raster on every rank.
        for name, field in tuple(fields.items()):
            if name.startswith(("camera_x_", "camera_y_")):
                fields[name] = BufferConfig(
                    (
                        1,
                        1,
                        self._cells(size).count * cell_tokens,
                        field.shape[-1],
                    ),
                    field.dtype,
                )
        return fields

    def prepare_constants(self, size, *, out):
        # Prepare global VSA indices and the frequencies once. Spatial camera
        # coefficients are local, so prepare them independently below.
        spatial = {
            name for name in out if name.startswith(("camera_x_", "camera_y_"))
        }
        if spatial:
            # The base helper would copy full-grid spatial coefficients.
            raise ValueError(
                "rectangular STDM currently requires camera use_2d_rope=false"
            )
        super().prepare_constants(size, out=out)
        camera, column = cell_tables(size.grid)
        layout = GridLayout.from_cells(
            camera,
            column,
            math.prod(self._token_grid(size)[1:]),
            device=out["time_frequencies"].device,
        )
        for name in (
            "canonical",
            "column_queries",
            "column_keys",
            "column_restore",
        ):
            value = getattr(layout, name)
            if value is not None:
                out[f"grid_{name}"].copy_(value)
        if "grid_tile_allowed" in out:
            tiles_per_cell = out["grid_tile_allowed"].shape[0] // size.frames
            allowed = (
                (column[:, None] == column[None, :])
                | (camera[:, None] == camera[None, :])
                | (camera[None, :] == 0)
            )
            out["grid_tile_allowed"].copy_(
                allowed.repeat_interleave(tiles_per_cell, 0).repeat_interleave(
                    tiles_per_cell, 1
                )
            )

    def bind_camera_inputs(
        self, *, latents, sizes, step, text_features, state, noise
    ):
        local_noise = (self._cells(sizes[0]).local(noise[0], dim=2),)
        return super().bind_camera_inputs(
            latents=latents,
            sizes=sizes,
            step=step,
            text_features=text_features,
            state=state,
            noise=local_noise,
        )

    def forward(self, inputs, *, state, constants, workspace):
        size = inputs.sizes[0]
        latent = inputs.latents["video"][0]
        sample, conditions = latent.tensor, latent.solver_inputs
        output = self.output_layout(size)["video"]
        if (
            sample.shape != self.camera_buffers(size)["clean"].shape
            or conditions is None
        ):
            raise ValueError(
                "grid sample must match its global output's local cell slice"
            )
        grid = GridLayout(
            size.grid.views,
            size.grid.columns,
            math.prod(self._token_grid(size)[1:]),
            constants.get("grid_canonical"),
            constants["grid_column_queries"],
            constants["grid_column_keys"],
            constants["grid_column_restore"],
        )
        tiles = self._tile_inputs(size, constants).get("main")
        keep = None
        if tiles is not None:
            # Largest permitted row: union of own camera, column, reference.
            maximum_cells = size.grid.views + 2 * size.grid.columns - 2
            maximum_tiles = maximum_cells * tiles.num_tiles // size.frames
            keep = tiles.keep(
                self.config.transformer.vsa.sparsity,
                allowed_count=maximum_tiles,
            )
        current = self._network_input(sample, conditions)
        velocity = self.network(
            current,
            inputs.context,
            latent.timestep.reshape(1),
            inputs.camera.viewmats,
            inputs.rotary,
            time_frequencies=constants["time_frequencies"],
            grid=grid,
            tiles=tiles,
            tile_allowed=constants.get("grid_tile_allowed"),
            tile_keep=keep,
            contiguous_tokens=self._cells(size).group.size > 1,
        )
        sigma = (latent.timestep.float() / self.config.train_timesteps).to(
            sample.dtype
        )
        x0 = torch.where(
            conditions.clean_mask,
            conditions.clean,
            sample - sigma * velocity.to(sample.dtype),
        )
        return {"video": (TensorOutput(x0, output),)}
