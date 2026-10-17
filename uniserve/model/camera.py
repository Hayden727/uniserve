"""Camera-conditioned video computation over caller-owned numerical views."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from uniserve.tensors import BufferConfig

from .denoiser import InputT, VideoDenoiser, VideoSizeT
from .inputs import LatentInput


@dataclass(frozen=True, slots=True)
class CameraGrid:
    """The camera and temporal axes of a complete latent-cell domain.

    ``view_major`` stores every column of a view before the next view:
    ``cell = view * columns + column``. ``time_major`` stores every view of
    a column before the next column: ``cell = column * views + view``.
    The order describes tensor storage; it does not reorder borrowed data or
    infer temporal RoPE positions from cell indices.
    """

    views: int
    columns: int
    order: Literal["view_major", "time_major"]

    def __post_init__(self):
        if (
            type(self.views) is not int
            or type(self.columns) is not int
            or self.views < 1
            or self.columns < 1
        ):
            raise ValueError("camera grid axes require positive integers")
        if self.order not in {"view_major", "time_major"}:
            raise ValueError("camera order must be view_major or time_major")

    @property
    def num_cells(self) -> int:
        """Number of cells in the complete, unsharded latent grid."""
        return self.views * self.columns


@dataclass(frozen=True, slots=True)
class CameraVideo:
    """Camera-major clips decoded from leading temporal cells of each view."""

    grid: CameraGrid
    frames_per_view: int
    latent_frames_per_view: int

    def __post_init__(self):
        if (
            not isinstance(self.grid, CameraGrid)
            or type(self.frames_per_view) is not int
            or self.frames_per_view < 1
            or type(self.latent_frames_per_view) is not int
            or not 1 <= self.latent_frames_per_view <= self.grid.columns
        ):
            raise ValueError("camera video clips must fit their complete grid")

    @property
    def num_frames(self):
        return self.grid.views * self.frames_per_view

    def cells(self, view: int) -> tuple[int, ...]:
        if type(view) is not int or not 0 <= view < self.grid.views:
            raise ValueError("camera video view is outside its grid")
        return tuple(
            view * self.grid.columns + column
            if self.grid.order == "view_major"
            else column * self.grid.views + view
            for column in range(self.latent_frames_per_view)
        )


@dataclass(frozen=True, slots=True)
class CameraCondition:
    """Borrow one ordered camera field and its fixed clean condition.

    Poses are N,4,4 OpenCV transforms; intrinsics are shared 3,3 or per-pose
    N,3,3 pixel matrices. Raw world-to-camera poses retain all input frames.
    Prepared camera-to-world poses already follow the latent-cell order and
    have their reference frame and distance scale resolved. ``clean_cells``
    indexes that latent-cell order, independent of the raw frame count.

    All tensors are CPU views supplied before graph capture. Clean latents,
    when supplied, are C,F,H,W; otherwise the caller encodes the selected
    RGB frames and fills clean storage through ``store_clean``. Scene scale
    is a one-element tensor in metres. Temporal indices are one integer per
    latent cell. Preparation must not mutate these borrowed inputs.

    ``grid`` declares the complete latent-cell domain, independently of raw
    pose/frame counts. Worker requests always supply it. Direct numerical
    callers may omit it when their model already supplies the cell layout;
    no grid is inferred from a raw camera timeline.
    """

    poses: torch.Tensor
    intrinsics: torch.Tensor
    clean_cells: tuple[int, ...]
    pose_space: Literal["raw_world_to_camera", "prepared_camera_to_world"]
    clean_latents: torch.Tensor | None = None
    scene_scale: torch.Tensor | None = None
    cell_time_index: torch.Tensor | None = None
    grid: CameraGrid | None = None

    def __post_init__(self):
        if self.pose_space not in {
            "raw_world_to_camera",
            "prepared_camera_to_world",
        }:
            raise ValueError(
                "camera poses require an explicit coordinate space"
            )
        if (
            self.poses.ndim != 3
            or self.poses.shape[1:] != (4, 4)
            or not self.poses.shape[0]
            or self.intrinsics.shape
            not in ((3, 3), (self.poses.shape[0], 3, 3))
        ):
            raise ValueError(
                "camera poses and pixel intrinsics require N,4,4 / N,3,3"
            )
        values = (
            self.poses,
            self.intrinsics,
            self.clean_latents,
            self.scene_scale,
        )
        for value in values:
            if value is not None and (
                value.device.type != "cpu"
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    "camera conditions require finite floating CPU tensors"
                )
        if bool((self.intrinsics[..., (0, 1), (0, 1)] <= 0).any()):
            raise ValueError("camera focal lengths must be positive")
        if self.clean_latents is not None and (
            self.clean_latents.ndim != 4 or min(self.clean_latents.shape) < 1
        ):
            raise ValueError("clean latents require C,F,H,W")
        if self.scene_scale is not None and (
            self.scene_scale.shape != (1,)
            or bool((self.scene_scale <= 0).any())
        ):
            raise ValueError(
                "scene scale requires one positive value in metres"
            )
        if (
            not isinstance(self.clean_cells, tuple)
            or any(
                type(cell) is not int or cell < 0 for cell in self.clean_cells
            )
            or len(set(self.clean_cells)) != len(self.clean_cells)
        ):
            raise ValueError(
                "clean cells must be distinct nonnegative integer indices"
            )
        if self.cell_time_index is not None and (
            self.cell_time_index.device.type != "cpu"
            or self.cell_time_index.dtype not in (torch.int32, torch.int64)
            or self.cell_time_index.ndim != 1
            or bool((self.cell_time_index < 0).any())
        ):
            raise ValueError("cell timestamps require nonnegative CPU integers")
        if self.grid is not None:
            if not isinstance(self.grid, CameraGrid):
                raise ValueError("camera conditions require a typed CameraGrid")
            cells = self.grid.num_cells
            if (
                self.pose_space == "prepared_camera_to_world"
                and self.poses.shape[0] != cells
            ):
                raise ValueError(
                    "prepared camera pose count must match the latent grid"
                )
            if (
                self.clean_latents is not None
                and self.clean_latents.shape[1] != cells
            ):
                raise ValueError("clean latents must contain every grid cell")
            if any(cell >= cells for cell in self.clean_cells):
                raise ValueError("clean cells must lie inside the camera grid")
            if (
                self.cell_time_index is not None
                and self.cell_time_index.shape != (cells,)
            ):
                raise ValueError("cell timestamps must contain every grid cell")


class CameraDenoiser(VideoDenoiser[InputT, VideoSizeT]):
    """A video denoiser with camera fields and independent sampling draws.

    Camera buffers describe one request without a runtime batch axis. Their
    allocation, retention, placement, staging and lifetime belong to callers.
    This capability performs only numerical preparation and input binding.
    """

    @property
    def camera_video(self) -> CameraVideo | None:
        """Explicit per-view decoding, or the ordinary timeline contract."""
        return None

    def make_camera_size(
        self,
        grid: CameraGrid,
        num_frames: int,
        num_text_tokens: int,
        *,
        canvas,
        conditions=(),
        vision_spans=(),
    ):
        """Resolve the latent grid separately from its output timeline."""
        return self.make_size(
            num_frames,
            num_text_tokens,
            canvas=canvas,
            conditions=conditions,
            vision_spans=vision_spans,
        )

    def camera_buffers(self, size: VideoSizeT) -> Mapping[str, BufferConfig]:
        """Describe the request fields filled by ``prepare_camera``."""
        raise NotImplementedError

    def output_camera(
        self, size: VideoSizeT, condition: CameraCondition
    ) -> CameraCondition:
        """Return canonical CPU cameras in the generated latent-cell order.

        Poses have ``prepared_camera_to_world`` semantics and preserve the
        exact relative frame, normalization and intrinsics used to prepare
        the denoiser. This numerical metadata operation uses no parameters;
        host callers may invoke it on a weightless model description.
        """
        raise NotImplementedError

    def sampling_noise_shapes(
        self, size: VideoSizeT
    ) -> tuple[tuple[int, ...], ...]:
        """Independent FP32 draws after initial modality noise, in order.

        Each shape is a separate draw from the request's CPU generator;
        concatenating them before drawing need not preserve the native RNG.
        Runtime selects and stages a step's draw before numerical evaluation.
        """
        raise NotImplementedError

    def prepare_camera(
        self,
        size: VideoSizeT,
        condition: CameraCondition,
        *,
        out: Mapping[str, torch.Tensor],
    ) -> None:
        """Fill camera fields on their caller-selected common device.

        Absent clean latents initialize clean storage to zero. The caller
        must then encode the selected frames and fill it before denoising.
        Host-only pose normalization may precede device ray arithmetic.
        """
        raise NotImplementedError

    def input_frame_indices(
        self, size: VideoSizeT, condition: CameraCondition, frame_count: int
    ) -> tuple[int, ...]:
        """Select and order RGB frames for the numerical VideoEncoder."""
        raise NotImplementedError

    def store_clean(
        self, size: VideoSizeT, encoded_rows: torch.Tensor, *, out: torch.Tensor
    ) -> None:
        """Rearrange encoded frame-major rows into borrowed clean storage."""
        raise NotImplementedError

    def bind_camera_inputs(
        self,
        *,
        latents: Mapping[str, tuple[LatentInput, ...]],
        sizes: tuple[VideoSizeT, ...],
        step: torch.Tensor,
        text_features: tuple[torch.Tensor, ...],
        state: Mapping[str, torch.Tensor],
        noise: tuple[torch.Tensor, ...],
    ) -> InputT:
        """Bind camera state and one selected noise draw per latent sample.

        Noise already has the sample's native shape, precision and device.
        A final step that does not re-noise may reuse any finite draw without
        consuming another random value. Every supplied tensor stays borrowed.
        """
        raise NotImplementedError
