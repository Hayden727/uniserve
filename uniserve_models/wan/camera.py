"""Wan camera mathematics, with explicit pose and token ordering.

Raw orbit preparation uses CPU NumPy to preserve the dataset's FP32 inverse
and relative-pose rounding. Ray construction uses caller-placed Torch tensors
in FP32 and casts only its final output. These functions perform no loading,
sampling, caching, device selection, or request management.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class CameraPoses:
    """Borrowed OpenCV camera-to-world poses and pixel intrinsics.

    Shapes are ``(B, N, 4, 4)`` and ``(B, N, 3, 3)``. N follows the latent
    slots, with no implied temporal reordering. The caller owns placement and
    must not mutate the tensors during their use. Relative/metric scaling is
    already resolved; ray construction never renormalizes these poses.
    """

    c2w: torch.Tensor
    intrinsics: torch.Tensor

    def __post_init__(self):
        if (
            self.c2w.ndim != 4
            or self.c2w.shape[-2:] != (4, 4)
            or min(self.c2w.shape[:2]) < 1
            or self.intrinsics.shape != (*self.c2w.shape[:2], 3, 3)
        ):
            raise ValueError(
                "camera poses and intrinsics require B,N,4,4 / B,N,3,3"
            )
        if (
            not self.c2w.is_floating_point()
            or not self.intrinsics.is_floating_point()
        ):
            raise ValueError(
                "camera poses and intrinsics must be floating point"
            )
        if self.c2w.device != self.intrinsics.device:
            raise ValueError("camera poses and intrinsics must share a device")


@dataclass(frozen=True, slots=True)
class PluckerInput:
    """Moment then direction ``(B,6,N,H,W)`` and optional scale ``(B,2K)``."""

    rays: torch.Tensor
    scale_embedding: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class UCPEInput:
    """World-to-ray transforms ``(B,N*H*W,4,4)`` in slot/row/column order."""

    viewmats: torch.Tensor


def segment_latent_indices(frame_count: int) -> np.ndarray:
    """Select bundle-leading cameras for independent 13-frame VAE clips.

    Every clip contributes four latents represented by frames 0, 1, 5, 9.
    The leading frame of each four-frame bundle is used, never its midpoint.
    """
    if type(frame_count) is not int or frame_count <= 0 or frame_count % 13:
        raise ValueError("camera grouping requires complete 13-frame clips")
    starts = np.arange(0, frame_count, 13, dtype=np.int64)
    return (starts[:, None] + np.asarray([0, 1, 5, 9])).reshape(-1)


def ring_frame_indices(frame_count: int) -> np.ndarray:
    """Select twelve contiguous 13-frame clips from a complete raw orbit.

    Extra frames are dropped evenly at clip boundaries. A 160-frame ring
    drops 39, 79, 119, 159. This index array also orders the RGB encoder input.
    """
    if type(frame_count) is not int or frame_count < 156:
        raise ValueError("an orbit requires at least 156 frames")
    extra = frame_count - 156
    if extra and 12 % extra:
        raise ValueError("surplus orbit frames must divide twelve segments")
    interval = 12 // extra if extra else 0
    indices = []
    cursor = 0
    for segment in range(12):
        indices.extend(range(cursor, cursor + 13))
        cursor += 13
        if interval and (segment + 1) % interval == 0:
            cursor += 1
    return np.asarray(indices, dtype=np.int64)


def ring_latent_indices(frame_count: int) -> np.ndarray:
    """Raw orbit camera indices in the 48-slot latent order."""
    return ring_frame_indices(frame_count)[segment_latent_indices(156)]


def normalize_cameras(c2w: np.ndarray) -> np.ndarray:
    """Express ordered OpenCV cameras relative to camera zero, in FP32.

    This preserves the native NumPy inverse/einsum sequence. Substituting a
    rigid inverse, Torch inverse, or FP64 matrix multiply changes rounding.
    """
    poses = np.asarray(c2w, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError("camera-to-world poses require N,4,4")
    if not np.isfinite(poses).all():
        raise ValueError("camera poses must be finite")
    reference = np.linalg.inv(poses[0]).astype(np.float32)
    return np.einsum("ij,njk->nik", reference, poses).astype(np.float32)


def prepare_ring_cameras(
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    *,
    unit_diameter: bool,
) -> CameraPoses:
    """Prepare one ordered raw orbit, returning FP32 CPU camera tensors.

    ``world_to_camera`` is ``(V,4,4)`` in OpenCV axes/metres, already sorted
    in ascending frame order. ``intrinsics`` is pixel K, either shared
    ``(3,3)`` or per-frame ``(V,3,3)``. Both inputs round to FP32 before
    arithmetic. The result is ``B=1,N=48`` in independent-clip latent order.

    Plücker sparse-orbit checkpoints require ``unit_diameter=True``; UCPE
    sparse-orbit checkpoints retain metric translations with ``False``.
    Diameter uses the median FP64 distance to the FP64 centroid of the FULL
    relative orbit. Absolute scene scale is a separate conditioning input.
    """
    raw = np.asarray(world_to_camera, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (4, 4):
        raise ValueError("world-to-camera poses require V,4,4")
    if not np.isfinite(raw).all():
        raise ValueError("camera poses must be finite")
    indices = ring_latent_indices(len(raw))
    pixel_k = np.asarray(intrinsics, dtype=np.float32)
    if pixel_k.shape == (3, 3):
        pixel_k = np.broadcast_to(pixel_k, (len(raw), 3, 3))
    if pixel_k.shape != (len(raw), 3, 3) or not np.isfinite(pixel_k).all():
        raise ValueError(
            "intrinsics require finite 3,3 or V,3,3 pixel matrices"
        )
    if np.any(pixel_k[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("camera focal lengths must be positive")

    selected = np.linalg.inv(raw[indices]).astype(np.float32)
    selected = normalize_cameras(selected)
    if unit_diameter:
        full = normalize_cameras(np.linalg.inv(raw).astype(np.float32))
        centers = np.asarray(full, dtype=np.float64)[:, :3, 3]
        radius = float(
            np.median(np.linalg.norm(centers - centers.mean(axis=0), axis=1))
        )
        diameter = 2.0 * radius
        if diameter <= 1e-3:
            raise ValueError(
                "unit-diameter conditioning requires a nondegenerate orbit"
            )
        # In-place division preserves the native FP32 destination rounding.
        selected[:, :3, 3] /= diameter
    return CameraPoses(
        torch.from_numpy(selected)[None],
        torch.from_numpy(pixel_k[indices].copy())[None],
    )


def scale_to_sinusoidal(
    scale_meters: float | torch.Tensor, *, num_bands: int = 8
) -> torch.Tensor:
    """Encode metres with log-clamped [0.5,20] sin-then-cos powers of two.

    Scalar input returns ``(2K,)``; ``(B,)`` and ``(B,1)`` return ``(B,2K)``.
    All arithmetic is FP32, including logarithm and angular multiplication.
    Batch size one keeps its batch dimension.
    """
    if type(num_bands) is not int or num_bands <= 0:
        raise ValueError("scale bands must be a positive integer")
    scalar = (
        not isinstance(scale_meters, torch.Tensor) or scale_meters.ndim == 0
    )
    if not isinstance(scale_meters, torch.Tensor):
        scale_meters = torch.tensor(float(scale_meters), dtype=torch.float32)
    if scale_meters.ndim > 2 or (
        scale_meters.ndim == 2 and scale_meters.shape[1] != 1
    ):
        raise ValueError("scene scales require a scalar, B, or B,1")
    values = scale_meters.to(torch.float32).reshape(-1)
    lo, hi = math.log(0.5), math.log(20.0)
    unit = (torch.log(values.clamp(min=0.5, max=20.0)) - lo) / max(
        hi - lo, 1e-6
    )
    bands = torch.arange(num_bands, dtype=torch.float32, device=unit.device)
    angles = unit.unsqueeze(-1) * (2.0**bands) * math.pi
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return embedding.squeeze(0) if scalar else embedding


def _check_grid(
    poses: CameraPoses,
    height: int,
    width: int,
    patch_size: int,
    token_grid: tuple[int, int, int],
):
    if any(type(n) is not int or n <= 0 for n in (height, width, patch_size)):
        raise ValueError(
            "camera image and patch dimensions must be positive integers"
        )
    if height % patch_size or width % patch_size:
        raise ValueError(
            "camera image dimensions must be divisible by patch size"
        )
    expected = (poses.c2w.shape[1], height // patch_size, width // patch_size)
    if token_grid != expected:
        raise ValueError(
            f"camera grid {expected} does not match token grid {token_grid}"
        )


def plucker_input(
    poses: CameraPoses,
    *,
    height: int,
    width: int,
    token_grid: tuple[int, int, int],
    dtype: torch.dtype,
    scale_meters: torch.Tensor | None,
    scale_bands: int = 8,
    patch_size: int = 16,
) -> PluckerInput:
    """Build native Plücker rays and optional absolute-scale conditioning.

    These trained weights require corner pixels, ``R.T @ direction``, and
    additive norm epsilon 1e-8. Channels are moment ``origin x direction``
    then direction. Flattening ``B,6,N,H,W`` as ``B,N*H*W,6`` aligns with DiT
    tokens; height/width are checked separately to catch transposed grids.
    Pass ``scale_meters=None`` only for a model without a scale branch.
    """
    _check_grid(poses, height, width, patch_size, token_grid)
    c2w = poses.c2w.to(torch.float32)
    intrinsics = poses.intrinsics.to(torch.float32).clone()
    intrinsics[..., :2, :] = intrinsics[..., :2, :] / patch_size
    batch, slots = c2w.shape[:2]
    grid_h, grid_w = token_grid[1:]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(
            0, grid_h - 1, grid_h, device=c2w.device, dtype=c2w.dtype
        ),
        torch.linspace(
            0, grid_w - 1, grid_w, device=c2w.device, dtype=c2w.dtype
        ),
        indexing="ij",
    )
    x = (grid_x.flatten() - intrinsics[..., 0, 2, None]) / intrinsics[
        ..., 0, 0, None
    ]
    y = (grid_y.flatten() - intrinsics[..., 1, 2, None]) / intrinsics[
        ..., 1, 1, None
    ]
    directions = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    directions = torch.einsum("bnpi,bnij->bnpj", directions, c2w[..., :3, :3])
    origins = c2w[..., :3, 3].unsqueeze(-2).expand_as(directions)
    directions = directions / (
        torch.norm(directions, dim=-1, keepdim=True) + 1e-8
    )
    origins = origins.reshape(batch, slots, grid_h, grid_w, 3).permute(
        0, 4, 1, 2, 3
    )
    directions = directions.reshape(batch, slots, grid_h, grid_w, 3).permute(
        0, 4, 1, 2, 3
    )
    rays = torch.cat(
        [torch.cross(origins, directions, dim=1), directions], dim=1
    )
    scale = None
    if scale_meters is not None:
        if scale_meters.shape not in ((batch,), (batch, 1)):
            raise ValueError("scene scale batch must match the camera batch")
        if scale_meters.device != c2w.device:
            raise ValueError(
                "scene scales and camera poses must share a device"
            )
        scale = scale_to_sinusoidal(scale_meters, num_bands=scale_bands).to(
            dtype
        )
    return PluckerInput(rays.to(dtype), scale)


def ucpe_input(
    poses: CameraPoses,
    *,
    height: int,
    width: int,
    token_grid: tuple[int, int, int],
    dtype: torch.dtype,
    patch_size: int = 16,
) -> UCPEInput:
    """Construct UCPE world-to-ray transforms at pixel patch centers.

    Uses true camera-to-world rotation and a ray frame defined by the camera
    y-axis. Normalization epsilons match the native helper: 1e-12 for camera
    directions, 1e-6 for the three world-space axes. Each batch element uses
    its own intrinsics; the native serving invocation has batch size one.
    Run before graph capture: native focal/principal values are Python
    scalars, and extracting them synchronizes device tensors.
    """
    _check_grid(poses, height, width, patch_size, token_grid)
    c2w = poses.c2w.to(torch.float32)
    intrinsics = poses.intrinsics.to(torch.float32)
    grid_h, grid_w = token_grid[1:]
    px = (
        torch.arange(grid_w, device=c2w.device, dtype=c2w.dtype) + 0.5
    ) * patch_size
    py = (
        torch.arange(grid_h, device=c2w.device, dtype=c2w.dtype) + 0.5
    ) * patch_size
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    batches = []
    for batch in range(c2w.shape[0]):
        views = []
        for view in range(c2w.shape[1]):
            pixel_k = intrinsics[batch, view]
            # Keep scalar arithmetic: CUDA may use a different division
            # kernel for a device tensor divisor than for a Python scalar.
            focal_x, focal_y = pixel_k[0, 0].item(), pixel_k[1, 1].item()
            center_x, center_y = pixel_k[0, 2].item(), pixel_k[1, 2].item()
            x = (grid_x - center_x) / focal_x
            y = (grid_y - center_y) / focal_y
            directions = F.normalize(
                torch.stack([x, y, torch.ones_like(x)], dim=-1), dim=-1
            )
            camera = c2w[batch : batch + 1, view : view + 1]
            rotation = camera[..., :3, :3]
            center = camera[..., :3, 3]
            directions = directions[None, None].expand(1, 1, grid_h, grid_w, 3)
            world = torch.einsum(
                "b t i j, b t h w j -> b t h w i", rotation, directions
            )
            camera_y = rotation[..., :, 1][:, :, None, None].expand_as(world)
            z_axis = F.normalize(world, dim=-1, eps=1e-6)
            x_axis = F.normalize(
                torch.cross(camera_y, z_axis, dim=-1), dim=-1, eps=1e-6
            )
            y_axis = F.normalize(
                torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-6
            )
            world_to_ray = torch.stack(
                [x_axis, y_axis, z_axis], dim=-1
            ).transpose(-1, -2)
            center = center[:, :, None, None].expand_as(world)
            translation = -torch.einsum(
                "b t h w i j, b t h w j -> b t h w i", world_to_ray, center
            )
            matrices = torch.zeros(
                1, 1, grid_h, grid_w, 4, 4, device=c2w.device, dtype=c2w.dtype
            )
            matrices[..., :3, :3] = world_to_ray
            matrices[..., :3, 3] = translation
            matrices[..., 3, 3] = 1.0
            views.append(matrices.reshape(1, grid_h * grid_w, 4, 4))
        batches.append(torch.cat(views, dim=1))
    return UCPEInput(torch.cat(batches, dim=0).to(dtype))
