"""Wan's camera-conditioned numerical capability for shared video execution."""

from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from uniserve.media import image
from uniserve.model.camera import CameraCondition, CameraDenoiser
from uniserve.model.denoiser import Condition
from uniserve.tensors import BufferConfig, OutputLayout

from .camera import (
    CameraPoses,
    plucker_input,
    prepare_ring_cameras,
    ring_frame_indices,
    ucpe_input,
)
from .denoiser import Denoiser
from .inputs import LatentFormat


@dataclass(frozen=True, slots=True)
class VideoSize:
    """Output timeline and exact text length for independent Wan segments."""

    num_frames: int
    canvas: image.Config
    num_text_tokens: int
    conditions: tuple[Condition, ...] = ()
    vision_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if (
            type(self.num_frames) is not int
            or self.num_frames < 1
            or self.num_frames % 13
        ):
            raise ValueError("Wan output requires complete 13-frame segments")
        if (
            type(self.num_text_tokens) is not int
            or not 1 <= self.num_text_tokens <= 512
        ):
            raise ValueError("Wan text conditioning requires 1..512 tokens")
        if self.canvas.height % 8 or self.canvas.width % 8:
            raise ValueError("Wan output dimensions must divide by eight")
        if self.conditions or self.vision_spans:
            raise ValueError(
                "Wan conditions are explicit camera fields and clean latents"
            )

    @property
    def condition_rows(self):
        return 0

    @property
    def frames(self):
        return self.num_frames // 13 * 4

    @property
    def height(self):
        return self.canvas.height // 8

    @property
    def width(self):
        return self.canvas.width // 8

    @property
    def text_tokens(self):
        return self.num_text_tokens


class ServingDenoiser(Denoiser, CameraDenoiser):
    """Expose the Wan numerical denoiser through the shared video contract.

    The inherited explicit ``bind_inputs`` remains available for direct
    diagnostics. Shared execution uses ``bind_camera_inputs`` with tensors
    whose storage it owns. No request values or random generators are saved.
    """

    @property
    def canvases(self):
        return MappingProxyType({(480, "26:15"): image.Config(480, 832)})

    @property
    def tasks(self):
        return ("camera2v",)

    @property
    def text_only(self):
        return False

    @property
    def latent_format(self):
        if self.config.ucpe is None:
            return None
        if self.config.ucpe.latent_format is not None:
            return self.config.ucpe.latent_format
        # Native ring VAE output is channels-last. The rectangular dataset
        # assembles its cells with stack(...).contiguous(). Existing exports
        # without this field resolve from their explicit numerical grid.
        return (
            LatentFormat.CONTIGUOUS
            if self.config.grid is not None
            else LatentFormat.CHANNELS_LAST_3D
        )

    @property
    def text_condition_width(self):
        return self.config.transformer.text_dim

    def text_condition_rows(self, layout):
        """Reserve Wan's post-encoding context, including zero feature rows.

        UMT5 evaluates the exact prompt length. Native Wan then pads its
        features to 512 before the DiT's biased projection and unmasked
        cross-attention, so omitting those rows changes the prediction.
        """
        return 512

    def layout_size(self, size):
        """All prompt lengths share the network's 512-row padded context.

        UMT5 still encodes exactly the requested token count. Camera values,
        clean cells and text features are live borrowed state and do not
        change the DiT shapes or its reusable numerical layout.
        """
        return replace(size, num_text_tokens=512)

    def legal_frame_count(self, requested):
        if type(requested) is not int or requested < 1:
            raise ValueError("requested frame count must be a positive integer")
        return (requested + 12) // 13 * 13

    def make_size(
        self,
        num_frames,
        num_text_tokens,
        *,
        canvas,
        conditions=(),
        vision_spans=(),
    ):
        if canvas not in self.canvases.values():
            raise ValueError("Wan camera serving uses a 480 by 832 canvas")
        size = VideoSize(
            num_frames, canvas, num_text_tokens, conditions, vision_spans
        )
        self.latent_shape("video", size)
        return size

    def camera_buffers(self, size):
        shape = self.latent_shape("video", size)
        pt, ph, pw = self.config.transformer.patch_size
        fields = {
            "clean": BufferConfig(shape, self.prediction_dtype),
            "clean_mask": BufferConfig((1, 1, size.frames, 1, 1), torch.bool),
        }
        if self.config.ucpe is None:
            fields["camera_rays"] = BufferConfig(
                (1, 6, size.frames // pt, size.height // ph, size.width // pw),
                self.prediction_dtype,
            )
            if self.config.transformer.use_scale_embedding:
                fields["scale_embedding"] = BufferConfig(
                    (1, 2 * self.config.transformer.scale_bands),
                    self.prediction_dtype,
                )
        else:
            tokens = (
                (size.frames // pt) * (size.height // ph) * (size.width // pw)
            )
            fields["viewmats"] = BufferConfig(
                (1, tokens, 4, 4), self.prediction_dtype
            )
            # Cell timestamps are request data. Their temporal phases belong
            # to request state, not the constants cached for a video size.
            for name in ("camera_cos", "camera_sin"):
                fields[name] = BufferConfig(
                    (tokens, self.config.transformer.head_dim), torch.float32
                )
        return MappingProxyType(fields)

    def state_buffers(self, size):
        return {
            "video": BufferConfig(
                self.latent_shape("video", size), self.prediction_dtype
            ),
            **self.camera_buffers(size),
        }

    def output_layout(self, size):
        shape = self.latent_shape("video", size)
        return {
            "video": OutputLayout(
                shape,
                self.prediction_dtype,
                tuple(slice(0, extent) for extent in shape),
            )
        }

    def sampling_noise_shapes(self, size):
        return (self.noise_shape("video", size),) * (self.num_steps - 1)

    def _check_camera_outputs(self, size, out):
        fields = self.camera_buffers(size)
        if set(out) != set(fields):
            raise ValueError(
                "camera outputs must cover the declared camera buffers"
            )
        if len({tensor.device for tensor in out.values()}) != 1:
            raise ValueError("camera outputs must share one device")
        for name, field in fields.items():
            tensor = out[name]
            if tensor.shape != field.shape or tensor.dtype != field.dtype:
                raise ValueError(
                    f"camera buffer {name!r} must match its shape and dtype"
                )

    def prepare_state(self, sizes, *, layouts, out):
        if (
            len(sizes) != 1
            or len(layouts) != 1
            or not self.holds(layouts[0], sizes[0])
        ):
            raise ValueError(
                "Wan request state requires one exact numerical layout"
            )
        self._check_camera_outputs(layouts[0], out)
        if any(tensor.device.type != "cpu" for tensor in out.values()):
            raise ValueError("initial request state must use CPU source views")
        # Startup and graph preparation use valid zero-valued borrowed fields.
        # Request admission subsequently fills camera inputs and clean slots.
        for tensor in out.values():
            tensor.zero_()

    def output_camera(self, size, condition: CameraCondition):
        if any(cell >= size.frames for cell in condition.clean_cells):
            raise ValueError(
                "clean camera cells must lie within the latent grid"
            )
        if condition.cell_time_index is not None:
            if condition.cell_time_index.shape != (size.frames,):
                raise ValueError("cell timestamps must match the latent grid")
            if self.config.ucpe is None and not torch.equal(
                condition.cell_time_index, torch.arange(size.frames)
            ):
                raise ValueError(
                    "this Plucker path requires sequential cell timestamps"
                )
        if condition.pose_space == "raw_world_to_camera":
            poses = prepare_ring_cameras(
                condition.poses.detach().float().numpy(),
                condition.intrinsics.detach().float().numpy(),
                unit_diameter=self.config.ucpe is None,
            )
        else:
            intrinsics = condition.intrinsics
            if intrinsics.ndim == 2:
                intrinsics = intrinsics.expand(condition.poses.shape[0], -1, -1)
            poses = CameraPoses(
                condition.poses.unsqueeze(0), intrinsics.unsqueeze(0)
            )
        if poses.c2w.shape[1] != size.frames:
            raise ValueError(
                "prepared cameras must match the latent frame count"
            )
        return replace(
            condition,
            poses=poses.c2w[0],
            intrinsics=poses.intrinsics[0],
            pose_space="prepared_camera_to_world",
        )

    def prepare_camera(self, size, condition: CameraCondition, *, out):
        self._check_camera_outputs(size, out)
        pt, ph, pw = self.config.transformer.patch_size
        if pt != 1 or ph != pw:
            raise ValueError(
                "Wan camera preparation requires unit temporal "
                "and square spatial patches"
            )
        canonical = self.output_camera(size, condition)
        poses = CameraPoses(
            canonical.poses.unsqueeze(0), canonical.intrinsics.unsqueeze(0)
        )
        has_scale = self.config.transformer.use_scale_embedding
        if has_scale and condition.scene_scale is None:
            raise ValueError(
                "this Plucker checkpoint requires an explicit scene scale"
            )
        # Native camera preparation rounds poses to FP32 on the compute
        # device before ray construction. CPU and CUDA ray arithmetic need
        # not round identically, so use the caller's output placement.
        device = out["clean"].device
        poses = CameraPoses(
            poses.c2w.to(device=device, dtype=torch.float32),
            poses.intrinsics.to(device=device, dtype=torch.float32),
        )
        scene_scale = (
            condition.scene_scale.to(device=device, dtype=torch.float32)
            if has_scale
            else None
        )
        camera_args = {
            "height": size.height * 8,
            "width": size.width * 8,
            "token_grid": (size.frames, size.height // ph, size.width // pw),
            "dtype": self.prediction_dtype,
            "patch_size": ph * 8,
        }
        if self.config.ucpe is None:
            camera = plucker_input(
                poses,
                **camera_args,
                scale_meters=scene_scale,
                scale_bands=self.config.transformer.scale_bands,
            )
            out["camera_rays"].copy_(camera.rays)
            if has_scale:
                out["scale_embedding"].copy_(camera.scale_embedding)
        else:
            camera = ucpe_input(poses, **camera_args)
            out["viewmats"].copy_(camera.viewmats)
            cos, sin = self.network.rotary_embedding(
                (size.frames, size.height, size.width),
                device=device,
                cell_time_index=canonical.cell_time_index,
            )
            out["camera_cos"].copy_(cos)
            out["camera_sin"].copy_(sin)
        out["clean"].zero_()
        if condition.clean_latents is not None:
            if condition.clean_latents.shape != out["clean"].shape[1:]:
                raise ValueError(
                    "clean latents must match the native C,F,H,W shape"
                )
            out["clean"].copy_(condition.clean_latents.unsqueeze(0))
        out["clean_mask"].zero_()
        out["clean_mask"][:, :, list(condition.clean_cells)] = True

    def input_frame_indices(self, size, condition, frame_count):
        if condition.pose_space == "raw_world_to_camera":
            if (
                frame_count != condition.poses.shape[0]
                or size.num_frames != 156
            ):
                raise ValueError(
                    "raw orbit RGB and poses must produce "
                    "twelve 13-frame segments"
                )
            return tuple(
                int(index) for index in ring_frame_indices(frame_count)
            )
        if frame_count != size.num_frames:
            raise ValueError(
                "prepared camera RGB must already follow the output timeline"
            )
        return tuple(range(frame_count))

    def store_clean(self, size, encoded_rows, *, out):
        shape = self.latent_shape("video", size)
        channels = shape[1]
        if encoded_rows.shape != (
            size.frames * size.height * size.width,
            channels,
        ):
            raise ValueError(
                "encoded clean rows must match the latent frame-major raster"
            )
        if out.shape != shape or out.dtype != self.prediction_dtype:
            raise ValueError(
                "clean destination must match the native latent shape and dtype"
            )
        clean = encoded_rows.reshape(
            size.frames, size.height, size.width, channels
        )
        out.copy_(clean.permute(3, 0, 1, 2).unsqueeze(0))

    def bind_camera_inputs(
        self, *, latents, sizes, step, text_features, state, noise
    ):
        if set(latents) != {"video"} or any(
            len(values) != 1
            for values in (latents["video"], sizes, text_features, noise)
        ):
            raise ValueError(
                "Wan camera denoising evaluates one conditioned grid per call"
            )
        size, latent = sizes[0], latents["video"][0]
        if text_features[0].shape != (
            self.text_condition_rows(size),
            self.text_condition_width,
        ):
            raise ValueError(
                "Wan text features require the complete 512-row context"
            )
        arguments = {
            "sample": latent.tensor,
            "timestep": latent.timestep,
            "step": step,
            "size": size,
            "context": text_features[0].unsqueeze(0),
            "noise": noise[0],
            "clean": state["clean"],
            "clean_mask": state["clean_mask"],
        }
        if self.config.ucpe is None:
            return self.bind_inputs(
                **arguments,
                camera_rays=state["camera_rays"],
                scale_embedding=state.get("scale_embedding"),
            )
        return self.bind_ucpe_inputs(
            **arguments,
            viewmats=state["viewmats"],
            rotary=(state["camera_cos"], state["camera_sin"]),
        )
