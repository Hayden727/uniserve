"""Immutable architectural and numerical configuration of Wan transformers."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from uniserve.model.camera import CameraGrid
from uniserve.nn.attention.vsa import Recipe
from uniserve.nn.convolution import Conv3dBackend

from .text import TextConfig
from .vae import VAEConfig

if TYPE_CHECKING:
    from .ucpe import UcpeConfig


@dataclass(frozen=True, slots=True)
class VsaConfig:
    """Trained video tiles and the fraction of their attention edges omitted."""

    tile_shape: tuple[int, int, int]
    sparsity: float = 0.9
    recipe: Recipe = Recipe.FA4_128X64

    def __post_init__(self):
        if not isinstance(self.recipe, Recipe):
            raise TypeError("VSA requires a typed attention Recipe")
        if (
            not isinstance(self.tile_shape, tuple)
            or len(self.tile_shape) != 3
            or any(
                type(value) is not int or value < 1 for value in self.tile_shape
            )
        ):
            raise ValueError("VSA tiles require positive integer triples")
        self.recipe.validate_tile(math.prod(self.tile_shape))
        if not math.isfinite(self.sparsity) or not 0 <= self.sparsity < 1:
            raise ValueError("VSA sparsity must be finite and in [0, 1)")


@dataclass(frozen=True, slots=True)
class NVFP4Config:
    """FP4 projection recipe applied after complete BF16 weight loading.

    Alpha may be omitted before Q/K RMSNorm or folded into UCPE's residual
    gates. Plucker camera projections retain their trained BF16 precision.
    These choices select no device, backend or execution resource.
    """

    fuse_bias: bool = True
    absorb_qk_alpha: bool = True
    fold_gate_alpha: bool = True

    def __post_init__(self):
        if any(
            type(value) is not bool
            for value in (
                self.fuse_bias,
                self.absorb_qk_alpha,
                self.fold_gate_alpha,
            )
        ):
            raise ValueError("Wan NVFP4 choices must be boolean")
        if (
            self.absorb_qk_alpha or self.fold_gate_alpha
        ) and not self.fuse_bias:
            raise ValueError("Wan alpha omission requires fused NVFP4 bias")


@dataclass(frozen=True, slots=True)
class QATConfig:
    """Checkpoint-resolved training precision with BF16 edge blocks.

    Eligible middle-block main/text attention and FFN projections use the
    dynamic-global-scale NVFP4 forward once the checkpoint reaches start_step.
    Camera projections, compression gates and conditioning remain BF16.
    """

    checkpoint_step: int
    start_step: int
    bf16_edge_blocks: int

    def __post_init__(self):
        if any(
            type(value) is not int or value < 0
            for value in (
                self.checkpoint_step,
                self.start_step,
                self.bf16_edge_blocks,
            )
        ):
            raise ValueError(
                "QAT steps and BF16 edge count must be nonnegative integers"
            )

    @property
    def active(self) -> bool:
        return self.checkpoint_step >= self.start_step


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    num_layers: int = 40
    num_heads: int = 40
    head_dim: int = 128
    ffn_dim: int = 13824
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    patch_size: tuple[int, int, int] = (1, 2, 2)
    patch_convolution: Conv3dBackend = "auto"
    eps: float = 1e-6
    scale_bands: int = 8
    use_scale_embedding: bool = True
    fused_modulation: bool = True
    rms_reduction_order: Literal["default", "ascending"] = "default"
    # Tier E: Inductor-fused elementwise chains (see fusion.py). Off keeps the
    # eager parity recipe.
    compile_elementwise: bool = False
    # The Plucker prefix joins camera/modulation and projection bias epilogues
    # at the original compiled model's BF16 store boundaries.
    fusion_recipe: Literal["elementwise", "plucker_prefix", "ucpe_prefix"] = (
        "elementwise"
    )
    vsa: VsaConfig | None = None
    nvfp4: NVFP4Config | None = None
    qat: QATConfig | None = None

    def __post_init__(self):
        if self.patch_convolution not in ("auto", "torch_unfold"):
            raise ValueError("unknown Wan patch convolution recipe")
        if self.qat is not None and not isinstance(self.qat, QATConfig):
            raise TypeError("Wan QAT requires a typed QATConfig")
        if self.qat is not None and self.nvfp4 is not None:
            raise ValueError(
                "training QAT and packed serving FP4 are distinct recipes"
            )
        if self.nvfp4 is not None and not isinstance(self.nvfp4, NVFP4Config):
            raise TypeError("Wan FP4 requires a typed NVFP4Config")
        if self.vsa is not None and not isinstance(self.vsa, VsaConfig):
            raise TypeError("Wan sparse attention requires a typed VsaConfig")
        if self.rms_reduction_order not in {"default", "ascending"}:
            raise ValueError("unknown Wan RMS reduction order")
        if self.compile_elementwise and self.rms_reduction_order != "default":
            raise ValueError(
                "fused elementwise chains use default RMS reduction"
            )
        if self.fusion_recipe not in {
            "elementwise",
            "plucker_prefix",
            "ucpe_prefix",
        }:
            raise ValueError("unknown Wan fusion recipe")
        if self.fusion_recipe == "ucpe_prefix" and (
            not self.compile_elementwise
            or not self.fused_modulation
            or self.use_scale_embedding
            or self.vsa is not None
            or self.nvfp4 is not None
            or self.qat is not None
        ):
            raise ValueError("UCPE prefix fusion requires compiled dense BF16")
        if self.fusion_recipe == "plucker_prefix" and (
            not self.compile_elementwise
            or not self.fused_modulation
            or not self.use_scale_embedding
            or self.vsa is not None
            or self.nvfp4 is not None
            or self.qat is not None
        ):
            raise ValueError(
                "Plucker prefix fusion requires compiled dense BF16 "
                "with its scale branch"
            )
        dimensions = (
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.ffn_dim,
            self.in_channels,
            self.out_channels,
            self.text_dim,
            self.freq_dim,
            self.scale_bands,
            *self.patch_size,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Wan dimensions must be positive integers")
        if self.head_dim % 2 or self.head_dim < 6:
            raise ValueError(
                "Wan rotary head dimension must be even and at least 6"
            )
        if len(self.patch_size) != 3:
            raise ValueError("Wan patches require time, height and width")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(
                "Wan normalization epsilon must be finite and positive"
            )

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class Config:
    """Wan component architecture; ``ucpe=None`` selects the Plucker branch.

    A typed UCPE declaration supplies its camera heads and checkpoint gate
    presence. Request cameras and execution resource policy are not config.
    """

    transformer: TransformerConfig
    text_encoder: TextConfig | None = None
    video_encoder: VAEConfig | None = None
    video_decoder: VAEConfig | None = None
    rungs: tuple[int, ...] = (1000, 750, 500, 250)
    flow_shift: float = 5.0
    train_timesteps: float = 1000.0
    ucpe: "UcpeConfig | None" = None
    grid: CameraGrid | None = None

    def __post_init__(self):
        from uniserve.diffusion import RungGrid

        if self.grid is not None:
            if not isinstance(self.grid, CameraGrid):
                raise TypeError(
                    "Wan rectangular connectivity requires CameraGrid"
                )
            if self.grid.views < 2 or self.grid.columns < 3:
                raise ValueError(
                    "Wan grid requires cameras and head/middle/tail"
                )
            if self.ucpe is None or self.transformer.patch_size[0] != 1:
                raise ValueError(
                    "Wan grid requires UCPE and unit temporal patching"
                )
            from .inputs import LatentFormat

            if self.ucpe.latent_format not in (None, LatentFormat.CONTIGUOUS):
                raise ValueError("Wan grid requires contiguous latent inputs")
            if self.ucpe.camera.vsa is not None:
                raise ValueError(
                    "Wan grid camera attention uses exact decomposition"
                )
            if self.ucpe.camera.use_2d_rope:
                raise ValueError(
                    "Wan grid camera attention requires use_2d_rope=false"
                )
            if (
                self.transformer.vsa is not None
                and self.transformer.vsa.tile_shape[0] != 1
            ):
                raise ValueError(
                    "grid VSA tiles must remain within one camera cell"
                )

        if (
            self.transformer.fusion_recipe == "ucpe_prefix"
            and self.ucpe is None
        ):
            raise ValueError("UCPE prefix fusion requires a UCPE architecture")
        if self.ucpe is not None:
            from .ucpe import UcpeConfig

            if self.transformer.fusion_recipe == "plucker_prefix":
                raise ValueError("Plucker prefix fusion does not apply to UCPE")

            if not isinstance(self.ucpe, UcpeConfig):
                raise TypeError("UCPE architecture requires a typed UcpeConfig")
            if self.transformer.fusion_recipe == "ucpe_prefix" and (
                self.grid is not None
                or self.ucpe.camera.use_2d_rope
                or self.ucpe.camera.vsa is not None
                or self.ucpe.camera.projection != "elementwise"
            ):
                raise ValueError(
                    "UCPE prefix fusion requires "
                    "dense elementwise orbit cameras"
                )
            if self.transformer.use_scale_embedding:
                raise ValueError("UCPE does not use Plucker scale embeddings")
            if (
                self.transformer.vsa is not None
                and not self.ucpe.main_compress_gate
            ):
                raise ValueError(
                    "UCPE VSA requires the trained main compression gate"
                )
        if (
            self.text_encoder is not None
            and self.text_encoder.hidden_size != self.transformer.text_dim
        ):
            raise ValueError("text encoder width must match Wan conditioning")
        for vae in (self.video_encoder, self.video_decoder):
            if vae is not None and (
                vae.z_dim != self.transformer.in_channels
                or vae.z_dim != self.transformer.out_channels
            ):
                raise ValueError("VAE channels must match Wan latent channels")
        RungGrid(
            self.rungs,
            self.flow_shift,
            self.train_timesteps,
            time_parameterization="sigma",
        )


config_sources = ()


def read_config(root: Path, io, *, sources) -> Config:
    """Read explicit Wan architecture metadata beside a student export."""
    if (root / ".incomplete").exists():
        raise ValueError("Wan student export is incomplete")
    values = json.loads((root / "config.json").read_text())
    if values.pop("architectures", None) != ["WanCameraPipeline"]:
        raise ValueError("expected the WanCameraPipeline architecture")
    transformer = dict(values.pop("transformer"))
    if "patch_size" in transformer:
        transformer["patch_size"] = tuple(transformer["patch_size"])
    if transformer.get("vsa") is not None:
        transformer["vsa"] = _read_vsa(transformer["vsa"])
    if transformer.get("nvfp4") is not None:
        transformer["nvfp4"] = NVFP4Config(**transformer["nvfp4"])
    if transformer.get("qat") is not None:
        transformer["qat"] = QATConfig(**transformer["qat"])
    if "rungs" in values:
        values["rungs"] = tuple(values["rungs"])
    if values.get("grid") is not None:
        values["grid"] = CameraGrid(**values["grid"])
    if values.get("text_encoder") is not None:
        values["text_encoder"] = TextConfig(**values["text_encoder"])
    if values.get("ucpe") is not None:
        from .inputs import LatentFormat
        from .ucpe import CameraConfig, UcpeConfig

        ucpe = dict(values["ucpe"])
        camera = dict(ucpe["camera"])
        if camera.get("vsa") is not None:
            camera["vsa"] = _read_vsa(camera["vsa"])
        ucpe["camera"] = CameraConfig(**camera)
        if ucpe.get("latent_format") is not None:
            ucpe["latent_format"] = LatentFormat(ucpe["latent_format"])
        values["ucpe"] = UcpeConfig(**ucpe)
    for name in ("video_encoder", "video_decoder"):
        if values.get(name) is not None:
            vae = dict(values[name])
            for field in (
                "dim_mult",
                "temporal_downsample",
                "latent_mean",
                "latent_std",
            ):
                if field in vae:
                    vae[field] = tuple(vae[field])
            values[name] = VAEConfig(**vae)
    return Config(TransformerConfig(**transformer), **values)


def _read_vsa(values) -> VsaConfig:
    fields = dict(values)
    fields["tile_shape"] = tuple(fields["tile_shape"])
    if "recipe" in fields:
        fields["recipe"] = Recipe(fields["recipe"])
    return VsaConfig(**fields)
