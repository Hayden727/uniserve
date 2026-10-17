"""Strict Wan student assignments using the public UniServe weight loader."""

from functools import partial
from pathlib import Path

import torch
from torch import nn

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.nn.linear import Linear
from uniserve.nn.nvfp4 import NVFP4Linear
from uniserve.nn.nvfp4_qat import NVFP4QATLinear
from uniserve.quantization.nvfp4 import NVFP4Epilogue

from .config import TransformerConfig
from .text import UMT5, TextConfig
from .transformer import PluckerTransformer
from .ucpe import CameraConfig, UcpeTransformer
from .vae import DecoderNetwork, EncoderNetwork, ResidualBlock, VAEConfig


def encoder_parameter_sources(model: EncoderNetwork):
    """Map shared residual blocks to the original Wan encoder checkpoint."""
    names = {name: name for name, _ in model.named_parameters()}
    fields = {
        "norm1": "residual.0",
        "conv1": "residual.2",
        "norm2": "residual.3",
        "conv2": "residual.6",
        "conv_shortcut": "shortcut",
    }
    for path, module in model.named_modules():
        if isinstance(module, ResidualBlock):
            for name, _ in module.named_parameters():
                field, suffix = name.split(".", 1)
                names[f"{path}.{name}"] = f"{path}.{fields[field]}.{suffix}"
    return names


def encoder_mapping(model: EncoderNetwork):
    parameters = dict(model.named_parameters())
    names = encoder_parameter_sources(model)

    def assignments(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(names[name]))
            for name, parameter in parameters.items()
            if names[name] in available
        )

    return (
        weights.ModuleMapping(
            model,
            "vae_encoder",
            assignments,
            frozenset(parameters),
            nonresident_prefixes=("decoder.", "conv2."),
        ),
    )


def load_encoder(
    path: str | Path, config: VAEConfig, *, device: str | torch.device
):
    """Materialize the BF16 encoder; decoder weights stay on CPU."""
    root = Path(path)
    declaration = (
        checkpoint.Config(name="vae_encoder", filenames=(root.name,))
        if root.is_file()
        else checkpoint.Config(name="vae_encoder")
    )
    source = declaration.resolve(
        root.parent if root.is_file() else root, io=loading.Config()
    )
    return loading.load_model(
        EncoderNetwork,
        config,
        checkpoint=(source,),
        mapping=encoder_mapping,
        device=device,
        weights=weights.Config(dtype=torch.bfloat16),
    )


def decoder_mapping(model: DecoderNetwork):
    parameters = dict(model.named_parameters())

    def assignments(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(name))
            for name, parameter in parameters.items()
            if name in available
        )

    return (
        weights.ModuleMapping(
            model,
            "vae",
            assignments,
            frozenset(parameters),
            nonresident_prefixes=("encoder.", "quant_conv."),
        ),
    )


def load_decoder(
    path: str | Path, config: VAEConfig, *, device: str | torch.device
):
    """Load every decoder parameter; the separate encoder is nonresident."""
    source = checkpoint.Config(name="vae").resolve(
        Path(path), io=loading.Config()
    )
    return loading.load_model(
        DecoderNetwork,
        config,
        checkpoint=(source,),
        mapping=decoder_mapping,
        device=device,
        weights=weights.Config(dtype=torch.float32),
    )


def text_mapping(model: UMT5):
    """Load all UMT5 weights, fusing Q/K/V without changing their values."""
    parameters = dict(model.named_parameters(remove_duplicate=False))

    def assignments(reader):
        available = frozenset(reader.names())
        result = []
        for name, parameter in parameters.items():
            source = (
                "shared.weight"
                if name == "encoder.embed_tokens.weight"
                else name
            )
            if name.endswith("qkv_proj.weight"):
                width = parameter.shape[0] // 3
                for index, branch in enumerate(("q", "k", "v")):
                    source = name.replace("qkv_proj", branch)
                    if source in available:
                        result.append(
                            weights.Assignment(
                                parameter,
                                reader.get(source),
                                target_slice=(
                                    slice(index * width, (index + 1) * width),
                                    slice(0, parameter.shape[1]),
                                ),
                            )
                        )
            elif source in available:
                result.append(weights.Assignment(parameter, reader.get(source)))
        return tuple(result)

    return (
        weights.ModuleMapping(
            model, "text_encoder", assignments, frozenset(parameters)
        ),
    )


def load_text_encoder(
    path: str | Path, config: TextConfig, *, device: str | torch.device
) -> loading.Result[UMT5]:
    """Materialize the full FP32 text encoder with strict source coverage."""
    source = checkpoint.Config(name="text_encoder").resolve(
        Path(path), io=loading.Config()
    )
    return loading.load_model(
        UMT5,
        config,
        checkpoint=(source,),
        mapping=text_mapping,
        device=device,
        weights=weights.Config(dtype=torch.float32),
    )


def transformer_mapping(model: PluckerTransformer | UcpeTransformer):
    """Map the complete exported student namespace without ignored tensors."""
    parameters = dict(model.named_parameters())

    def assignments(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(name))
            for name, parameter in parameters.items()
            if name in available
        )

    return (
        weights.ModuleMapping(
            model,
            "transformer",
            assignments,
            frozenset(parameters),
            post_load=(lambda reader: quantize_transformer(model))
            if model.config.nvfp4 is not None or model.config.qat is not None
            else None,
        ),
    )


def qat_projection_names(
    model: PluckerTransformer | UcpeTransformer,
) -> tuple[str, ...]:
    """Return the native QAT projection set in a completely built transformer.

    Native LinearBase eligibility is mapped to these mathematical call sites,
    without including ordinary camera Linear, gate or conditioning layers.
    Checkpoint activation is resolved before any CUDA encoding is attempted.
    """
    recipe = model.config.qat
    if recipe is None or not recipe.active:
        return ()
    projections = {
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out",
        "ffn.fc_in",
        "ffn.fc_out",
    }
    names = []
    for index in range(
        recipe.bf16_edge_blocks,
        model.config.num_layers - recipe.bf16_edge_blocks,
    ):
        for projection in sorted(projections):
            name = f"blocks.{index}.{projection}"
            layer = model.get_submodule(name)
            if type(layer) is not Linear:
                raise ValueError(
                    f"QAT requires an unencoded local projection: {name}"
                )
            names.append(name)
    return tuple(names)


def quantize_transformer(
    model: PluckerTransformer | UcpeTransformer,
) -> tuple[str, ...]:
    """Encode the declared native FP4 projections after complete placement.

    Packed serving FP4 includes gates, conditioning and output heads, while
    retaining Plucker camera/scale projections in BF16. Training QAT instead
    selects middle-block main/text attention and FFN projections only. Packed
    UCPE residual alphas are resolved before capture; QAT keeps native alphas
    inside its GEMM. No checkpoint reader or runtime owner is retained.
    """
    if model.config.qat is not None:
        names = qat_projection_names(model)
        for name in names:
            parent, _, field = name.rpartition(".")
            layer = model.get_submodule(name)
            setattr(
                model.get_submodule(parent),
                field,
                NVFP4QATLinear.from_linear(layer),
            )
        return names
    recipe = model.config.nvfp4
    if recipe is None:
        raise ValueError(
            "Wan FP4 conversion requires an explicit numerical recipe"
        )
    if any(isinstance(layer, NVFP4Linear) for layer in model.modules()):
        raise ValueError("Wan projections have already been FP4 encoded")
    plucker = isinstance(model, PluckerTransformer)
    targets = []
    for name, layer in model.named_modules():
        if type(layer) not in (Linear, nn.Linear):
            continue
        if plucker and any(
            field in name.split(".") for field in ("cam_encoder", "scale_mlp")
        ):
            continue
        if layer.weight.shape[1] % 32:
            continue
        if (
            layer.weight.is_meta
            or not layer.weight.is_cuda
            or layer.weight.dtype != torch.bfloat16
        ):
            raise ValueError(
                "Wan FP4 conversion requires placed CUDA BF16 weights"
            )
        if layer.bias is not None and (
            layer.bias.is_meta
            or layer.bias.device != layer.weight.device
            or layer.bias.dtype != torch.bfloat16
        ):
            raise ValueError("Wan FP4 conversion requires a placed BF16 bias")
        parent, _, field = name.rpartition(".")
        owner = model.get_submodule(parent)
        # These are the native RMSNorm-fed Q/K projections in main and
        # text cross-attention. Compressed PRoPE camera projections have no
        # Q/K norm and therefore keep their alpha.
        absorb = recipe.absorb_qk_alpha and field in ("to_q", "to_k")
        fold = (
            not plucker
            and recipe.fold_gate_alpha
            and (
                name.endswith(".ffn.fc_out")
                or (
                    field == "to_out"
                    and parent.startswith("blocks.")
                    and parent.count(".") == 1
                )
            )
        )
        targets.append(
            (
                name,
                owner,
                field,
                layer,
                fold,
                NVFP4Epilogue(
                    fuse_bias=recipe.fuse_bias,
                    apply_alpha=not (absorb or fold),
                ),
            )
        )
    if not targets:
        raise ValueError(
            "Wan FP4 architecture has no eligible local projections"
        )
    for name, owner, field, layer, fold, epilogue in targets:
        encoded = NVFP4Linear.from_linear(layer, epilogue=epilogue)
        setattr(owner, field, encoded)
        if fold:
            block_path = (
                name.removesuffix(".ffn.fc_out")
                if name.endswith(".ffn.fc_out")
                else name.removesuffix(".to_out")
            )
            block = model.get_submodule(block_path)
            setattr(
                block,
                "ffn_alpha" if field == "fc_out" else "output_alpha",
                float(encoded.alpha),
            )
    return tuple(name for name, *_ in targets)


def load_transformer(
    path: str | Path,
    config: TransformerConfig,
    *,
    device: str | torch.device,
) -> loading.Result[PluckerTransformer]:
    """Load a verified BF16 student export, rejecting missing/extra tensors.

    Parameters are constructed on meta and materialized by UniServe's shared
    loader. Source dtypes remain documented in the export manifest; compute
    weights round to BF16 as in the native Wan inference load.
    A sparse config includes every trained main compression gate, so a
    Quality export without those tensors cannot stand in for Fast weights.
    """
    root = Path(path)
    if (root / ".incomplete").exists():
        raise ValueError("Wan student export is incomplete")
    source = checkpoint.Config(name="transformer").resolve(
        root, io=loading.Config()
    )
    return loading.load_model(
        PluckerTransformer,
        config,
        checkpoint=(source,),
        mapping=transformer_mapping,
        device=device,
        weights=weights.Config(dtype=torch.bfloat16),
    )


def load_ucpe_transformer(
    path: str | Path,
    config: TransformerConfig,
    camera: CameraConfig,
    *,
    main_compress_gate: bool = False,
    device: str | torch.device,
) -> loading.Result[UcpeTransformer]:
    """Load every UCPE student tensor through the strict public loader.

    Camera Q/K/V/output and optional main/camera compression gates retain
    their native names. Gate presence describes checkpoint architecture;
    dense Quality does not evaluate the sparse-attention gate projections;
    a VSA config activates the same strictly loaded learned gates.
    No source tensors, including Plucker-only tensors, are ignored.
    """
    root = Path(path)
    if (root / ".incomplete").exists():
        raise ValueError("Wan student export is incomplete")
    source = checkpoint.Config(name="transformer").resolve(
        root, io=loading.Config()
    )
    return loading.load_model(
        partial(
            UcpeTransformer,
            camera=camera,
            main_compress_gate=main_compress_gate,
        ),
        config,
        checkpoint=(source,),
        mapping=transformer_mapping,
        device=device,
        weights=weights.Config(dtype=torch.bfloat16),
    )


checkpoint_sources = (
    checkpoint.Config(name="transformer"),
    checkpoint.Config(name="text_encoder", directory="text_encoder"),
    checkpoint.Config(name="vae_encoder", directory="vae_encoder"),
    checkpoint.Config(name="vae", directory="vae"),
)


def checkpoint_mappings(model):
    result = transformer_mapping(model.transformer.network)
    if model.config.text_encoder is not None:
        result += text_mapping(model.text_encoder.network)
    if model.config.video_encoder is not None:
        result += encoder_mapping(model.video_encoder.encoder.encoder)
    if model.config.video_decoder is not None:
        result += decoder_mapping(model.video_decoder.decoder.decoder)
    return result


def precisions(config):
    return {"quality": checkpoint_precision(config)}


def checkpoint_precision(config):
    dtypes = {}
    if config.text_encoder is not None:
        dtypes["text_encoder"] = torch.float32
    if config.video_decoder is not None:
        dtypes["video_decoder"] = torch.float32
    return weights.Config(
        dtype=torch.bfloat16,
        dtypes=dtypes,
    )
