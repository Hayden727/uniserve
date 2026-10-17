"""Read immutable numerical references without importing their native stack."""

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash logical contiguous CPU bytes, including scalar and BF16 tensors."""
    return hashlib.sha256(
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()


def reference_root() -> Path:
    """Require an explicitly captured reference directory for GPU parity."""
    value = os.environ.get("UNISERVE_NATIVE_REFERENCE")
    if not value:
        raise RuntimeError(
            "set UNISERVE_NATIVE_REFERENCE to the native capture directory; "
            "candidate tests do not import or regenerate native references"
        )
    return Path(value)


def load_case(
    root: str | Path, name: str
) -> tuple[dict[str, torch.Tensor], dict]:
    """Verify manifest/file/tensor identities and return borrowed CPU values.

    Multiple safetensors shards may form a case, but tensor names must be
    unique. The caller selects devices only after immutable references have
    passed validation. No pickle, native imports or numerical regeneration.
    """
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "uniserve.native-reference.1":
        raise ValueError("unknown native reference manifest format")
    case = manifest["cases"][name]
    tensors = {}
    for relative, record in case["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix != ".safetensors":
            raise ValueError(
                "reference shard must be safetensors within its directory"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"reference shard changed: {relative}")
        fields = load_file(str(path), device="cpu")
        if (
            fields.keys() != record["tensors"].keys()
            or tensors.keys() & fields.keys()
        ):
            raise ValueError(
                "reference tensor names differ or overlap across shards"
            )
        for key, value in fields.items():
            descriptor = record["tensors"][key]
            if (
                list(value.shape) != descriptor["shape"]
                or str(value.dtype) != descriptor["dtype"]
                or tensor_sha256(value) != descriptor["sha256"]
            ):
                raise ValueError(f"reference tensor changed: {key}")
        tensors.update(fields)
    if not tensors:
        raise ValueError("native reference case contains no tensors")
    return tensors, case
