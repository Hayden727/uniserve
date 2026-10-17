"""Export and verify the student transformer of a trusted Wan DCP.

Only the student tensor namespace is requested from Distributed Checkpoint;
optimizer, teacher and RNG state are never materialized. Conversion retains
the source dtype. Each shard is checked against its in-memory source before
the export becomes loadable; ``verify_export`` independently re-reads DCP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

PREFIX = "roles.student.transformer."


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    # Viewing bytes also distinguishes signed zero and preserves BF16 bits.
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def _reader(checkpoint: str | Path):
    root = Path(checkpoint).resolve()
    if not (root / ".metadata").is_file():
        root = root / "dcp"
    reader = dcp.FileSystemReader(root)
    metadata = reader.read_metadata()
    tensors = {}
    for key, item in metadata.state_dict_metadata.items():
        if key.startswith(PREFIX):
            if not isinstance(item, TensorStorageMetadata):
                raise ValueError(f"student entry is not a tensor: {key}")
            name = key.removeprefix(PREFIX)
            if not name:
                raise ValueError("student tensor name must be nonempty")
            tensors[name] = item
    if not tensors:
        raise ValueError("checkpoint contains no student transformer tensors")
    return root, reader, dict(sorted(tensors.items()))


def _load(reader, metadata, names):
    state = {
        PREFIX + name: torch.empty(
            metadata[name].size,
            dtype=metadata[name].properties.dtype,
            device="cpu",
        )
        for name in names
    }
    dcp.load(state, storage_reader=reader, no_dist=True)
    return {key.removeprefix(PREFIX): value for key, value in state.items()}


def export_student(
    checkpoint: str | Path,
    output: str | Path,
    *,
    max_shard_bytes: int = 2 * 1024**3,
) -> dict:
    """Create a new student-only safetensors directory without overwriting.

    A single tensor larger than the shard limit occupies its own shard.
    Interrupted exports retain ``.incomplete`` and are rejected by readers.
    Files and the marker remain available for diagnosing a failed conversion.
    """
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be a positive integer")
    root, reader, metadata = _reader(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / ".incomplete"
    marker.touch()
    groups, names, size = [], [], 0
    for name, item in metadata.items():
        nbytes = item.size.numel() * item.properties.dtype.itemsize
        if names and size + nbytes > max_shard_bytes:
            groups.append(names)
            names, size = [], 0
        names.append(name)
        size += nbytes
    if names:
        groups.append(names)

    manifest = {
        "format": "wan-student-v1",
        "source": str(root),
        "source_metadata_sha256": _file_hash(root / ".metadata"),
        "source_prefix": PREFIX,
        "torch_version": torch.__version__,
        "tensors": {},
        "shards": {},
    }
    total_bytes, weight_map = 0, {}
    for index, names in enumerate(groups, 1):
        values = _load(reader, metadata, names)
        filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        save_file(values, output / filename, metadata={"format": "pt"})
        restored = load_file(output / filename)
        if set(restored) != set(values):
            raise ValueError(f"export shard key set mismatch: {filename}")
        for name, value in values.items():
            digest = _tensor_hash(value)
            if (
                restored[name].shape != value.shape
                or restored[name].dtype != value.dtype
                or _tensor_hash(restored[name]) != digest
            ):
                raise ValueError(f"export tensor mismatch: {name}")
            manifest["tensors"][name] = {
                "source_key": PREFIX + name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": digest,
                "shard": filename,
            }
            weight_map[name] = filename
            total_bytes += value.numel() * value.element_size()
        manifest["shards"][filename] = _file_hash(output / filename)
        print(
            f"verified shard {index}/{len(groups)}: {len(names)} tensors",
            flush=True,
        )
        del values, restored

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    (output / "student_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    marker.unlink()
    return manifest


def verify_export(checkpoint: str | Path, output: str | Path) -> dict:
    """Re-read every student tensor and require exact exported byte equality."""
    output = Path(output)
    if (output / ".incomplete").exists():
        raise ValueError("incomplete student export")
    manifest = json.loads((output / "student_manifest.json").read_text())
    root, reader, metadata = _reader(checkpoint)
    if manifest["format"] != "wan-student-v1":
        raise ValueError("unknown student export format")
    if set(manifest["tensors"]) != set(metadata):
        raise ValueError("student export key set differs from checkpoint")
    if manifest["source_metadata_sha256"] != _file_hash(root / ".metadata"):
        raise ValueError("source metadata checksum mismatch")
    expected_map = {k: v["shard"] for k, v in manifest["tensors"].items()}
    index = json.loads((output / "model.safetensors.index.json").read_text())
    if index["weight_map"] != expected_map:
        raise ValueError("safetensors index key set or shard mapping mismatch")
    if set(expected_map.values()) != set(manifest["shards"]):
        raise ValueError("student export shard set mismatch")
    verified = 0
    for filename, digest in manifest["shards"].items():
        if Path(filename).name != filename or not filename.endswith(
            ".safetensors"
        ):
            raise ValueError("shard filename must stay within export directory")
        if _file_hash(output / filename) != digest:
            raise ValueError(f"shard checksum mismatch: {filename}")
        names = [
            name for name, shard in expected_map.items() if shard == filename
        ]
        restored = load_file(output / filename)
        if set(restored) != set(names):
            raise ValueError(f"shard tensor key set mismatch: {filename}")
        source = _load(reader, metadata, names)
        for name in names:
            item, value, original = (
                manifest["tensors"][name],
                restored[name],
                source[name],
            )
            if (
                item["source_key"] != PREFIX + name
                or item["shape"] != list(original.shape)
                or item["dtype"] != str(original.dtype)
                or value.shape != original.shape
                or value.dtype != original.dtype
                or item["sha256"] != _tensor_hash(original)
                or item["sha256"] != _tensor_hash(value)
            ):
                raise ValueError(f"student tensor verification failed: {name}")
            verified += 1
        del restored, source
    return {"verified_tensors": verified, "shards": len(manifest["shards"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-shard-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if args.verify:
        result = verify_export(args.checkpoint, args.output)
    else:
        manifest = export_student(
            args.checkpoint, args.output, max_shard_bytes=args.max_shard_bytes
        )
        result = {"exported_tensors": len(manifest["tensors"])}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
