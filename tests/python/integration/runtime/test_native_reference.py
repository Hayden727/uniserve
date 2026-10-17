"""Saved-reference integrity is checked before candidate device placement."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from tests.python.integration.runtime.native_reference import (
    load_case,
    tensor_sha256,
)

pytestmark = pytest.mark.integration


def test_saved_reference_roundtrip_and_byte_tampering(tmp_path):
    fields = {
        "weight": torch.tensor([[1.5, -3.0]], dtype=torch.bfloat16),
        "alpha": torch.tensor(0.25),
        "mask": torch.tensor([True, False]),
    }
    path = tmp_path / "case.safetensors"
    save_file(fields, str(path))
    case = {
        "kind": "test",
        "config": {"width": 2},
        "files": {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tensors": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": tensor_sha256(value),
                    }
                    for name, value in fields.items()
                },
            }
        },
    }
    manifest = {"format": "uniserve.native-reference.1", "cases": {"one": case}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, metadata = load_case(tmp_path, "one")
    assert metadata["config"] == {"width": 2}
    for name, value in loaded.items():
        assert value.device.type == "cpu"
        torch.testing.assert_close(value, fields[name], rtol=0, atol=0)

    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="shard changed"):
        load_case(tmp_path, "one")


def test_reference_shard_cannot_escape_its_directory(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "uniserve.native-reference.1",
                "cases": {"one": {"files": {"../outside.safetensors": {}}}},
            }
        )
    )
    with pytest.raises(ValueError, match="within its directory"):
        load_case(tmp_path, "one")
