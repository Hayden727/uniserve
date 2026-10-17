"""Student exports preserve camera/gate weights and reject incomplete data."""

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    student = {
        "blocks.0.cam.weight": torch.arange(12, dtype=torch.bfloat16)
        .view(3, 4)
        .t(),
        "blocks.0.quant_gate": torch.tensor(0.75, dtype=torch.float32),
        "blocks.0.bias": torch.tensor([-0.0, 2.0], dtype=torch.float16),
    }
    source = tmp_path / "dcp"
    state = {"roles.student.transformer." + k: v for k, v in student.items()}
    state["roles.teacher.transformer.unused"] = torch.ones(3)
    state["optimizer.exp_avg"] = torch.ones(4)
    dcp.save(state, checkpoint_id=source, no_dist=True)
    return source, student


def test_export_preserves_all_student_values_without_training_state(
    checkpoint, tmp_path
):
    from uniserve_models.wan.weights import export_student, verify_export

    source, student = checkpoint
    output = tmp_path / "student"
    manifest = export_student(source, output, max_shard_bytes=24)
    assert set(manifest["tensors"]) == set(student)
    restored = {}
    for filename in manifest["shards"]:
        restored.update(load_file(output / filename))
    assert set(restored) == set(student)
    for key, expected in student.items():
        assert restored[key].dtype == expected.dtype
        assert restored[key].shape == expected.shape
        assert torch.equal(
            restored[key].reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )
    assert verify_export(source, output)["verified_tensors"] == len(student)


def test_export_never_overwrites_existing_directory(checkpoint, tmp_path):
    from uniserve_models.wan.weights import export_student

    source, _ = checkpoint
    output = tmp_path / "student"
    output.mkdir()
    original = output / "user-data.txt"
    original.write_text("keep")
    with pytest.raises(FileExistsError):
        export_student(source, output)
    assert original.read_text() == "keep"


def test_verification_rejects_changed_gate_and_missing_tensor(
    checkpoint, tmp_path
):
    from uniserve_models.wan.weights import export_student, verify_export

    source, _ = checkpoint
    output = tmp_path / "student"
    manifest = export_student(source, output)
    shard = output / next(iter(manifest["shards"]))
    values = load_file(shard)
    values["blocks.0.quant_gate"] += 1
    save_file(values, shard)
    with pytest.raises(ValueError, match="checksum"):
        verify_export(source, output)
    # Missing metadata must fail even if a caller rewrites the JSON.
    manifest["tensors"].pop("blocks.0.cam.weight")
    (output / "student_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="key set"):
        verify_export(source, output)


def test_verification_rejects_unfinished_export(checkpoint, tmp_path):
    from uniserve_models.wan.weights import export_student, verify_export

    source, _ = checkpoint
    output = tmp_path / "student"
    export_student(source, output)
    (output / ".incomplete").touch()
    with pytest.raises(ValueError, match="incomplete"):
        verify_export(source, output)
