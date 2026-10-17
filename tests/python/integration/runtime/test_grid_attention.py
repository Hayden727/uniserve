"""Real FA4 grid segments and the native full-grid tensor-reference boundary."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.python.integration.runtime.native_reference import load_case
from uniserve.nn.attention.grid import GridAttention, GridLayout
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve.runtime.backends.attention import grid as grid_backend

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _backend():
    name = os.environ.get("UNISERVE_GRID_BACKEND", "flash_attn_4_sm100")
    if name not in ("flash_attn_4_sm100", "flash_attn_4"):
        raise ValueError("grid GPU checks require an explicit FA4 provider")
    return name


def _file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _difference(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    mismatch, maximum, first = 0, 0.0, None
    for offset in range(0, actual.numel(), 2**20):
        left, right = (
            actual[offset : offset + 2**20],
            expected[offset : offset + 2**20],
        )
        different = left != right
        count = int(different.sum())
        mismatch += count
        if count and first is None:
            first = offset + int(different.nonzero()[0, 0])
        maximum = max(
            maximum, float((left.float() - right.float()).abs().max())
        )
    return {
        "mismatched": mismatch,
        "max_abs": maximum,
        "first_flat_index": first,
    }


def _environment():
    from uniserve_kernels.attention import grid as grid_merge

    from uniserve.nn.attention import grid as grid_layer

    versions = {}
    for name in ("torch", "flash-attn-4", "nvidia-cutlass-dsl", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = [
        Path(__file__),
        Path(grid_backend.__file__),
        Path(grid_layer.__file__),
        Path(grid_merge.__file__),
    ]
    if _backend() == "flash_attn_4_sm100":
        from uniserve_kernels.attention import dense_sm100

        sources.append(Path(dense_sm100.__file__))
    else:
        from flash_attn.cute import flash_fwd_sm100, interface, softmax

        sources.extend(
            Path(module.__file__)
            for module in (interface, flash_fwd_sm100, softmax)
        )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "visible_devices": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "world_size": 1,
        "source_sha256": {str(path): _file_sha256(path) for path in sources},
    }


@pytest.mark.parametrize("heads", [1, 5])
@torch.inference_mode()
def test_fa4_grid_segments_obey_the_dense_union_on_real_cuda(heads):
    device = torch.device("cuda", 0)
    camera = torch.tensor([0, 1, 2, 0, 1, 2])
    column = torch.tensor([0, 0, 0, 1, 1, 1])
    layout = GridLayout.from_cells(camera, column, 16, device=device)
    generator = torch.Generator(device=device).manual_seed(110)
    q, k, v = [
        torch.randn(
            96, heads, 128, device=device, generator=generator
        ).bfloat16()
        for _ in range(3)
    ]
    result = GridAttention(heads, 128)(
        q, k, v, layout, operator=grid_backend.resolve(_backend(), device)
    )
    cameras, columns = (
        camera.repeat_interleave(16),
        column.repeat_interleave(16),
    )
    allowed = (
        (columns[:, None] == columns[None, :])
        | (cameras[:, None] == cameras[None, :])
        | (cameras[None, :] == 0)
    ).to(device)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * 128**-0.5
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    expected = torch.einsum("hqk,khd->qhd", weights, v.float()).bfloat16()
    torch.testing.assert_close(
        result, expected, rtol=0, atol=4 * torch.finfo(torch.bfloat16).eps
    )


@torch.inference_mode()
def test_full_grid_fa4_matches_saved_native_output_bitwise():
    # Generate this reference in the independent native environment. Do not
    # import its research/training stack into the candidate worker runtime.
    reference_path = os.environ.get("UNISERVE_GRID_NATIVE_REFERENCE")
    if reference_path is None:
        pytest.skip(
            "set UNISERVE_GRID_NATIVE_REFERENCE for the full native fixture"
        )
    reference = Path(reference_path)
    expected_hash = os.environ.get("UNISERVE_GRID_NATIVE_SHA256")
    assert expected_hash, "the full native fixture requires its recorded SHA256"
    identity = reference / "manifest.json" if reference.is_dir() else reference
    digest = _file_sha256(identity)
    assert digest == expected_hash
    case = None
    if reference.is_dir():
        assert not (reference / ".incomplete").exists()
        heads = int(os.environ.get("UNISERVE_GRID_HEADS", "1"))
        assert heads in (1, 5)
        values, case = load_case(reference, f"grid_heads{heads}")
        assert case["kind"] == "full_grid_attention"
        for key, expected in (
            ("views", 32),
            ("columns", 16),
            ("cell_tokens", 1560),
            ("tokens", 798720),
            ("heads", heads),
            ("head_dim", 128),
            ("order", "time_major"),
            ("scale", 128**-0.5),
            ("execution_world_size", 1),
        ):
            assert case["config"][key] == expected
        assert case["rng"]["seed"] == 712
        assert case["config"]["segments"] == ["D1", "A", "C", "B_PRIME"]
    else:
        values = load_file(reference, device="cpu")
    assert {
        "q",
        "k",
        "v",
        "output",
        "cell_camera",
        "cell_column",
    } <= set(values)
    assert values["q"].shape[0] == 798720
    assert values["q"].shape[1] in (1, 5)
    assert values["q"].shape[2] == 128
    assert values["q"].dtype == torch.bfloat16
    if case is not None:
        assert values["q"].shape[1] == heads
        assert torch.equal(values["cell_camera"], torch.arange(32).repeat(16))
        assert torch.equal(
            values["cell_column"], torch.arange(16).repeat_interleave(32)
        )
        assert (
            case["rng"]["draws"]
            == [
                {
                    "op": "randn",
                    "shape": [1, 798720, heads, 128],
                    "dtype": "torch.float32",
                    "stored_dtype": "torch.bfloat16",
                }
            ]
            * 3
        )
    for key in ("k", "v", "output"):
        assert values[key].shape == values["q"].shape
        assert values[key].dtype == torch.bfloat16
    device = torch.device("cuda", 0)
    layout = GridLayout.from_cells(
        values["cell_camera"], values["cell_column"], 1560, device=device
    )
    assert (layout.views, layout.columns) == (32, 16)
    q, k, v = (values[name].to(device) for name in ("q", "k", "v"))
    destination = os.environ.get("UNISERVE_GRID_RESULT_DIR")
    result_dir = None if destination is None else Path(destination)
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "full grid numerical diagnostic; excluded from timing",
        "backend": _backend(),
        "environment": _environment(),
        "reference_sha256": digest,
        "shape": list(values["q"].shape),
        "threshold": {"rtol": 0, "atol": 0},
        "segments": [],
        "passed": False,
    }

    def publish():
        if result_dir is not None:
            (result_dir / "comparison.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )

    operator = grid_backend.resolve(_backend(), device)

    def record_segment(query, key, value, *, scale, out=None):
        state = operator(query, key, value, scale=scale, out=out)
        if case is not None:
            name = case["config"]["segments"][len(report["segments"])]
            report["segments"].append(
                {
                    "name": name,
                    "output": _difference(
                        state.output.cpu(), values[f"segment.{name}.output"]
                    ),
                    "lse": _difference(
                        state.lse.cpu(), values[f"segment.{name}.lse"]
                    ),
                }
            )
            publish()
        return state

    publish()
    result = GridAttention(q.shape[1], 128)(
        q, k, v, layout, operator=record_segment
    )
    actual = result.cpu().contiguous()
    report["output"] = _difference(actual, values["output"])
    report["passed"] = report["output"]["mismatched"] == 0
    if case is not None:
        assert len(report["segments"]) == 4
    if result_dir is not None:
        output_path = result_dir / "output.safetensors"
        save_file({"output": actual}, str(output_path))
        report["output_sha256"] = _file_sha256(output_path)
    publish()
    print(json.dumps(report), flush=True)
    torch.testing.assert_close(actual, values["output"], rtol=0, atol=0)


@torch.inference_mode()
def test_bound_grid_captures_and_replays_changed_projections():
    device = torch.device("cuda", 0)
    layout = GridLayout.from_cells(
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
        16,
        device=device,
    )
    q = torch.zeros(64, 1, 128, dtype=torch.bfloat16, device=device)
    v = torch.ones_like(q)
    layer = GridAttention(1, 128)
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    stream.wait(torch.cuda.current_stream(device))
    try:
        with (
            stream,
            ExecutionContext(layer, stream=stream, grid=_backend()) as context,
        ):
            context.prepare(None)
            result = layer(q, q, v, layout)
            torch.testing.assert_close(result, v, rtol=0, atol=0)
            with CUDAGraph(context=context) as graph:
                graph.capture(lambda: layer(q, q, v, layout, out=result))
                v.fill_(3)
                graph.replay()
                torch.testing.assert_close(result, v, rtol=0, atol=0)
    finally:
        stream.close()
