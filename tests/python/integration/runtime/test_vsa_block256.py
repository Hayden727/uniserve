"""Logical 128/256-token VSA against immutable native FA4 captures.

UNISERVE_NATIVE_REFERENCE selects the separately captured archive directory.
The candidate process imports no native research packages.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from uniserve_kernels.attention import vsa_native

from tests.python.integration.runtime.native_reference import (
    load_case,
    reference_root,
)
from uniserve.nn.attention import vsa
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve.runtime.backends.attention.vsa.sparse import (
    buffer_configs,
    split_blocks,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def stream():
    if not vsa_native.supported():
        pytest.skip("the sparse FA4 provider requires SM100")
    owner = CUDAStream.external(torch.cuda.Stream())
    yield owner
    owner.close()


def _batch(indices, counts, valid, *, tile=256):
    return vsa.BlockInput(
        vsa.Pattern(((indices.shape[-1],) * indices.shape[1],), 0, 0, tile),
        indices,
        counts,
        valid,
        0,
    )


@pytest.mark.parametrize(
    ("tile", "recipe"),
    [(128, None), (256, None), (256, vsa.Recipe.FA4_128X128)],
    ids=["logical128", "logical256", "hoisted256"],
)
@torch.inference_mode()
def test_sparse_preserves_strided_values_partial_keys_and_live_graph_maps(
    stream, tile, recipe
):
    device = stream.device
    projected = torch.zeros(
        5 * tile, 2, 3, 128, device=device, dtype=torch.bfloat16
    )
    query, key, value = projected.unbind(2)
    query = query[: 3 * tile]
    value.fill_(-32)  # Finite poisoned padding must never enter a softmax.
    for start, length, number in (
        (0, tile, 2),
        (tile, tile // 2 + 1, 6),
        (2 * tile, tile // 2, 8),
        (3 * tile, 1, 6),
    ):
        value[start : start + length] = number
    valid = torch.tensor(
        [tile, tile // 2 + 1, tile // 2, 1, 0], dtype=torch.int32, device=device
    )
    indices = torch.tensor(
        [[[0, 2], [1, 3], [4, 0]], [[0, 0], [2, 0], [3, 0]]],
        dtype=torch.int32,
        device=device,
    )
    counts = torch.tensor(
        [[2, 2, 1], [1, 1, 1]], dtype=torch.int32, device=device
    )
    batch = _batch(indices, counts, valid, tile=tile)
    module = vsa.BlockAttention(128**-0.5, recipe=recipe)
    expected = torch.tensor(
        [[4, 2], [6, 8], [0, 6]], dtype=torch.bfloat16, device=device
    )
    expected = (
        expected.repeat_interleave(tile, 0).unsqueeze(-1).expand_as(query)
    )
    stream.wait(torch.cuda.current_stream(device))
    with ExecutionContext(module, stream=stream, vsa="sm100") as context:
        context.prepare(None)
        actual = module(query, key, value, batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        replayed = torch.empty_like(query)
        with CUDAGraph(context=context) as graph:
            graph.capture(
                lambda: replayed.copy_(module(query, key, value, batch))
            )
            counts[0, 0] = 0
            indices[0, 1] = 0
            counts[0, 1] = 1
            valid[2] = 0
            graph.replay()
            changed = torch.tensor(
                [[0, 2], [2, 0], [0, 6]], dtype=torch.bfloat16, device=device
            )
            changed = (
                changed.repeat_interleave(tile, 0)
                .unsqueeze(-1)
                .expand_as(query)
            )
            torch.testing.assert_close(replayed, changed, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [10, 40])
@torch.inference_mode()
def test_block256_meets_fixed_fidelity_gate_against_native_fa4(stream, heads):
    _check_native_fidelity(stream, heads, tile=256)


@pytest.mark.parametrize("heads", [10, 40])
@torch.inference_mode()
def test_block128_meets_fixed_fidelity_gate_against_native_fa4(stream, heads):
    _check_native_fidelity(stream, heads, tile=128)


def _check_native_fidelity(
    stream,
    heads,
    *,
    tile,
    recipe=vsa.Recipe.FA4_128X64,
    name=None,
    captured=None,
):
    name = name or f"vsa{tile}-h{heads}"
    root = reference_root()
    fields, case = load_case(root, name) if captured is None else captured
    assert case["kind"] == f"vsa{tile}"
    assert case["config"]["heads"] == heads
    key_tile = 64 if recipe is vsa.Recipe.FA4_128X64 else 128
    assert case["config"]["native_tile_mn"] == [128, key_tile]
    assert case["config"].get("recipe", "fa4_128x64") == recipe.value
    assert case["config"]["logical_block"] == tile
    _check_native_lists(fields, tile=tile, recipe=recipe)
    device = stream.device
    projected = fields["projected"].to(device)
    query, key, value = projected.unbind(2)
    query = query[: case["config"]["query_rows"]]
    assert list(query.stride()) == case["config"]["query_stride"]
    valid = fields["valid"].to(device)
    indices = fields["indices"].to(device)
    counts = fields["counts"].to(device)
    batch = _batch(indices, counts, valid, tile=tile)
    reference = fields["output"].to(device)
    module = vsa.BlockAttention(128**-0.5, recipe=recipe)
    stream.wait(torch.cuda.current_stream(device))
    with ExecutionContext(module, stream=stream, vsa="sm100") as context:
        context.prepare(None)
        actual = module(query, key, value, batch)
        candidate, expected = (
            actual.double().flatten(),
            reference.double().flatten(),
        )
        assert torch.isfinite(candidate).all()
        assert torch.isfinite(expected).all()
        error = candidate - expected
        nrmse = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
            expected
        ).clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            candidate, expected, dim=0
        )
        result = {
            "case": name,
            "recipe": recipe.value,
            "heads": heads,
            "nrmse": nrmse.item(),
            "cosine": cosine.item(),
            "max_abs": error.abs().max().item(),
            "mismatched": (candidate != expected).sum().item(),
            "shape": list(actual.shape),
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
            },
            "reference": str(root),
            "manifest_sha256": hashlib.sha256(
                (root / "manifest.json").read_bytes()
            ).hexdigest(),
            "limits": {"nrmse_max": 0.001, "cosine_min": 0.99999},
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        result_directory = os.environ.get("UNISERVE_VSA_RESULT_DIR")
        if result_directory:
            directory = Path(result_directory)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{name}.json").open("x") as output:
                json.dump(result, output, indent=2, sort_keys=True)
                output.write("\n")
        # Fixed before observing candidate outputs: PROTOCOL.md v1's
        # equivalent-backend gate, with no BF16 tolerance relaxation.
        assert nrmse.item() <= 0.001, (
            f"NRMSE={nrmse.item():.9g}; max_abs={error.abs().max().item():.9g}"
        )
        assert cosine.item() >= 0.99999, f"cosine={cosine.item():.12g}"


def _check_native_lists(fields, *, tile, recipe=vsa.Recipe.FA4_128X64):
    """Localize traversal differences against independently captured maps."""
    indices, counts, valid = (
        fields[name] for name in ("indices", "counts", "valid")
    )
    maps = {
        name: torch.empty(field.shape, dtype=field.dtype)
        for name, field in buffer_configs(
            *indices.shape, valid.numel(), tile=tile, recipe=recipe
        ).items()
    }
    split_blocks(indices, counts, valid, out=maps, tile=tile, recipe=recipe)
    torch.testing.assert_close(
        maps["valid"], fields["native_child_valid"], rtol=0, atol=0
    )
    for head in range(indices.shape[0]):
        for query in range(indices.shape[1]):
            reference = []
            for kind in ("mask", "full"):
                length = fields[f"native_{kind}_counts"][0, head, query].item()
                reference.extend(
                    fields[f"native_{kind}_indices"][0, head, query, :length]
                    .flip(0)
                    .tolist()
                )
            actual = []
            for kind in ("mask", "full"):
                length = maps[f"{kind}_counts"][0, head, query].item()
                actual.extend(
                    maps[f"{kind}_indices"][0, head, query, :length]
                    .flip(0)
                    .tolist()
                )
            assert actual == reference, (head, query)
