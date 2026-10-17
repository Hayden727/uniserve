"""Replay immutable native FP4 calls through the candidate runtime.

The native environment records original weights, inputs, numerical payloads
and outputs with capture_native_operators.py (or a model observer using the
same uniserve.native-reference.1 manifest). This process verifies hashes and
uses only the locked candidate dependencies. It never loads native modules.

Run:
    python tests/python/integration/runtime/replay_nvfp4.py --reference DIR
"""

import argparse
import json
from pathlib import Path

import torch

from tests.python.integration.runtime.native_reference import load_case
from uniserve.nn import Linear
from uniserve.nn.nvfp4 import NVFP4Linear
from uniserve.quantization.nvfp4 import NVFP4Epilogue
from uniserve.runtime import CUDAGraph, ExecutionContext


def measured(actual, expected):
    """Apply the unchanged equivalent-backend fidelity gate."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("candidate changed output shape or dtype")
    value, reference = actual.double().flatten(), expected.double().flatten()
    if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
        raise AssertionError("candidate or captured native output is nonfinite")
    error = value - reference
    norm = torch.linalg.vector_norm(reference)
    nrmse = (torch.linalg.vector_norm(error) / norm.clamp_min(1e-12)).item()
    cosine = (
        float(torch.equal(value, reference))
        if norm.item() == 0.0
        else torch.nn.functional.cosine_similarity(
            value, reference, dim=0
        ).item()
    )
    result = {
        "nrmse": nrmse,
        "cosine": cosine,
        "max_abs": error.abs().max().item(),
        "exact": torch.equal(actual, expected),
    }
    if nrmse > 0.001 or cosine < 0.99999:
        raise AssertionError(json.dumps(result))
    return result


@torch.inference_mode()
def replay_case(root: str | Path, name: str, *, device: str = "cuda:0") -> dict:
    """Run a saved native call through standalone, bound and graph paths."""
    fields, case = load_case(root, name)
    if case["kind"] != "nvfp4":
        raise ValueError("expected a native NVFP4 case")
    weight, expected = fields["weight"], fields["output"].to(device)
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("replay requires the original BF16 weight")
    source = Linear(
        weight.shape[1],
        weight.shape[0],
        bias="bias" in fields,
        device="meta",
        dtype=torch.bfloat16,
    )
    source.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
    if source.bias is not None:
        source.bias = torch.nn.Parameter(
            fields["bias"].to(device), requires_grad=False
        )
    config = case["config"]
    recipe = NVFP4Epilogue(
        fuse_bias=config["fuse_bias"],
        apply_alpha=config["apply_alpha"],
    )
    candidate = NVFP4Linear.from_linear(source, epilogue=recipe)
    if "input_storage" in fields:
        inputs = fields["input_storage"].to(device)[:, ::2]
        if list(inputs.stride()) != config["input_stride"]:
            raise AssertionError("saved noncontiguous input layout changed")
        torch.testing.assert_close(
            inputs.cpu(), fields["input"], rtol=0, atol=0
        )
    else:
        inputs = fields["input"].to(device)
    results = {"standalone": measured(candidate(inputs), expected)}
    with ExecutionContext(candidate, matmul="torch") as context:
        context.prepare(None)
        results["bound"] = measured(candidate(inputs), expected)
        captured = torch.empty_like(expected)
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: candidate(inputs, out=captured))
            graph.replay()
            results["graph"] = measured(captured, expected)
            if "input_replay" in fields:
                inputs.copy_(fields["input_replay"].to(device))
                graph.replay()
                results["graph_changed"] = measured(
                    captured, fields["output_replay"].to(device)
                )
        providers = context.kernels()
    return {
        "case": name,
        "reference": str(root),
        "config": config,
        "candidate_torch": torch.__version__,
        "candidate_cuda": torch.version.cuda,
        "kernels": providers,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads((args.reference / "manifest.json").read_text())
    names = args.case or [
        name
        for name, case in manifest["cases"].items()
        if case["kind"] == "nvfp4"
    ]
    if not names:
        raise ValueError("reference manifest has no NVFP4 cases")
    for name in names:
        print(
            json.dumps(replay_case(args.reference, name, device=args.device)),
            flush=True,
        )


if __name__ == "__main__":
    main()
