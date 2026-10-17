"""Attention provider choices belong to one public execution context."""

import pytest
import torch
import torch.nn.functional as F

from uniserve.nn.attention import Attention, AttentionBatch, DenseInput
from uniserve.runtime import ExecutionContext

pytestmark = pytest.mark.unit


@torch.inference_mode()
def test_attention_overrides_are_context_local_and_reported_per_layer():
    model = torch.nn.ModuleDict(
        {"main": Attention(2, 2, 8), "camera": Attention(2, 2, 8)}
    )
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    inputs = AttentionBatch.single(DenseInput(False, None))
    expected = F.scaled_dot_product_attention(q, k, v)
    declarations = {model["camera"]: "torch"}
    configured = ExecutionContext(
        model, attention="auto", attention_overrides=declarations
    )
    # A caller's mutable launch map cannot retarget an existing context.
    declarations.clear()
    for context, providers in (
        (configured, {"main": "auto", "camera": "torch"}),
        (
            ExecutionContext(model, attention="auto"),
            {"main": "auto", "camera": "auto"},
        ),
    ):
        with context:
            context.prepare(None)
            for layer in model.values():
                torch.testing.assert_close(
                    layer(q, k, v, inputs), expected, rtol=0, atol=0
                )
            assert {
                record["path"]: record["provider"].split(":", 1)[0]
                for record in context.kernels()
                if record["op"] == "attention"
            } == providers


def test_attention_overrides_reject_unrelated_modules_and_unknown_providers():
    root, outside = Attention(2, 2, 8), Attention(2, 2, 8)
    with pytest.raises(ValueError, match="within the execution root"):
        ExecutionContext(root, attention_overrides={outside: "torch"})
    with pytest.raises(ValueError, match="unknown attention backend"):
        ExecutionContext(root, attention_overrides={root: "misspelled"})
