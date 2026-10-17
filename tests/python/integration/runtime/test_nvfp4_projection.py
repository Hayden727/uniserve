"""Real deployed FP4 kernels through standalone and bound public layers.

UNISERVE_NATIVE_REFERENCE selects immutable cases captured in the native
environment. This candidate process needs only its locked dependencies.
"""

import pytest
import torch

from tests.python.integration.runtime.native_reference import (
    load_case,
    reference_root,
)
from uniserve.nn import Linear
from uniserve.nn.nvfp4 import NVFP4Linear
from uniserve.quantization.nvfp4 import NVFP4Epilogue, quantize
from uniserve.runtime import CUDAGraph, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("rows", [1, 32, 33, 128])
@pytest.mark.parametrize(
    "fuse_bias,apply_alpha,has_bias",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (True, False, False),
    ],
)
@torch.inference_mode()
def test_projection_matches_native_epilogue_at_row_boundary(
    rows, fuse_bias, apply_alpha, has_bias
):
    name = (
        f"nvfp4-m{rows}-f{int(fuse_bias)}-a{int(apply_alpha)}-b{int(has_bias)}"
    )
    fields, case = load_case(reference_root(), name)
    assert case["kind"] == "nvfp4"
    assert case["config"]["input_global_scale"] == 1.0
    channels = 64 if rows == 128 else 128
    source = Linear(
        256, channels, bias=has_bias, device="meta", dtype=torch.bfloat16
    )
    source.weight = torch.nn.Parameter(
        fields["weight"].to("cuda"), requires_grad=False
    )
    if has_bias:
        source.bias = torch.nn.Parameter(
            fields["bias"].to("cuda"), requires_grad=False
        )
    inputs = fields["input_storage"].to("cuda")[:, ::2]
    assert list(inputs.stride()) == case["config"]["input_stride"]
    torch.testing.assert_close(inputs.cpu(), fields["input"], rtol=0, atol=0)
    recipe = NVFP4Epilogue(fuse_bias=fuse_bias, apply_alpha=apply_alpha)
    layer = NVFP4Linear.from_linear(source, epilogue=recipe)
    expected = fields["output"].to("cuda")
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(
        layer.weight.buffers()["values"].cpu(),
        fields["weight_fp4"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layer.weight.buffers()["block_scale"].cpu(),
        fields["weight_scales"].reshape(-1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layer.alpha.cpu(), fields["alpha"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        layer.global_scale.cpu(), fields["global_scale"], rtol=0, atol=0
    )
    packed, scales = quantize(
        inputs.bfloat16().contiguous(),
        torch.ones((), device="cuda", dtype=torch.float32),
    )
    torch.testing.assert_close(
        packed.cpu(), fields["activation_fp4"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        scales.cpu(), fields["activation_scales"], rtol=0, atol=0
    )
    torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
    state = layer.state_dict()
    assert set(state) == {
        "weight",
        "global_scale",
        "alpha",
        *({"bias"} if has_bias else set()),
    }
    assert state["alpha"].dtype == state["global_scale"].dtype == torch.float32

    # Dense matmul policy must not replace an explicit FP4 numerical recipe.
    with ExecutionContext(layer, matmul="torch") as context:
        context.prepare(None)
        output = torch.empty(
            rows, 2 * channels, device="cuda", dtype=torch.bfloat16
        )[:, ::2]
        assert layer(inputs, out=output) is output
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert any(item["provider"] == "nvfp4" for item in context.kernels())
        if rows in (32, 33) and fuse_bias and apply_alpha:
            with CUDAGraph(context=context) as graph:
                graph.capture(lambda: layer(inputs, out=output))
                inputs.copy_(fields["input_replay"].to(inputs.device))
                graph.replay()
                torch.testing.assert_close(
                    output,
                    fields["output_replay"].to(output.device),
                    rtol=0,
                    atol=0,
                )


@torch.inference_mode()
def test_projection_checkpoint_constants_and_dense_rounding_hint_are_explicit():
    source = Linear(128, 128, device="cuda", dtype=torch.bfloat16)
    source.weight.fill_(0.125)
    source.bias.zero_()
    layer = NVFP4Linear.from_linear(source)
    values = torch.ones(33, 128, device="cuda", dtype=torch.bfloat16)
    restored = NVFP4Linear.from_linear(source)
    restored.load_state_dict(layer.state_dict(), strict=True)
    torch.testing.assert_close(restored(values), layer(values), rtol=0, atol=0)
    with pytest.raises(ValueError, match="explicit epilogue"):
        layer(values, round_before_bias=True)
    with pytest.raises(ValueError, match="BF16 output"):
        layer(values, output_dtype=torch.float32)
