"""Static fused operations serve many layouts and release request values."""

import gc
import weakref

import pytest
import torch

from uniserve_models.wan import fusion

pytestmark = pytest.mark.unit


@torch.inference_mode()
def test_modulation_supports_more_than_eight_static_layouts_and_reuses_them():
    operation = fusion.compiled(fusion.layer_norm_modulate)
    cases = []
    for tokens in range(12, 0, -1):
        hidden = torch.full((1, tokens, 32), 2.0, dtype=torch.bfloat16)
        scale = torch.ones(1, 1, 32, dtype=torch.bfloat16)
        shift = torch.full_like(scale, 0.25)
        actual = operation(hidden, scale, shift, 1e-6, True)
        torch.testing.assert_close(
            actual, shift.expand_as(hidden), rtol=0, atol=0
        )
        cases.append((hidden, scale, shift))

    # New numerical values must reuse the warmed program, including after
    # visiting the other layouts. This public stance rejects recompilation.
    with torch.compiler.set_stance("fail_on_recompile"):
        for hidden, scale, shift in (*cases, cases[0]):
            hidden[..., ::2] = -1
            hidden[..., 1::2] = 1
            expected = hidden * 2 + 0.25
            torch.testing.assert_close(
                operation(hidden, scale, shift, 1e-6, True),
                expected,
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("axis", ["epsilon", "stride", "offset"])
@torch.inference_mode()
def test_static_scalar_and_strided_view_contracts(axis):
    operation = fusion.compiled(fusion.layer_norm_affine)
    cases = []
    weight = torch.ones(32, dtype=torch.bfloat16)
    bias = torch.full_like(weight, 0.25)
    for index in range(12):
        stride = index + 1 if axis == "stride" else 1
        offset = index if axis == "offset" else 0
        eps = 1e-6 * (index + 1) if axis == "epsilon" else 1e-6
        storage = torch.full((offset + 96 * stride,), 2.0, dtype=torch.bfloat16)
        hidden = storage.as_strided(
            (1, 3, 32), (96 * stride, 32 * stride, stride), offset
        )
        expected = bias.expand_as(hidden)
        torch.testing.assert_close(
            operation(hidden, weight, bias, eps), expected, rtol=0, atol=0
        )
        cases.append((hidden, eps))
    with torch.compiler.set_stance("fail_on_recompile"):
        for hidden, eps in reversed(cases):
            # Keyword calls consume the same declared numerical arguments.
            actual = operation(hidden, weight=weight, bias=bias, eps=eps)
            torch.testing.assert_close(
                actual, bias.expand_as(hidden), rtol=0, atol=0
            )


def _alias_cases():
    separate = [torch.full((1, 4, 16), float(value)) for value in (1, 2, 3)]
    left, middle, right = separate
    shared = torch.arange(192.0).remainder(4)
    a, b, c = (
        shared[start : start + 64].view(1, 4, 16) for start in (0, 64, 128)
    )
    overlap = shared[1:65].view(1, 4, 16)
    return (
        (left, middle, right),
        (left, left, right),
        (left, middle, left),
        (left, middle, middle),
        (left, left, left),
        (a, b, c),
        (a, overlap, c),
        (a, a.view_as(a), c),
        (a, a, c),
    )


@torch.inference_mode()
def test_repeated_arguments_and_shared_storage_reuse_without_value_retention():
    operation = fusion.compiled(fusion.gated_residual)
    cases = _alias_cases()
    for args in cases:
        for fused in (False, True):
            torch.testing.assert_close(
                operation(*args, fused),
                args[0] + args[1] * args[2],
                rtol=0,
                atol=0,
            )
    references = [weakref.ref(value) for args in cases for value in args]
    del args, cases
    gc.collect()
    assert all(reference() is None for reference in references)
    with torch.compiler.set_stance("fail_on_recompile"):
        for args in _alias_cases():
            # Fresh allocations preserve alias relationships without relying
            # on any previous request's tensor identity or storage address.
            args[0].add_(1)
            for fused in (False, True):
                torch.testing.assert_close(
                    operation(*args, fused),
                    args[0] + args[1] * args[2],
                    rtol=0,
                    atol=0,
                )


def test_grad_inference_autocast_dtype_and_parameter_variants():
    operation = fusion.compiled(fusion.gated_residual)

    def run():
        for context in (
            torch.no_grad(),
            torch.inference_mode(),
            torch.enable_grad(),
        ):
            with context:
                for dtype in (torch.float32, torch.bfloat16):
                    for parameter in (False, True):
                        hidden = torch.full((1, 3, 16), 2.0, dtype=dtype)
                        gate = torch.ones_like(hidden)
                        value = torch.full_like(hidden, 3.0)
                        if parameter:
                            hidden = torch.nn.Parameter(hidden)
                        for enabled in (False, True):
                            with torch.autocast(
                                "cpu", dtype=torch.bfloat16, enabled=enabled
                            ):
                                torch.testing.assert_close(
                                    operation(hidden, gate, value, True),
                                    torch.full_like(hidden, 5.0),
                                    rtol=0,
                                    atol=0,
                                )

    run()
    with torch.compiler.set_stance("fail_on_recompile"):
        run()
