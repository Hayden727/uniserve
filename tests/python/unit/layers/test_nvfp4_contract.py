"""Native deployment FP4 scaling stays distinct from generic NVFP4."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from uniserve.quantization.nvfp4 import (
    NVFP4Epilogue,
    bias_over_alpha,
    fold_gate,
    weight_scales,
)

pytestmark = pytest.mark.unit


def test_weight_encoder_scale_precedes_bfloat16_alpha_rounding():
    weight = torch.tensor([[-5.0, 1.0]], dtype=torch.bfloat16)
    scales = weight_scales(weight)

    assert scales.global_scale.dtype == torch.float32
    assert scales.stored_global_scale.dtype == torch.bfloat16
    assert scales.alpha.dtype == torch.float32
    # Scalar / Tensor evaluates reciprocal then multiplication in PyTorch.
    # For amax=5 this is one FP32 ULP above rounding real 2688/5 once.
    torch.testing.assert_close(
        scales.global_scale, torch.tensor(537.60003662109375), rtol=0, atol=0
    )
    assert scales.stored_global_scale.item() == 536.0
    torch.testing.assert_close(
        scales.alpha, torch.tensor(1.0) / 536.0, rtol=0, atol=0
    )
    assert scales.alpha != 1.0 / scales.global_scale
    with pytest.raises(FrozenInstanceError):
        scales.alpha = torch.tensor(0.0)


def test_zero_weight_retains_native_unclamped_scale_contract():
    scales = weight_scales(torch.zeros(2, 32, dtype=torch.bfloat16))
    assert scales.global_scale.isinf()
    assert scales.stored_global_scale.isinf()
    assert scales.alpha.item() == 0.0


def test_bias_and_gate_fold_keep_their_separate_rounding_domains():
    alpha = weight_scales(torch.tensor([[5.0]], dtype=torch.bfloat16)).alpha
    bias = torch.tensor([0.0039215087890625, -0.5078125], dtype=torch.bfloat16)
    folded = bias_over_alpha(bias, alpha)
    torch.testing.assert_close(
        folded, (bias.float() * 536.0).bfloat16(), rtol=0, atol=0
    )
    gate = torch.tensor([1.0078125, -0.99609375], dtype=torch.bfloat16)
    result = fold_gate(gate, float(alpha))
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result, (gate.double() * float(alpha)).bfloat16(), rtol=0, atol=0
    )
    assert fold_gate(gate, 1.0) is gate


def test_epilogue_selection_is_immutable_and_explicit():
    epilogue = NVFP4Epilogue(fuse_bias=True, apply_alpha=False)
    with pytest.raises(FrozenInstanceError):
        epilogue.apply_alpha = True
    with pytest.raises(ValueError, match="boolean"):
        NVFP4Epilogue(fuse_bias=1)
    with pytest.raises(ValueError, match="BF16"):
        weight_scales(torch.ones(2, 32))
