"""Text numerical metadata preserves ordinary multi-token encoding."""

from dataclasses import replace

import pytest
import torch

from uniserve_models.wan.text import UMT5, TextConfig

pytestmark = pytest.mark.unit


@torch.inference_mode()
def test_explicit_single_row_recipe_keeps_multi_token_outputs():
    config = TextConfig(
        vocab_size=64,
        hidden_size=16,
        head_dim=8,
        ffn_dim=32,
        num_layers=2,
        num_heads=2,
    )
    ordinary = UMT5(config).eval()
    ordered = UMT5(
        replace(
            config,
            attention_gemv_reduction="strided16",
            ffn_output_gemv_reduction="strided16",
        )
    ).eval()
    ordered.load_state_dict(ordinary.state_dict())
    for length in (2, 7):
        ids = torch.arange(length)[None]
        torch.testing.assert_close(ordered(ids), ordinary(ids), rtol=0, atol=0)


@pytest.mark.parametrize(
    "field", ["attention_gemv_reduction", "ffn_output_gemv_reduction"]
)
def test_checkpoint_recipe_requires_a_known_reduction(field):
    with pytest.raises(ValueError, match="GEMV reduction"):
        TextConfig(**{field: "unknown"})
