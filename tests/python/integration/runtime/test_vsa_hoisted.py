"""Formal 128x128/q_stage2 VSA against separate immutable native captures.

The full-grid cases retain all 798720 live tokens and 3584 logical tiles.
They check the fine operator after head exchange, not distributed transport
or the learned coarse branch. Use distinct output directories per run.
"""

import pytest
import torch

from tests.python.integration.runtime.native_reference import (
    load_case,
    reference_root,
)
from tests.python.integration.runtime.test_vsa_block256 import (
    _check_native_fidelity,
)
from tests.python.integration.runtime.test_vsa_block256 import (
    stream as stream,
)
from uniserve.nn.attention.vsa import Recipe

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("heads", [1, 5])
@torch.inference_mode()
def test_hoisted_partial_blocks_meet_native_gate(stream, heads):
    _check_native_fidelity(
        stream,
        heads,
        tile=256,
        recipe=Recipe.FA4_128X128,
        name=f"vsa256-hoisted-h{heads}",
    )


@pytest.mark.parametrize("heads", [1, 5])
@torch.inference_mode()
def test_hoisted_full_grid_meets_native_gate(stream, heads):
    name = f"vsa256-hoisted-grid-h{heads}"
    fields, case = load_case(reference_root(), name)
    config = case["config"]
    assert (config["views"], config["columns"], config["real_tokens"]) == (
        32,
        16,
        798720,
    )
    assert config["padded_tokens"] == 917504
    assert config["keep"] == 217 and config["allowed_max"] == 434
    assert config["tile_shape"] == [1, 32, 8]
    assert config["order"] == "time_major"
    assert fields["valid"].tolist() == ([240] * 6 + [120]) * 512
    camera, column = fields["cell_camera"], fields["cell_column"]
    allowed_cells = (
        (camera[:, None] == camera[None])
        | (column[:, None] == column[None])
        | (camera[None] == 0)
    )
    expected = allowed_cells.repeat_interleave(7, 0).repeat_interleave(7, 1)
    torch.testing.assert_close(fields["allowed_tile"], expected, rtol=0, atol=0)
    assert fields["counts"].eq(217).all()
    for head in range(heads):
        assert expected.gather(1, fields["indices"][head].long()).all()
    _check_native_fidelity(
        stream,
        heads,
        tile=256,
        recipe=Recipe.FA4_128X128,
        name=name,
        captured=(fields, case),
    )
