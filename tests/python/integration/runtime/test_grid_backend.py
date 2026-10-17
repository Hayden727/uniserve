"""Worker launch settings select the real grid segment provider."""

from dataclasses import dataclass
from types import MappingProxyType

import pytest
import torch
from torch import nn

from tests.python.fixtures.launch import worker_args
from uniserve.model import ComponentEntry, Encoder, EntryPoint
from uniserve.nn.attention.grid import GridAttention, GridLayout
from uniserve_worker.execution.model_executor import ModelExecutor

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Config:
    views: int = 3
    columns: int = 3
    head_dim: int = 128


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GridAttention(1, config.head_dim)
        self.camera = torch.arange(config.views).repeat_interleave(
            config.columns
        )
        self.column = torch.arange(config.columns).repeat(config.views)

    def forward(self, values):
        layout = GridLayout.from_cells(
            self.camera, self.column, 1, device=values.device
        )
        return torch.stack(
            [self.attention(*value.unbind(1), layout) for value in values]
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.projection = Encoder(Network(self.config))


def entry_points(config):
    return MappingProxyType(
        {"projection": ComponentEntry("projection", (EntryPoint("encode"),))}
    )


@torch.inference_mode()
@pytest.mark.parametrize(
    ("backend", "device"),
    [
        ("auto", "cpu"),
        ("torch", "cpu"),
        pytest.param("flash_attn_4", "cuda:0", marks=pytest.mark.gpu),
        pytest.param("flash_attn_4_sm100", "cuda:0", marks=pytest.mark.gpu),
    ],
)
def test_launch_grid_provider_controls_public_encoder_execution(
    tmp_path, backend, device
):
    if device.startswith("cuda") and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability(0) != (10, 0)
    ):
        pytest.skip("the FA4 grid providers require SM100")
    launch = worker_args(
        tmp_path,
        device=device,
        attention_backend="torch",
        grid_attention_backend=backend,
        components={"projection": {"ranks": [0]}},
        deployment_components=["projection"],
    )
    runner = ModelExecutor(Model(), launch.execution)
    # Zero Q/K gives uniform attention over the row/column/reference union.
    values = torch.zeros((9, 3, 1, 128), dtype=torch.bfloat16, device=device)
    values[:, 2] = torch.arange(9, device=device).reshape(9, 1, 1)
    try:
        output = runner.run_encoder("conditioning", values).values[0]
        camera = torch.arange(3).repeat_interleave(3)
        column = torch.arange(3).repeat(3)
        expected = torch.tensor(
            [
                torch.arange(9)
                .float()[
                    (camera == camera[index])
                    | (column == column[index])
                    | (camera == 0)
                ]
                .mean()
                for index in range(9)
            ]
        )
        torch.testing.assert_close(
            output[:, 0, 0].float().cpu(), expected, rtol=0.01, atol=0.02
        )
        prepared = runner.prepare_module(
            "projection", (values.shape,), method="encode"
        )
        assert {
            record["provider"]
            for record in prepared.kernels()
            if record["op"] == "grid"
        } == {"torch" if backend == "auto" else backend}
    finally:
        runner.close()
