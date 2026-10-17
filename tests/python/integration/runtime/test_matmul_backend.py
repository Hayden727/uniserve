"""Worker launch settings select real numerical providers per context."""

from dataclasses import dataclass
from types import MappingProxyType

import pytest
import torch
from torch import nn

from tests.python.fixtures.launch import worker_args
from uniserve.model import ComponentEntry, Encoder, EntryPoint
from uniserve.nn.linear import Linear
from uniserve_worker.execution.model_executor import ModelExecutor

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Config:
    input_width: int = 4
    output_width: int = 2


class Model(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.projection = Encoder(
            Linear(
                config.input_width,
                config.output_width,
                bias=False,
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            self.projection.network.weight.copy_(
                torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
            )


def entry_points(config):
    return MappingProxyType(
        {"projection": ComponentEntry("projection", (EntryPoint("encode"),))}
    )


@pytest.mark.parametrize("backend", ["auto", "torch", "cublas"])
def test_launch_matmul_provider_controls_public_encoder_execution(
    tmp_path, backend
):
    launch = worker_args(
        tmp_path,
        matmul_backend=backend,
        components={"projection": {"ranks": [0]}},
        deployment_components=["projection"],
    )
    runner = ModelExecutor(Model(), launch.execution)
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    try:
        if backend == "cublas":
            # This backend's public operand contract requires CUDA. A CPU
            # success would mean the worker silently ignored the descriptor.
            with pytest.raises(
                ValueError, match="cuBLAS requires CUDA operands"
            ):
                runner.run_encoder("conditioning", values)
        else:
            output = runner.run_encoder("conditioning", values)
            torch.testing.assert_close(
                output.values[0], torch.tensor([[30.0, 20.0]]), rtol=0, atol=0
            )
    finally:
        runner.close()
