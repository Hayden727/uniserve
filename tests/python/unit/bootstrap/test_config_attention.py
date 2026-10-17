"""Numerical backend settings carried by a worker launch."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args
from uniserve_worker.errors import WorkerError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "backend", ["auto", "torch", "flash_attn_4", "flash_attn_4_sm100"]
)
def test_grid_attention_provider_survives_the_real_worker_launch_descriptor(
    tmp_path, backend
):
    assert worker_args(tmp_path).execution.grid_attention_backend == "auto"
    config = worker_args(tmp_path, grid_attention_backend=backend)
    assert config.execution.grid_attention_backend == backend


@pytest.mark.parametrize("backend", ["unknown", "", None, 7, True, [], {}])
def test_invalid_grid_attention_provider_is_rejected_at_launch(
    tmp_path, backend
):
    with pytest.raises(WorkerError, match="grid attention backend"):
        worker_args(tmp_path, grid_attention_backend=backend)


@pytest.mark.parametrize("backend", ["auto", "torch", "cublas", "flashinfer"])
def test_matmul_provider_survives_the_real_worker_launch_descriptor(
    tmp_path, backend
):
    assert worker_args(tmp_path).execution.matmul_backend == "auto"
    config = worker_args(tmp_path, matmul_backend=backend)
    assert config.execution.matmul_backend == backend


def test_unknown_matmul_provider_is_rejected_at_launch(tmp_path):
    with pytest.raises(WorkerError, match="matmul backend"):
        worker_args(tmp_path, matmul_backend="unknown")


def test_numerical_backend_settings_survive_the_launch_descriptor(tmp_path):
    config = worker_args(
        tmp_path,
        convolution_plans={"video_encoder": "/models/encoder-plans.json"},
        blas_workspace_size=0,
    )
    assert config.execution.convolution_plans == (
        ("video_encoder", "/models/encoder-plans.json"),
    )
    assert config.execution.blas_workspace_size == 0


@pytest.mark.parametrize("plans", [{"": "plan.json"}, {"video_encoder": ""}])
def test_convolution_plans_require_component_names_and_paths(tmp_path, plans):
    with pytest.raises(WorkerError, match="convolution plans require"):
        worker_args(tmp_path, convolution_plans=plans)


def test_negative_blas_workspace_is_rejected(tmp_path):
    with pytest.raises(WorkerError, match="BLAS workspace size"):
        worker_args(tmp_path, blas_workspace_size=-1)


def test_attention_overrides_survive_the_launch_descriptor(tmp_path):
    declarations = {
        "transformer.network.blocks.0.cam_self_attn.attn": "flash_attn_4"
    }
    config = worker_args(tmp_path, attention_overrides=declarations)
    assert dict(config.execution.attention_overrides) == declarations


@pytest.mark.parametrize(
    "declarations",
    [{"": "torch"}, {"camera..attn": "torch"}, {"camera.attn": "unknown"}],
)
def test_invalid_attention_overrides_are_rejected_at_launch(
    tmp_path, declarations
):
    with pytest.raises(WorkerError, match="attention overrides require"):
        worker_args(tmp_path, attention_overrides=declarations)


@pytest.mark.parametrize("path", ["missing", "projection"])
def test_attention_overrides_must_resolve_to_attention_at_worker_startup(path):
    from dataclasses import replace

    import torch

    from tests.python.fixtures.worker_config import cpu_worker_config
    from uniserve_worker.execution.model_executor import ModelExecutor

    model = torch.nn.ModuleDict({"projection": torch.nn.Linear(2, 2)})
    config = replace(
        cpu_worker_config(), attention_overrides=((path, "torch"),)
    )
    with pytest.raises(WorkerError, match="attention override"):
        ModelExecutor(model, config)


def test_the_flashinfer_workspace_size_reaches_the_attention_config(
    tmp_path,
):
    config = worker_args(tmp_path, flashinfer_workspace_size=64 << 20)

    assert config.execution.flashinfer.workspace_size == 64 << 20


@pytest.mark.parametrize("size", [0, -5])
def test_a_non_positive_flashinfer_workspace_is_a_usage_error(
    size, tmp_path, capsys
):
    # argparse reports a usage error with exit status 2.
    with pytest.raises(SystemExit) as exit_info:
        worker_args(tmp_path, flashinfer_workspace_size=size)

    assert exit_info.value.code == 2
    assert "workspace size must be positive" in capsys.readouterr().err
