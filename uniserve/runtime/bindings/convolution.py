"""Numerical convolution recipes and context-owned cuDNN plan resources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from uniserve.nn.convolution import conv3d_unfold
from uniserve.tensors import BufferConfig

from ..backends import record_kernel_choice
from . import capturing


def _strides(shape, strides):
    # Singleton strides do not change addressed elements. Torch canonicalizes
    # them when constructing cuDNN descriptors, including 1x1x1 filters.
    return tuple(
        0 if n == 1 else s for n, s in zip(shape, strides, strict=True)
    )


@dataclass(frozen=True, slots=True)
class ConvPlan:
    """An executed convolution's layouts, engine and immutable tuning knobs."""

    input_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    weight_shape: tuple[int, ...]
    weight_stride: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_stride: tuple[int, ...]
    dtype: torch.dtype
    padding: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    engine: int
    knobs: tuple[tuple[str, int], ...]

    @property
    def signature(self):
        return (
            self.input_shape,
            _strides(self.input_shape, self.input_stride),
            self.weight_shape,
            _strides(self.weight_shape, self.weight_stride),
            self.dtype,
            self.padding,
            self.stride,
            self.dilation,
        )

    @classmethod
    def read(cls, path: str | Path) -> tuple[ConvPlan, ...]:
        """Read plan records emitted by the native convolution profiler."""
        records = json.loads(Path(path).read_text())["plans"]
        dtypes = {
            "BFLOAT16": torch.bfloat16,
            "FLOAT": torch.float32,
            "HALF": torch.float16,
        }
        result = []
        for record in records:
            if len({record[name]["dtype"] for name in ("x", "w", "y")}) != 1:
                raise ValueError("convolution plans require one I/O dtype")
            result.append(
                cls(
                    tuple(record["x"]["shape"]),
                    tuple(record["x"]["stride"]),
                    tuple(record["w"]["shape"]),
                    tuple(record["w"]["stride"]),
                    tuple(record["y"]["shape"]),
                    tuple(record["y"]["stride"]),
                    dtypes[record["x"]["dtype"]],
                    tuple(record["padding"]),
                    tuple(record["stride"]),
                    tuple(record["dilation"]),
                    record["engine"],
                    tuple(sorted(record["knobs"].items())),
                )
            )
        if len({plan.signature for plan in result}) != len(result):
            raise ValueError("convolution plan signatures must be unique")
        return tuple(result)


class ConvolutionBinding:
    """Specialize a numerical convolution without retaining request tensors.

    An explicit module recipe takes precedence over the cuDNN table. For
    automatic modules, a configured table is strict: an unrecorded shape is
    an error rather than an implicit change of summation algorithm. Eager
    warmup builds plans and allocates workspace before CUDA capture. The
    context retires its handle after every borrowing graph has retired.
    """

    def __init__(self, module, plans: tuple[ConvPlan, ...] | None, allocate):
        if module.groups != 1 or module.padding_mode != "zeros":
            raise ValueError(
                "explicit convolution plans support ungrouped zero padding"
            )
        self.padding, self.stride, self.dilation = (
            module.padding,
            module.stride,
            module.dilation,
        )
        self.backend = getattr(module, "backend", "auto")
        self.plans = {plan.signature: plan for plan in plans or ()}
        self.allocate = allocate
        self.operators = {}
        self.handle = None
        self.used = False

    def _prepare(self, x, weight):
        key = (
            tuple(x.shape),
            _strides(x.shape, x.stride()),
            tuple(weight.shape),
            _strides(weight.shape, weight.stride()),
            x.dtype,
            self.padding,
            self.stride,
            self.dilation,
        )
        if key in self.operators:
            return self.operators[key]
        if (
            not x.is_cuda
            or x.dtype != weight.dtype
            or x.device != weight.device
        ):
            raise ValueError(
                "explicit cuDNN operands require one CUDA device and dtype"
            )
        if capturing(x.device):
            raise RuntimeError(
                "prepare convolution layouts before CUDA capture"
            )
        plan = self.plans.get(key)
        if plan is None:
            # TensorIterator can change a preceding elementwise operation's
            # chosen memory order across torch versions. A unique recorded
            # plan still declares the operator's required packing explicitly.
            matches = [
                value
                for value in self.plans.values()
                if value.input_shape == tuple(x.shape)
                and value.weight_shape == tuple(weight.shape)
                and value.dtype == x.dtype
                and (value.padding, value.stride, value.dilation)
                == (self.padding, self.stride, self.dilation)
            ]
            if len(matches) != 1:
                raise ValueError(f"no unambiguous convolution plan for {key}")
            plan = matches[0]
        import cudnn

        if self.handle is None:
            self.handle = cudnn.create_handle()
        graph = cudnn.pygraph(
            io_data_type={
                torch.bfloat16: cudnn.data_type.BFLOAT16,
                torch.float16: cudnn.data_type.HALF,
                torch.float32: cudnn.data_type.FLOAT,
            }[x.dtype],
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        tx = graph.tensor_like(
            x.as_strided(plan.input_shape, plan.input_stride)
        )
        tw = graph.tensor_like(
            weight.as_strided(plan.weight_shape, plan.weight_stride)
        )
        ty = graph.conv_fprop(
            image=tx,
            weight=tw,
            padding=list(plan.padding),
            stride=list(plan.stride),
            dilation=list(plan.dilation),
        )
        ty.set_output(True).set_dim(list(plan.output_shape)).set_stride(
            list(plan.output_stride)
        )
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plan(
            plan.engine,
            {
                getattr(cudnn.knob_type, name): value
                for name, value in plan.knobs
            },
        )
        graph.check_support()
        graph.build_plans()
        workspace = self.allocate(
            {
                "workspace": BufferConfig(
                    (graph.get_workspace_size(),), torch.uint8
                )
            },
            x.device,
        )["workspace"]
        result = (plan, graph, tx, tw, ty, workspace)
        self.operators[key] = result
        record_kernel_choice()
        return result

    def __call__(self, x, weight, bias):
        if self.backend == "torch_unfold":
            if not self.used:
                record_kernel_choice()
                self.used = True
            return conv3d_unfold(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        import cudnn

        plan, graph, tx, tw, ty, workspace = self._prepare(x, weight)
        if _strides(x.shape, x.stride()) != _strides(
            plan.input_shape, plan.input_stride
        ):
            x = torch.empty_strided(
                plan.input_shape,
                plan.input_stride,
                device=x.device,
                dtype=x.dtype,
            ).copy_(x)
        if _strides(weight.shape, weight.stride()) != _strides(
            plan.weight_shape, plan.weight_stride
        ):
            weight = torch.empty_strided(
                plan.weight_shape,
                plan.weight_stride,
                device=weight.device,
                dtype=weight.dtype,
            ).copy_(weight)
        output = torch.empty_strided(
            plan.output_shape,
            plan.output_stride,
            device=x.device,
            dtype=x.dtype,
        )
        cudnn.set_stream(
            self.handle, torch.cuda.current_stream(x.device).cuda_stream
        )
        graph.execute(
            {tx: x, tw: weight, ty: output}, workspace, handle=self.handle
        )
        # Torch's cuDNN dispatch rounds before a separate bias addition.
        # Fusing bias into the engine would change this contract.
        if bias is not None:
            output = output + bias.view(1, -1, *((1,) * (x.ndim - 2)))
        return output

    def kernels(self):
        if self.backend == "torch_unfold":
            return (
                [{"op": "convolution", "provider": "torch_unfold"}]
                if self.used
                else []
            )
        return [
            {
                "op": "convolution",
                "provider": "cudnn",
                "engine": plan.engine,
                "knobs": dict(plan.knobs),
                "input": plan.input_shape,
            }
            for plan, *_ in self.operators.values()
        ]

    def close(self):
        self.operators.clear()
        if self.handle is not None:
            import cudnn

            cudnn.destroy_handle(self.handle)
            self.handle = None
