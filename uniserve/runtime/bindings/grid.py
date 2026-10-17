"""Runtime-owned grid providers and borrowed serialized numerical work areas."""

from __future__ import annotations

import torch

from uniserve.tensors import BufferConfig

from ..backends import record_kernel_choice
from ..backends.attention.grid import resolve
from . import capturing
from .attention import ExchangeBuffers


class GridBinding:
    """Bind one call site to shared scratch without retaining grid indices.

    The caller owns each immutable GridLayout. This binding retains only
    numerical shape signatures; every invocation supplies its current indices.
    All layers sharing the context scratch execute serially on its stream.
    """

    def __init__(self, module, backend, scratch):
        self.module, self.backend, self.scratch = module, backend, scratch
        self.operators = {}
        self.prepared = set()

    def prepare(self, layout, q):
        module = self.module
        key = (
            q.device,
            layout.views,
            layout.columns,
            layout.cell_tokens,
            layout.canonical is not None,
            module.local_heads,
            module.head_dim,
        )
        if key not in self.prepared:
            if capturing(q.device):
                raise RuntimeError(
                    "prepare grid numerical shapes before capture"
                )
            if q.device not in self.operators:
                self.operators[q.device] = resolve(self.backend, q.device)
            self.prepared.add(key)
            record_kernel_choice()
        buffers = self.scratch(
            "grid",
            layout.buffer_configs(module.local_heads, module.head_dim),
            q.device,
        )
        storage = None
        group = module.exchange.group
        if group.size > 1:
            rows = (
                (layout.num_tokens + group.size - 1) // group.size * group.size
            )
            # Match HeadExchange's one packed Q/K/V payload; this is shared
            # transient storage, not a separate full-grid backing per layer.
            requirements = {
                name: BufferConfig(
                    (rows * heads * module.head_dim * q.element_size(),),
                    torch.uint8,
                )
                for name, heads in (
                    ("heads_send", 3 * module.local_heads),
                    ("heads_receive", 3 * module.local_heads),
                    ("output_receive", module.local_heads),
                )
            }
            storage = ExchangeBuffers(
                self.scratch("grid_exchange", requirements, q.device)
            )
        return self.operators[q.device], buffers, storage

    def kernels(self):
        return [
            {
                "op": "grid",
                "provider": operator.name,
                "heads": self.module.local_heads,
                "head_dim": self.module.head_dim,
                "dtype": "bfloat16",
            }
            for operator in self.operators.values()
        ]

    def close(self):
        self.operators.clear()
        self.prepared.clear()
