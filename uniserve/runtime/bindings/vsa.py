"""Shared VSA numerical plans for a layer call site."""

from __future__ import annotations

import torch

from uniserve.tensors import BufferConfig

from ..backends import record_kernel_choice
from . import capturing
from .attention import ExchangeBuffers


class VsaBinding:
    """Borrow the context's VSA plans for one call site.

    Every operator is bound only to numerical dimensions and a block-count
    pattern. Serialized layers with the same signature can share its plans;
    the live block maps and projected rows remain inputs of each invocation.
    """

    def __init__(self, backend, transient, scratch, exchange, operators):
        # ``transient(role, requirements, device)`` lends the context's
        # shared per-call work areas (``ExecutionContext.scratch``);
        # ``scratch`` allocates the operators' own workspace.
        self.backend, self.transient, self.scratch = backend, transient, scratch
        self.exchange = exchange
        self.operators = operators
        # Provider name of each prepared operator key.
        self.providers = {}
        self.preparations = {}

    def prepare_tiles(self, module, q, k, v, gate, layout, rotary):
        """Borrow fused storage without retaining rotary or input tensors."""
        recipe = module.attention.recipe
        key = (
            q.device,
            q.dtype,
            q.shape[1:],
            layout.grid,
            layout.tile_shape,
            tuple(
                (tuple(value.shape), tuple(value.stride()))
                for value in (q, k, v, gate, *rotary, layout.valid_counts)
            ),
            recipe,
        )
        if key not in self.preparations:
            if capturing(q.device):
                raise RuntimeError("prepare VSA tile shapes before capture")
            from ..backends.attention import vsa

            self.preparations[key] = vsa.resolve(
                self.backend, device=q.device, tile=layout.tile, recipe=recipe
            )
            record_kernel_choice()
        return self.preparations[key].prepare_tiles(
            q, k, v, gate, layout, rotary, transient=self.transient
        )

    def prepare(self, pattern, q, *, recipe=None):
        key = (q.device, q.dtype, q.shape[1], q.shape[2], pattern, recipe)
        if key not in self.operators:
            if capturing(q.device):
                raise RuntimeError(
                    "prepare VSA numerical shapes before capture"
                )
            from ..backends.attention import vsa

            provider = vsa.resolve(
                self.backend, device=q.device, tile=pattern.tile, recipe=recipe
            )
            options = {
                "num_heads": q.shape[1],
                "head_dim": q.shape[2],
                "dtype": q.dtype,
            }
            requirements = provider.workspace_buffers(pattern, **options)
            operator = provider.prepare(
                pattern,
                **options,
                workspace=self.scratch(requirements, q.device),
                transient=self.transient,
                recipe=recipe,
            )
            self.operators[key] = (provider.name, operator)

        provider_name, operator = self.operators[key]
        if key not in self.providers:
            self.providers[key] = provider_name
            record_kernel_choice()

        return operator

    def prepare_exchange(self, module, layout, q):
        """Borrow one four-projection Ulysses payload and inverse output.

        The feature-packed payload uses the existing HeadExchange transport;
        its projection axis never participates in numerical arithmetic. Every
        serialized tile layer shares this storage through context scratch.
        """
        members = module.exchange.group.size
        rows = (layout.num_tokens + members - 1) // members * members
        unit = rows * module.local_heads * module.head_dim * q.element_size()
        requirements = {
            name: BufferConfig((unit * count,), torch.uint8)
            for name, count in (
                ("projection_input", 4),
                ("heads_send", 4),
                ("heads_receive", 4),
                ("output_receive", 1),
            )
        }
        return ExchangeBuffers(
            self.transient("vsa_tile_exchange", requirements, q.device)
        )

    def kernels(self):
        """Describe the block-sparse kernel of each prepared query shape.

        One record per prepared (device, dtype, heads, head dimension,
        pattern, recipe): provider, local heads, dimension, dtype and recipe.
        """
        return [
            {
                "op": "vsa",
                "provider": name,
                "heads": heads,
                "head_dim": head_dim,
                "dtype": str(dtype).removeprefix("torch."),
                "recipe": None if recipe is None else recipe.value,
            }
            for (
                _,
                dtype,
                heads,
                head_dim,
                _,
                recipe,
            ), name in self.providers.items()
        ] + [
            {
                "op": "vsa_prepare",
                "provider": provider.name,
                "recipe": "rope_pool_bt64",
                "heads": key[2][0],
                "head_dim": key[2][1],
            }
            for key, provider in self.preparations.items()
        ]

    def close(self):
        # The execution context retires shared operators once every call site
        # and graph has finished borrowing them.
        self.providers.clear()
        self.preparations.clear()
