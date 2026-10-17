"""SM100 block attention choosing its kernel by tile size and shape.

64-row complete calls use the native CUDA kernel for the large key domains it
was measured to serve and the CuTe kernel below that boundary. Logical 128/256
calls default to the native FA4 128x64 two-stream recipe. Explicit 128x128
uses two query stages and a hoisted validity mask with direct BF16 output.
Both preserve descending partial/full traversal.
"""

import torch
from uniserve_kernels.attention import vsa_cute, vsa_native

from uniserve.nn.attention.vsa.inputs import Recipe
from uniserve.tensors import BufferConfig

from . import Backend as BaseBackend
from . import Operator as BaseOperator
from .sparse import buffer_configs, split_blocks

# Logical tile sizes this provider addresses.
TILES = (64, 128, 256)


def available(device):
    """Report whether ``device`` runs both SM100 kernels."""
    return vsa_cute.available(device) and vsa_native.supported(device)


class _Operator(BaseOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_launches = {}

    def kernel(
        self, q, k, v, out, indices, counts, variable_block_sizes, *, scale
    ):
        tile = self.pattern.tile
        if tile == 64 and vsa_cute.should_use(
            rows=k.shape[0], prefix_tiles=self.pattern.dense_prefix_tiles
        ):
            vsa_cute.block_sparse_attention(
                q, k, v, out, indices, counts, variable_block_sizes, scale=scale
            )
            return
        if tile in (128, 256):
            self._sparse(
                q, k, v, out, indices, counts, variable_block_sizes, scale=scale
            )
            return
        vsa_native.block_sparse_attention(
            q,
            k,
            v,
            out,
            indices,
            counts,
            variable_block_sizes,
            scale=scale,
            block=tile,
        )

    def _sparse(self, q, k, v, out, indices, counts, valid_sizes, *, scale):
        from uniserve_kernels.attention.sparse_sm100 import SparseAttention

        recipe = self.recipe or Recipe.FA4_128X64
        launch_type = SparseAttention
        if recipe is Recipe.FA4_128X128:
            from uniserve_kernels.attention.sparse_hoisted_sm100 import (
                HoistedAttention,
            )

            launch_type = HoistedAttention
        heads, queries, selected = indices.shape
        requirements = buffer_configs(
            heads,
            queries,
            selected,
            valid_sizes.numel(),
            tile=self.pattern.tile,
            recipe=recipe,
        )
        for name, value in (("query", q), ("key", k), ("value", v)):
            requirements[name] = BufferConfig((1, *value.shape), value.dtype)
        requirements["output"] = BufferConfig((1, *q.shape), torch.bfloat16)
        if recipe is Recipe.FA4_128X128:
            # Preserve the hoisted kernel's existing output/LSE ABI.
            requirements["lse"] = BufferConfig(
                (1, heads, q.shape[0]), torch.float32
            )
        # Serialized layers borrow these buffers. Maps are rebuilt from live
        # inputs on every replay; compiled launches retain no tensor storage.
        buffers = (
            {
                name: torch.empty(
                    config.shape, dtype=config.dtype, device=indices.device
                )
                for name, config in requirements.items()
            }
            if self.transient is None
            else self.transient("vsa_sparse", requirements, indices.device)
        )
        split_blocks(
            indices,
            counts,
            valid_sizes,
            out=buffers,
            tile=self.pattern.tile,
            recipe=recipe,
        )
        for name, value in (("query", q), ("key", k), ("value", v)):
            buffers[name].copy_(value)
        projections = tuple(buffers[name] for name in ("query", "key", "value"))
        signature = (
            recipe,
            self.pattern.tile,
            scale,
            q.device,
            tuple(
                (name, value.dtype, tuple(value.shape), tuple(value.stride()))
                for name, value in buffers.items()
            ),
        )
        if signature not in self.sparse_launches:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare sparse FA4 shapes before capture")
            self.sparse_launches[signature] = launch_type(
                *projections,
                buffers["output"],
                buffers,
                tile=self.pattern.tile,
                scale=scale,
            )
        launch = self.sparse_launches[signature]
        launch(*projections, buffers["output"], buffers, scale=scale)
        out.copy_(buffers["output"][0])

    def close(self):
        self.sparse_launches.clear()
        super().close()


class Backend(BaseBackend):
    name = "sm100"
    operator_class = _Operator
    recipes = (Recipe.FA4_128X64, Recipe.FA4_128X128)

    def prepare_tiles(self, q, k, v, gate, layout, rotary, *, transient=None):
        """Borrow destinations for the BT64 stored-BF16 pooling recipe."""
        from uniserve_kernels.attention.tile_rope_pool import tile_rope_pool

        inputs = dict(
            zip(
                ("q", "k", "v", "gate", "cos", "sin"),
                (q, k, v, gate, *rotary),
                strict=True,
            )
        )
        inputs["counts"] = layout.valid_counts
        requirements = {
            "source": BufferConfig((layout.padded_tokens,), torch.int64),
            **{
                f"tile_{name}": BufferConfig(
                    (layout.padded_tokens, *q.shape[1:]), q.dtype
                )
                for name in ("q", "k", "v", "gate")
            },
            **{
                f"pool_{name}": BufferConfig(
                    (q.shape[1], layout.num_tiles, q.shape[2]), q.dtype
                )
                for name in ("q", "k", "v")
            },
            **{
                f"input_{name}": BufferConfig(tuple(value.shape), value.dtype)
                for name, value in inputs.items()
                if not value.is_contiguous()
            },
        }
        buffers = (
            {
                name: torch.empty(
                    config.shape, dtype=config.dtype, device=q.device
                )
                for name, config in requirements.items()
            }
            if transient is None
            else transient("vsa_tile_prepare", requirements, q.device)
        )
        for name, value in tuple(inputs.items()):
            if not value.is_contiguous():
                inputs[name] = buffers[f"input_{name}"]
                inputs[name].copy_(value)
        source = buffers["source"]
        source.fill_(-1)
        source[layout.slots] = layout.permutation
        tiled = tuple(
            buffers[f"tile_{name}"] for name in ("q", "k", "v", "gate")
        )
        pooled = tuple(buffers[f"pool_{name}"] for name in ("q", "k", "v"))
        tile_rope_pool(
            *(inputs[name] for name in ("q", "k", "v", "gate", "cos", "sin")),
            source,
            inputs["counts"],
            tiled=tiled,
            pooled=pooled,
        )
        return (*tiled, *(value.transpose(0, 1) for value in pooled))
