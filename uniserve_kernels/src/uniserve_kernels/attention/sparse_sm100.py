"""SM100 sparse attention merging two raw accumulators inside one CTA.

The original exp2/reciprocal correction owns the sole BF16 rounding. Runtime
maps contain combined partial/full lists; the kernel distributes traversal
positions across its two streams. See FA4_LICENSE and SPARSE_SOURCES.json.
"""

import cutlass
import cutlass.cute as cute

from ._sparse_sm100 import utils
from ._sparse_sm100.block_sparsity import (
    BlockSparseTensorsTorch,
    fast_sampling,
    to_cute_block_sparse_tensors,
)
from ._sparse_sm100.flash_fwd import FlashAttentionForwardSm100
from ._sparse_sm100.tensor import to_cute_aux_tensor, to_cute_tensor
from .sparse import MAP_FIELDS, SparseLaunch


@fast_sampling
@cute.jit
def _valid_key(batch, head, query, key, seqlen, aux_tensors):
    """Apply the native per-element validity predicate to 64-row children."""
    block = utils.scalar_to_ssa(64, cutlass.Int32)
    zero = utils.scalar_to_ssa(0, cutlass.Int32)
    child = key // block
    valid = utils.scalar_to_ssa(aux_tensors[0][child[0]], cutlass.Int32)
    return (valid > zero) & (key % block < valid)


class _SparseKernel(FlashAttentionForwardSm100):
    def __init__(self, tile):
        super().__init__(
            128,
            128,
            m_block_size=128,
            n_block_size=64,
            q_stage=1,
            q_subtile_factor=tile // 128,
            mask_mod=_valid_key,
            has_aux_tensors=True,
            use_2cta_instrs=False,
        )


class SparseAttention(SparseLaunch):
    """One merged BF16 launch with borrowed maps and no global LSE buffer."""

    logical_tiles = (128, 256)
    key_tile = 64
    kernel_class = _SparseKernel

    def _arguments(self, q, k, v, out, buffers, scale, *, compiling):
        sparse = tuple(buffers[name] for name in MAP_FIELDS)
        valid = buffers["valid"]
        stream = ()
        if compiling:
            q, k, v, out = (to_cute_tensor(t) for t in (q, k, v, out))
            sparse = to_cute_block_sparse_tensors(
                BlockSparseTensorsTorch(*sparse)
            )
            valid = to_cute_aux_tensor(valid)
            stream = (
                cute.runtime.make_fake_stream(
                    use_tvm_ffi_env_stream=True,
                ),
            )
        # The stream is implicit in TVM FFI at invocation. Its compile-time
        # placeholder occupies the original kernel's seventh argument.
        return (
            q,
            k,
            v,
            out,
            None,
            scale,
            *stream,
            *([None] * 8),
            sparse,
            [valid],
        )
