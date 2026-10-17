"""FA4 128x128 sparse attention with two query stages and hoisted VBS.

The masked score predicate follows the native fragment-uniform VBS applier:
one validity load per accumulator row fragment, then column comparisons.
Upstream FA4 owns MMA and scheduling; the shared softmax method preserves
masked hardware exp2 and unmasked emulation frequency 16. See FA4_LICENSE.
"""

# Preserve upstream CuTe fragment names to keep the mask port auditable.
# ruff: noqa: N803, N806

from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    fast_sampling,
    to_cute_block_sparse_tensors,
)
from flash_attn.cute.cute_dsl_utils import to_cute_aux_tensor, to_cute_tensor
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.utils import AuxData, scalar_to_ssa

from .dense_sm100 import _DenseKernel
from .sparse import MAP_FIELDS, SparseLaunch


@fast_sampling
@cute.jit
def _valid_key(batch, head, query, key, seqlen, aux_tensors):
    """Semantic VBS predicate; the forward applier hoists its scalar load."""
    block = scalar_to_ssa(128, Int32)
    zero = scalar_to_ssa(0, Int32)
    child = key // block
    valid = scalar_to_ssa(aux_tensors[0][child[0]], Int32)
    return (valid > zero) & (key % block < valid)


class _HoistedMask(AttentionMask):
    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        vec_size: cutlass.Constexpr[int] = 1,
        check_q_boundary: bool = False,
        r2p: bool = True,
        rBitmask: cute.Tensor | None = None,
    ) -> None:
        if const_expr(mask_mod is None):
            AttentionMask.apply_mask_sm100(
                self,
                acc_S,
                m_block,
                n_block,
                thr_mma,
                thr_tmem_load,
                mask_seqlen,
                mask_causal,
                mask_local,
                mask_mod,
                batch_idx,
                head_idx,
                aux_data,
                fastdiv_mods,
                head_divmod,
                vec_size,
                check_q_boundary,
                r2p,
                rBitmask,
            )
        else:
            assert not mask_causal and not mask_local
            assert self.tile_m == 128 and self.tile_n == 128
            assert self.qhead_per_kvhead_packgqa == 1 and not self.swap_AB
            assert rBitmask is None
            coordinates = cute.make_identity_tensor((self.tile_m, self.tile_n))
            partition = thr_mma.partition_C(coordinates)[(None, None), 0, 0]
            fragment = thr_tmem_load.partition_D(partition)
            if n_block < 0:
                n_block = 0
            # Every thread owns one accumulator row spanning one KV block.
            # This scalar load sits outside the per-score masking loop.
            valid = Int32(aux_data.tensors[0][n_block])
            for index in cutlass.range_constexpr(cute.size(fragment.shape)):
                column = fragment[index][1]
                acc_S[index] = acc_S[index] if column < valid else -Float32.inf
                if const_expr(mask_seqlen):
                    absolute_column = column + n_block * self.tile_n
                    acc_S[index] = (
                        -Float32.inf
                        if absolute_column >= self.seqlen_k
                        else acc_S[index]
                    )
                if check_q_boundary:
                    row = fragment[index][0] + m_block * self.tile_m
                    acc_S[index] = (
                        -Float32.inf if row >= self.seqlen_q else acc_S[index]
                    )


class _HoistedKernel(_DenseKernel):
    def __init__(self, tile):
        if tile != 256:
            raise ValueError("two query stages require logical 256-row tiles")
        FlashAttentionForwardSm100.__init__(
            self,
            128,
            128,
            m_block_size=128,
            n_block_size=128,
            q_stage=2,
            q_subtile_factor=1,
            mask_mod=_valid_key,
            has_aux_tensors=True,
            use_2cta_instrs=False,
        )
        self._tune = {
            **self._tune,
            "ex2_emu_freq": 16,
            "ex2_emu_start_frg": 1,
        }

    def _generate_attention_mask_cls(self, window_size_left, window_size_right):
        return partial(
            _HoistedMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=1,
        )


class HoistedAttention(SparseLaunch):
    """One direct sparse forward; BF16 output has no partial-output merge."""

    logical_tiles = (256,)
    key_tile = 128
    needs_lse = True
    kernel_class = _HoistedKernel

    def _arguments(self, q, k, v, out, buffers, scale, *, compiling):
        sparse = BlockSparseTensorsTorch(
            *(buffers[name] for name in MAP_FIELDS)
        )
        valid, lse = buffers["valid"], buffers["lse"]
        stream = ()
        if compiling:
            q, k, v, out = (to_cute_tensor(t) for t in (q, k, v, out))
            lse = to_cute_tensor(lse, assumed_align=4)
            sparse = to_cute_block_sparse_tensors(sparse)
            valid = to_cute_aux_tensor(valid)
            stream = (
                cute.runtime.make_fake_stream(
                    use_tvm_ffi_env_stream=True,
                ),
            )
        else:
            sparse = (*sparse[:4], None, None, None, None)
        return (
            q,
            k,
            v,
            out,
            lse,
            scale,
            *([None] * 9),
            sparse,
            AuxData([valid]),
            *([None] * 7),
            self.query_length,
            *stream,
        )
