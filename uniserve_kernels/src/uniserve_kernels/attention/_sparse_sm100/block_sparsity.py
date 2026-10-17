"""
Block-sparsity utilities for FlexAttention
"""

from typing import Callable, NamedTuple, Tuple

import cutlass.cute as cute
import torch

from .tensor import to_cute_tensor


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None
    full_block_idx: cute.Tensor | None

    def __new_from_mlir_values__(self, values):
        if len(values) == 2:
            values = (*values, None, None)
        return BlockSparseTensors(*values)


class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None


def is_block_sparsity_enabled(tensors: BlockSparseTensorsTorch) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch, enable_tvm_ffi: bool = True
) -> BlockSparseTensors | None:
    """Convert torch block sparsity tensors to CuTe tensors, optionally for tvm ffi"""
    if not is_block_sparsity_enabled(tensors):
        return None
    (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        *_,
    ) = tensors

    (
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
    ) = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        for t in (mask_block_cnt, mask_block_idx)
    ]
    (
        full_block_cnt_tensor,
        full_block_idx_tensor,
    ) = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (full_block_cnt, full_block_idx)
    ]

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
    )


def fast_sampling(mask_mod):
    """Convenience decorator to mark mask_mod as safe for 5-point fast sampling"""
    mask_mod.use_fast_sampling = True
    return mask_mod
