"""
Block-sparse runtime utilities for CUTE DSL kernels.

This module contains runtime execution functions for block-sparse attention kernels.
These utilities are used by CUTE DSL kernels to produce and consume block-sparse loads.
"""

from typing import Callable, Optional
from functools import partial
import math
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import copy_utils

# Import data structures from block_sparsity
from .block_sparsity import BlockSparseTensors


# NOTE [SM100 block-sparse empty tiles: mbarrier contract]
#
# For block-sparse SM100 forward, a given (m_block, stage) Q tile can have zero active
# KV blocks (total_block_cnt == 0). In that case there is no seqlen_kv iteration, so
# the softmax warp-group has no row stats to publish.
#
# The correction warp-group seeds fully-masked-row stats and runs the usual correction
# epilogue so output/LSE have well-defined values. Both warp-groups must still perform
# the softmax<->correction mbarrier handshake so phases advance correctly across
# empty->empty and empty->non-empty tile sequences.
#
# In the no-sink case, this corresponds to the usual fully-masked-row convention:
# output is zero and LSE is -inf.
#
# Barrier contract (each is `mbar_ptr + <offset> + stage`):
#
# Producer/consumer pairs:
# - `mbar_softmax_corr_full`    : softmax arrive        -> correction wait
# - `mbar_softmax_corr_empty`   : correction arrive     -> softmax wait
# - `mbar_P_full_O_rescaled`    : softmax arrive (+ correction arrive) -> MMA wait
# - `mbar_P_full_2`             : softmax arrive        -> MMA wait
# - `mbar_corr_epi_full_/empty` : correction <-> epilogue (only when epilogue is separate)
#
# Empty tile (`total_block_cnt == 0`):
# - Softmax: skips the seqlen_kv softmax path entirely (no P stores, no `mbar_P_full_*`).
#   It only arrives `mbar_softmax_corr_full` once per stage as a synthetic "no work" signal.
#   At the `softmax_loop` level, softmax unconditionally waits `mbar_softmax_corr_empty`
#   before each tile (when block-sparse) to drain a prior correction arrival and keep
#   phases aligned across non-empty -> empty transitions.
# - Correction: waits `mbar_softmax_corr_full`, seeds stats + runs `correction_epilogue(scale=0)`,
#   and arrives `mbar_softmax_corr_empty` (and `mbar_corr_epi_full_/empty` when applicable).
# - No `mbar_P_full_*` barriers are arrived (no P, no MMA O); only the softmax<->correction
#   (and correction<->epilogue) handshakes advance phases.
#
# Non-empty tile:
# - Softmax: runs `softmax_step` (produces P) and uses `mbar_softmax_corr_full/empty` to
#   publish row_max (during seqlen_kv) and final row stats (once per tile), and to advance phases;
#   arrives `mbar_P_full_*` when P is stored.
# - Correction: waits `mbar_softmax_corr_full`, may rescale/release O, arrives `mbar_softmax_corr_empty`
#   to ack/advance, and arrives `mbar_P_full_O_rescaled` when MMA can proceed.
#
# Backward (SM100):
# - Empty KV tile: for a given `n_block`, `total_m_block_cnt == 0` means no Q tiles contribute.
# - Both the load and compute loops guard all pipeline work on `process_tile`, so empty tiles
#   skip producer/consumer operations entirely (no per-tile mbarrier phase handshake like forward).
# - In the `not dKV_postprocess` path, dK/dV for empty KV tiles are explicitly written as zeros
#   even when `process_tile == False` (see `flash_bwd_sm100.py` `should_zero_dKV`).


@cute.jit
def finish_overlap_v_load(
    block_indices: cute.Tensor,
    block_count,
    load_V,
    pipeline_v,
    kv_producer_state,
):
    """Load the final V block after overlapped K/V loads."""
    if block_count > 0:
        n_block_last = block_indices[0]
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block_last, producer_state=kv_producer_state)
        kv_producer_state.advance()

    return kv_producer_state


@cute.jit
def sparse_tensor_m_block(
    m_block,
    qhead_per_kvhead: cutlass.Constexpr[int],
    q_subtile_factor: cutlass.Constexpr[int],
):
    """Map packed m_block indices to block-sparse tensor indices."""
    block = m_block
    if const_expr(qhead_per_kvhead != 1):
        block = block // qhead_per_kvhead
    if const_expr(q_subtile_factor != 1):
        block = block // q_subtile_factor
    return block


@cute.jit
def load_block_list_sm100(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    q_stage: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
):
    """SM100 version of load_block_list (no intra_wg_overlap, no extra_tx_count)."""
    if block_count > 0:
        # First iteration: load Q alongside K if requested
        n_block_first = block_indices[block_count - 1]

        if const_expr(load_q_with_first):
            # SM100 loads Q0 and optionally Q1
            load_Q(block=0, stage=0)
            if const_expr(q_stage == 2):
                load_Q(block=1, stage=1)

        # SM100 doesn't use producer_acquire for pipeline_kv in load path
        # The pipeline barriers are handled inside load_KV
        load_K(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()
        load_V(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()

        # Remaining blocks
        for offset in cutlass.range(1, block_count):
            n_block = block_indices[block_count - 1 - offset]
            load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()

    return kv_producer_state


# SM100-specific tile processor using SM100 helpers
@cute.jit
def produce_block_sparse_loads_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    """SM100 entry point for sparse block iteration.

    SM100 uses PipelineTmaUmma which doesn't support extra_tx_count, so we use
    simplified block processing that just calls producer_acquire without extras.

    Args:
        m_block: which tile of m we are processing
        qhead_per_kvhead: Constexpr pack factor
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    q_phase_flipped = False

    if mask_empty:
        # No masked blocks: process full list with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_full_block_idx,
            curr_full_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = not full_empty
    else:
        # Process masked blocks with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = True

        if not full_empty:
            # Process full blocks without Q loading
            kv_producer_state = load_block_list_sm100(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_q_with_first=False,
                q_stage=q_stage,
                kv_producer_state=kv_producer_state,
                load_Q=load_Q,
                load_K=load_K,
                load_V=load_V,
                pipeline_kv=pipeline_kv,
            )

    if q_phase_flipped:
        q_producer_phase ^= 1

    return kv_producer_state, q_producer_phase


@cute.jit
def get_total_block_count(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors
    if const_expr(full_block_cnt is not None):
        return (
            mask_block_cnt[batch_idx, head_idx, m_block_sparse]
            + full_block_cnt[batch_idx, head_idx, m_block_sparse]
        )
    else:
        return mask_block_cnt[batch_idx, head_idx, m_block_sparse]


@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    qhead_per_kvhead,
    pack_gqa: cutlass.Constexpr,
    is_split_kv: cutlass.Constexpr,
    learnable_sink,
    mLSE,
    seqlen,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    split_idx: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.ThrMma,
    tOtO: cute.Tensor,
    sO: cute.Tensor,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    pipeline_o_epi: cutlass.pipeline.PipelineAsync,
    sm_stats_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    softmax_scale_log2: Float32,
    mO_cur: Optional[cute.Tensor] = None,
    gO: Optional[cute.Tensor] = None,
    gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
):
    """Handle SM100 forward block-sparse tiles with no active KV blocks.

    This path is taken when `total_block_cnt == 0`. The softmax warp-group still
    arrives `mbar_softmax_corr_full` (synthetic "no work") so the correction
    warp-group can:

    - seed fully-masked-row stats (row_sum=1; row_max=-inf when tracked) for LSE
    - run `correction_epilogue` with `scale=0` so the output tile is written as zeros
      (independent of any prior tmem contents)
    - wait on `mbar_softmax_corr_full` and arrive `mbar_softmax_corr_empty`
      (and `mbar_corr_epi_*` when applicable) so phases stay aligned across tiles

    This helper intentionally does not touch `mbar_P_full_*` since no P is produced.
    See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
    """
    LOG2_E = Float32(math.log2(math.e))
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

    for stage in cutlass.range_constexpr(q_stage):
        row_sum_value = Float32(1.0)
        row_max_value = (
            -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None
        )
        if const_expr(learnable_sink is not None):
            sink_val = -Float32.inf
            if const_expr(not pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            elif tidx < m_block_size:
                q_head_idx = (
                    (q_stage * m_block + stage) * m_block_size + tidx
                ) % qhead_per_kvhead + head_idx * qhead_per_kvhead
                sink_val = Float32(learnable_sink[q_head_idx])
            if sink_val != -Float32.inf and (const_expr(not is_split_kv) or split_idx == 0):
                if row_max_value == -Float32.inf:
                    row_max_value = sink_val * (LOG2_E / softmax_scale_log2)
                    row_sum_value = Float32(1.0)
                else:
                    row_sum_value = row_sum_value + cute.math.exp2(
                        sink_val * LOG2_E - row_max_value * softmax_scale_log2, fastmath=True
                    )
        if tidx < m_block_size:
            scale_row_idx = tidx + stage * m_block_size
            sScale[scale_row_idx] = row_sum_value
            if const_expr(mLSE is not None or learnable_sink is not None):
                sScale[scale_row_idx + q_stage * m_block_size] = row_max_value
        acc_flag = row_sum_value == Float32(0.0) or row_sum_value != row_sum_value
        stats[stage] = (row_sum_value, row_max_value, acc_flag)

        # See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
        pipeline_sm_stats.consumer_release_w_index(stage)

        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
        correction_epilogue(
            thr_mma_pv,
            tOtO[None, None, None, stage],
            tidx,
            stage,
            m_block,
            seqlen.seqlen_q,
            Float32(0.0),  # zero scale ensures empty tile writes zeros into staged outputs
            sO[None, None, stage],
            mO_cur,
            gO[None, None, stage],
            gmem_tiled_copy_O,
        )
        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_commit_w_index(stage)

    sm_stats_consumer_phase ^= 1
    corr_epi_producer_phase ^= 1

    return (
        sm_stats_consumer_phase,
        o_corr_consumer_phase,
        corr_epi_producer_phase,
    )


# traversal for the intra-CTA KV split (flash_fwd_sm100.kv_split)
@cute.jit
def split_block_counts(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """``(total, n_stream0, n_stream1)`` for a two-way KV split of one tile's block list.

    Stream ``s`` owns the blocks at global index ``s, s+2, ...`` of the concatenated
    ``[mask..., full...]`` list, so stream 0 gets the extra block when the count is odd.
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
    mask_block_cnt, _, full_block_cnt, _ = blocksparse_tensors
    total = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    if const_expr(full_block_cnt is not None):
        total = total + full_block_cnt[batch_idx, head_idx, m_block_sparse]
    return total, (total + 1) // 2, total // 2


@cute.jit
def softmax_block_sparse_split_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    num_stage_warps: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """One KV stream's half of the block list, in the same global order the MMA issues it.

    Stream ``s`` takes global indices ``s, s+2, ...``. The list is physically two segments
    (``mask`` then ``full``) and the two are masked differently, so the parity is applied
    per segment: within ``mask`` the stream's local indices start at ``s``, and within
    ``full`` they start at ``(s - mask_cnt) mod 2``. That keeps ``mask_fn`` / ``is_first``
    constexpr -- selecting between ``mask_fn`` and ``mask_fn_none`` per iteration would
    have to trace the whole of ``softmax_step`` twice.

    ``mask_seqlen=True`` goes to the FIRST block of each stream, where stock gives it to
    global index 0 alone. It has to: ``stage_idx`` comes from the warp id and is dynamic,
    while ``mask_seqlen`` is consumed by a ``const_expr`` inside ``apply_mask_sm100``. The
    widening is safe rather than merely convenient -- the flag masks columns at or past
    ``seqlen_k``, so on any block that does not straddle the end of the sequence it masks
    nothing at all. Every other ``(n_block, mask_fn)`` pairing is exactly the stock one.
    """
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % num_stage_warps
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors
    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]
    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    # How many of each segment this stream owns.
    n_mask = (curr_mask_block_cnt - stage_idx + 1) // 2
    if curr_mask_block_cnt < stage_idx:
        n_mask = Int32(0)
    # (a - b) % 2 == (a + b) % 2, and the sum form keeps both operands non-negative.
    # `stage_idx - curr_mask_block_cnt` goes negative whenever there is at least one mask
    # block, and the DSL lowers `%` to a truncating remainder (C semantics, sign follows
    # the dividend) rather than Python's floored one -- so the difference form yields
    # full_start = -1, which over-counts n_full and indexes past the end of the full list.
    # Only shapes with mask blocks hit it, which is why an all-full shape ran clean.
    full_start = (stage_idx + curr_mask_block_cnt) % 2
    n_full = (curr_full_block_cnt - full_start + 1) // 2
    if curr_full_block_cnt < full_start:
        n_full = Int32(0)
    my_count = n_mask + n_full

    if my_count == 0:
        # Same synthetic "no work" signal an empty tile uses; see the NOTE at the top of
        # this file. Reached whenever the tile has fewer blocks than streams (total == 1).
        sm_stats_barrier.arrive_w_index(index=stage_idx * num_stage_warps + warp_idx)
    else:
        if n_mask > 0:
            n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - stage_idx]
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                n_block,
                is_first=True,
                mask_fn=partial(
                    mask_fn,
                    mask_seqlen=True,
                    check_q_boundary=check_m_boundary,
                ),
            )
            for i in cutlass.range(1, n_mask):
                n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - stage_idx - 2 * i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    mask_fn=partial(
                        mask_fn, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )

        if n_full > 0:
            n_block = curr_full_block_idx[curr_full_block_cnt - 1 - full_start]
            if n_mask == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    is_first=True,
                    mask_fn=partial(
                        mask_fn_none,
                        mask_seqlen=True,
                        check_q_boundary=check_m_boundary,
                    ),
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    is_first=False,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )
            for i in cutlass.range(1, n_full):
                n_block = curr_full_block_idx[curr_full_block_cnt - 1 - full_start - 2 * i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        my_count == 0,
    )


@cute.jit
def softmax_block_sparse_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block_sparse, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block_sparse]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block_sparse, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    total_block_cnt = curr_mask_block_cnt + curr_full_block_cnt

    if total_block_cnt == 0:
        # See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
        # pipeline_sm_stats.producer_commit_w_index(stage_idx)
        sm_stats_barrier.arrive_w_index(index=stage_idx * 4 + warp_idx)
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                mask_n_block,
                is_first=True,
                mask_fn=partial(mask_fn, mask_seqlen=True, check_q_boundary=check_m_boundary),
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mask_n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False, check_q_boundary=check_m_boundary),
                )

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=True,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=True, check_q_boundary=check_m_boundary
                    ),
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=False,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    mask_fn=partial(
                        mask_fn_none, mask_seqlen=False, check_q_boundary=check_m_boundary
                    ),
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_block_cnt == 0,
    )


# =============================================================================
# Backward-specific block-sparse helpers (SM100)
# =============================================================================
#
# In backward, iteration is transposed compared to forward:
# - Forward: outer loop over m_blocks (Q tiles), inner loop over n_blocks (KV tiles)
# - Backward: outer loop over n_blocks (KV tiles), inner loop over m_blocks (Q tiles)
#
# The backward block-sparse tensors use "Q direction" indexing:
# - q_block_cnt[batch, head, n_block] → count of m_blocks to process for this KV tile
# - q_block_idx[batch, head, n_block, :] → indices of m_blocks to process
#
