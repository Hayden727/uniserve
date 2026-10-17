"""TMEM/SMEM layout and warp policy for the private sparse forward.

The public 128x64 recipe uses 128 query rows and one query stage. The original
layout helpers keep copy atoms, stage offsets, scale storage and accumulator
correction consistent across every kernel call site.
"""

import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Float32, const_expr

#: A TMEM accumulator is always consumed by 4 warps: ``get_tmem_load_op`` hard-codes
#: ``tmem_warp_shape_mn = (4, 1)`` for the non-2CTA case, so ``num_dp = m_block_size / 4``.
#: At m=64 that is 16 datapaths per warp. (An earlier note here said "2 threads per row"; the
#: measured layout below shows it is **4 lanes per row and 2 rows per thread** — see
#: ``lanes_per_row`` / ``rows_per_thread``.)
#: Shrinking the stage to 2 warps was tried and measured: it fixes neither the compile error nor the
#: numerics, and it contradicts the derived atom.
STAGE_WARPS = 4

#: Below this, the accumulator has a (16, 2) TMEM lane layout that 32-datapath atoms cannot express.
DP32_MIN_M = 128


def uses_16dp(m_block_size: int) -> bool:
    """Does this m require the 16-datapath atom family?"""
    return m_block_size < DP32_MIN_M


def num_stage_warps(m_block_size: int) -> int:
    """Warps per softmax / correction stage. Constant by CUTLASS convention — see ``STAGE_WARPS``."""
    del m_block_size
    return STAGE_WARPS


def num_scale_slots(m_block_size: int) -> int:
    """Slots per ``sScale`` half — one per (THREAD, row-of-that-thread), never per accumulator row.

    ``sScale[tidx + r * 128 + stage * slots]`` is written by the softmax warps and read with the
    same ``tidx`` by the correction warps, so the array must be indexed by thread. Rows and threads
    coincide at m=128 (128 threads, 128 rows, one row each), which is why stock FA4 sizes it by
    ``m_block_size`` — but at m=64 there are still 128 threads, each spanning
    ``rows_per_thread`` = 2 rows that need separate scales. Exactly 128 at m=128, so this is
    byte-identical to stock there.
    """
    return cute.arch.WARP_SIZE * STAGE_WARPS * rows_per_thread(m_block_size)


# ── the m=64 accumulator fragment ─────────────────────────────────────────────────────────
# MEASURED, not derived — by printing the identity-tensor partition of the real atoms inside
# the running kernel (both the S accumulator in `softmax_step` and the O accumulator in
# `correction_epilogue`; they agree exactly). For thread ``t`` with ``w = t // 32``,
# ``l = t % 32``:
#
#     rows owned = { w*16 + l//4 ,  w*16 + l//4 + 8 }        4 warps x 16 rows = 64  ✓
#
# So at m=64 a thread's fragment spans **two different rows**, and **4 lanes** share each
# row pair, splitting that row's columns between them. Element order interleaves the two
# rows: on the S side ``(lo, hi, lo, hi, ...)``, on the O side in pairs
# ``([0,1]=lo, [2,3]=hi, [4,5]=lo, ...)``.
#
# This is the real m=64 numeric blocker, and it is NOT "a hard-coded quad width" anywhere.
# `SoftmaxSm100.create` builds its state with ``num_rows = 1`` because at m=128 a thread
# owns exactly one row -- which is also why stock's `fadd_reduce`/`fmax_reduce` are correct
# with no cross-lane step at all. At m=64 that single scalar conflates row r and row r+8, so
# no choice of shuffle width can repair it; the state has to carry one entry per row.

#: Element order within one thread's fragment: which of the thread's rows element ``i`` sits
#: in. MEASURED per atom with ``read_probe.py`` -- the families genuinely differ, so
#: this cannot be derived from ``m_block_size`` alone and every consumer must name its own.
INTERLEAVED = "interleaved"  # period 2:  lo, hi, lo, hi, ...       (Ld16x128b / St16x128b)
PAIRED = "paired"  # period 4:  [0,1] lo, [2,3] hi, ...   (Ld16x256b, O epilogue)

#: Per consumer, measured. ``S_LOAD`` is PAIRED *because* ``s_load_atom`` asks
#: ``get_tmem_load_op`` for a 16-bit destination -- see that function.
S_LOAD_ORDER = PAIRED
CORRECTION_RESCALE_ORDER = INTERLEAVED
EPILOGUE_ORDER = PAIRED


def row_of_element(m_block_size: int, order: str, i: int) -> int:
    """Which of this thread's rows element ``i`` of its fragment belongs to.

    Always 0 at m>=128, where a thread owns exactly one row -- so every caller collapses to
    the stock single-row expression there.
    """
    rows = rows_per_thread(m_block_size)
    if rows == 1:
        return 0
    assert order in (INTERLEAVED, PAIRED), f"unmeasured element order {order!r}"
    return (i % rows) if order == INTERLEAVED else ((i // 2) % rows)


def elements_of_row(m_block_size: int, order: str, row: int, n_elements: int) -> list:
    """Fragment element indices belonging to ``row``. Trace-time list, no device cost."""
    return [i for i in range(n_elements) if row_of_element(m_block_size, order, i) == row]


def lanes_per_row(m_block_size: int) -> int:
    """Threads that share one accumulator row (they split its columns)."""
    return 4 if uses_16dp(m_block_size) else 1


def rows_per_thread(m_block_size: int) -> int:
    """Accumulator rows a single thread's fragment spans.

    ``128 threads * rows_per_thread / lanes_per_row == m_block_size``: 1 at m=128 (the stock
    assumption), 2 at m=64.
    """
    threads = cute.arch.WARP_SIZE * STAGE_WARPS
    rows = m_block_size * lanes_per_row(m_block_size) // threads
    assert rows * threads == m_block_size * lanes_per_row(m_block_size)
    return rows


def row_stride_in_thread(m_block_size: int) -> int:
    """Row distance between the entries of one thread's fragment (8 at m=64, unused at 128)."""
    rows_per_warp = m_block_size // STAGE_WARPS
    return rows_per_warp // rows_per_thread(m_block_size)


def warp_id_ranges(m_block_size: int):
    """``(softmax0, softmax1, correction, empty)`` warp-id tuples for a 16-warp CTA."""
    w = num_stage_warps(m_block_size)
    return (
        tuple(range(0, w)),
        tuple(range(w, 2 * w)),
        tuple(range(2 * w, 3 * w)),
        tuple(range(3 * w, 12)) + (15,),
    )


def use_kv_split(m_block_size: int, q_stage: int, use_tma_KV: bool) -> bool:
    """Should one Q tile be split across two independent KV streams?

    At ``q_stage == 2`` the kernel's two stages are two *different* Q tiles that share one
    block-sparse top-k list, which is what makes K0 a 256-row Q granularity. A config that
    wants a 128-row granularity must run ``q_stage == 1``, and then stock FA4 parks warps
    4-7 (and 15) in ``empty_warp_ids``: 11 of 16 warps active, measured. A TMEM accumulator
    is consumed by exactly 4 warps (see ``STAGE_WARPS``), so the only way to give warps 4-7
    work is a **second accumulator** — hence two KV streams over the same Q tile, merged in
    the epilogue.

    Gated off at ``q_stage == 2`` so the m>=128 reference path stays byte-identical to
    stock FA4 (boundary 4), and off without TMA KV because that variant spends warps 4-7 on
    loads instead (``flash_fwd_sm100.py`` q_stage==1 branch).

    No longer gated below m=128. Two things had to be true first, and the note that used to
    be here named neither: it guessed the m=64 NaNs came from the merge amplifying the P-store
    bug, which was falsified by lifting the gate after ``s_load_atom`` was fixed — single-stream
    m=64 was exact and the split still NaN'd. The real blockers were

    1. ``kv_stage``'s SMEM budget, which assumed a 2048-byte ``sScale`` — see
       ``kv_stage_budget_bytes``; and
    2. ``correction_kv_split_tile`` / ``correction_epilogue_merge`` still being single-row.
       They read ``sScale[tidx + s*slots]``, row 0's slot only, and gated the rescale on a
       ballot over that one row, so when row 1's max moved and row 0's did not, row 1's O
       accumulator was never rescaled and its online sum overflowed to inf.
    """
    return q_stage == 1 and use_tma_KV


def num_streams(m_block_size: int, q_stage: int, use_tma_KV: bool) -> int:
    """Independent ``(row_max, row_sum, O)`` accumulators per CTA.

    Sizes every per-stage resource: ``tmem_o_offset``, ``tOtO``, ``sScale``, and the
    softmax<->correction<->MMA mbarrier arrays. Equals ``q_stage`` in every stock
    configuration, so nothing moves at m=128.
    """
    return 2 if use_kv_split(m_block_size, q_stage, use_tma_KV) else q_stage


# ── the SMEM budget that sets kv_stage ────────────────────────────────────────────────────
#: Usable dynamic SMEM per CTA on sm_100a. The launcher itself is the authority here -- it
#: rejects anything larger with "launch shared memory exceeds ... Max: 232448 bytes".
SMEM_CAPACITY = 232448

#: Everything in ``SharedStorage`` that is neither sQ/sO/sK nor sScale: the mbarrier arrays
#: and the TMEM bookkeeping. MEASURED at 1024 B in every config, K0 through K3 -- it is an
#: alignment quantum (the barriers total ~450 B and sO is 1024-B aligned), not a coincidence,
#: which is why it does not move with ``kv_stage``. Asserted at trace time in
#: ``flash_fwd_sm100.py`` so it cannot drift silently.
SMEM_FIXED_OVERHEAD = 1024


def scale_smem_bytes(m_block_size: int, q_stage: int, use_tma_KV: bool) -> int:
    """Bytes of ``sScale`` -- one fp32 per (stream, thread, row, {row_sum, row_max})."""
    slots = num_streams(m_block_size, q_stage, use_tma_KV) * num_scale_slots(m_block_size)
    return slots * 2 * Float32.width // 8


def kv_stage_budget_bytes(m_block_size: int, q_stage: int, use_tma_KV: bool) -> int:
    """SMEM left for sQ + sO + sK once the fixed parts are paid for.

    Upstream writes this as a bare ``224 * 1024``, which is exactly
    ``SMEM_CAPACITY - 2048 - SMEM_FIXED_OVERHEAD`` -- i.e. the constant silently assumes
    ``sScale`` is 2048 B, which it is in every stock configuration and in K2. At m=64 it is
    4096 B, because ``num_scale_slots`` doubles when a thread owns two accumulator rows, and
    the assumption overdraws by exactly that 2048 B: measured, ``kv_stage`` came out 12 and
    the launch was refused at 234496 bytes against the 232448 limit.

    Stating the budget in terms of the bytes actually consumed reproduces every stock
    ``kv_stage`` exactly -- K0 3, K1 5, K2 10, all verified -- and gives m=64 one stage
    fewer instead of one stage too many.
    """
    return SMEM_CAPACITY - SMEM_FIXED_OVERHEAD - scale_smem_bytes(m_block_size, q_stage, use_tma_KV)


def stage_by_columns(tStS: cute.Tensor, n_block_size: int) -> cute.Tensor:
    """Force the S accumulator's two stages ``n_block_size`` COLUMNS apart.

    Everything else in the kernel already assumes that: ``gemm_Si[stage]`` hands the MMA
    ``tmem_s_offset[stage]`` (``[0, n_block_size]``) as the accumulator address, and
    ``tmem_p_offset`` derives P's per-stage address the same way. ``make_fragment_C`` does
    NOT always agree. MEASURED, by printing ``tStS.layout``:

    ===========  =========================  ==============================
    m            stage stride it chose      meaning
    ===========  =========================  ==============================
    128          64                         ``n_block_size`` columns  (agrees)
    64           1048576 (== 16 << 16)      16 LANES, same columns    (disagrees)
    ===========  =========================  ==============================

    At m=64 the accumulator uses only 16 of each warp's 32 TMEM lanes, so ``make_fragment_C``
    packs stage 1 into the other 16 rather than into later columns. That is denser, but it is
    not what the MMA is told, and the lane-offset region is not a legal MMA A-operand base
    either -- pointing P's descriptor at it faults with `misaligned address`. So the stages
    are moved back onto columns, which costs ``n_block_size`` TMEM columns that were already
    reserved by ``tmem_s_offset`` and left unused.

    The symptom this fixes: under ``use_kv_split`` at m=64 the MMA wrote S1 at column
    ``n_block_size`` while the softmax warps read stage 1 at lane+16, so stream 1 consumed
    uninitialized TMEM and O1 came out NaN on every row -- while stream 0, whose stride is 0
    either way, stayed exact. At m>=128 this returns ``tStS`` with the stride it already had.
    """
    layout = tStS.layout
    strides = list(layout.stride)
    strides[-1] = n_block_size
    return cute.make_tensor(tStS.iterator, cute.make_layout(layout.shape, stride=tuple(strides)))


def rep_16dp(cols_per_iteration: int) -> int:
    """Repetition for a 16-dp / 128-bit TMEM op covering ``cols_per_iteration`` fp32 columns.

    One repetition of a 128-bit op moves 4 fp32 columns, so ``rep = cols / 4``. Cross-checked
    against ``get_tmem_load_op``, which derives ``Ld16x128b(Rep32)`` for the 128-column S
    accumulator — and 128/4 = 32.

    Pass the columns the STOCK 32-dp op covered **per iteration**, not the full tensor width: the
    P-store loop derives its mid-loop ``split_P_arrive`` notification index from
    ``tStP_r2t.shape[2]``, so collapsing 4 iterations into 1 silently disables that release
    (``shape[2] * 96 // 128`` becomes 0) and the MMA warp is never signalled — a hang, not a
    slowdown.
    """
    rep = cols_per_iteration // 4
    assert rep in (1, 2, 4, 8, 16, 32, 64, 128), (
        f"no legal 16-dp repetition for {cols_per_iteration} fp32 columns/iteration (got {rep})"
    )
    return rep


def align_tmem(tensor: cute.Tensor, dtype, offset: int = 0) -> cute.Tensor:
    """Re-declare a TMEM tensor's pointer as 8B (2-column) aligned, optionally after an offset.

    ``iterator + offset`` drops the declared alignment to 1 byte. Harmless for 32-dp atoms
    (1-column granularity), but the 16-dp atoms m=64 needs demand 2-column alignment and are
    rejected at ``cute.copy``. Offsets here are static even numbers of fp32 columns (multiples of
    ``n_block_size`` / ``head_dim_v_padded``), so the declaration is truthful — asserted, not
    trusted. ``Pointer.align()`` refuses TMEM pointers, so go through ``make_ptr`` directly (which
    is what ``align`` itself does).
    """
    assert offset % 2 == 0, (
        f"TMEM offset {offset} must be an even number of fp32 columns for the 8B-aligned atoms"
    )
    it = tensor.iterator + offset if offset else tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(dtype, it.toint(), cute.AddressSpace.tmem, assumed_align=8), tensor.layout
    )


# ── copy atoms ────────────────────────────────────────────────────────────────────────────────
# Hand-picking a 16-dp op that merely makes the SHAPES match compiles and runs, but the resulting
# thread<->element mapping is wrong: measured, every output row was incorrect while only ~2% were
# NaN. CuTe validates shape, not semantics. So the load side is always DERIVED via CUTLASS's own
# `get_tmem_load_op`, and the store side is paired to it through `rep_16dp` rather than guessed.


def s_load_atom(
    m_block_size, mma_tiler_qk, o_layout, acc_dtype, n_block_size, use_2cta_instrs, p_dtype
):
    """TMEM->register atom for the S accumulator in the softmax stage.

    ``elem_ty_d`` is ``p_dtype`` (16-bit), **not** ``acc_dtype``. That is the whole m=64
    fix, and it is not a hand-pick -- it is telling ``get_tmem_load_op`` the truth about
    what this fragment is for. The S registers are never consumed as fp32: they are
    converted to P's dtype and written back through ``p_store_atom``, which packs two
    16-bit values into one 32-bit TMEM word, so **both halves of a word are the same
    accumulator row at adjacent columns**. The softmax stage relies on that: it builds ONE
    fp32 register fragment shaped by the store atom and views it as 16-bit with the load
    atom's layout, so the two element orders have to agree or every P value lands in the
    wrong place.

    Asking for a 32-bit destination gets ``Ld16x128b(Rep32)``, whose order interleaves the
    thread's two rows every element (``lo, hi, lo, hi``) -- so the recast pairs elements
    from *different rows* into one word, which no store atom can undo. Asking for a 16-bit
    destination gets ``Ld16x256b(Rep16)``, which groups them in same-row pairs (``PAIRED``)
    and matches ``p_store_atom`` exactly. Verified by partitioning an identity tensor with
    both real atoms inside the running kernel: MATCH on every probed thread at m=64, and
    the same check MATCHes at m=128 where stock is known correct
    (``read_probe.py``; ``FA4_VSA_LAYOUT_PROBE=1`` to reproduce).

    At m>=128 this branch is not taken at all, so nothing about the reference path moves.
    """
    if const_expr(not uses_16dp(m_block_size)):
        return cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), acc_dtype)
    return sm100_utils_basic.get_tmem_load_op(
        mma_tiler_qk,
        o_layout,
        p_dtype,
        acc_dtype,
        (m_block_size, n_block_size),
        use_2cta_instrs=use_2cta_instrs,
    )


def p_store_atom(m_block_size):
    """Register->TMEM atom for P. Stock covers 16 fp32 columns per iteration; keep that."""
    if const_expr(not uses_16dp(m_block_size)):
        return cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32)
    return cute.make_copy_atom(
        tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(rep_16dp(16))), Float32
    )


def correction_rescale_atoms(m_block_size, corr_tile_size, acc_dtype):
    """``(load, store)`` atoms for the O-accumulator rescale. This path is LIVE, not dead plumbing."""
    if const_expr(not uses_16dp(m_block_size)):
        ld = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size))
        st = tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size))
    else:
        rep = rep_16dp(corr_tile_size)
        ld = tcgen05.copy.Ld16x128bOp(tcgen05.copy.Repetition(rep))
        st = tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(rep))
    return cute.make_copy_atom(ld, acc_dtype), cute.make_copy_atom(st, acc_dtype)


def smem_store_atom(m_block_size, o_layout, o_dtype, acc_dtype, tiled_tmem_load):
    """Register->SMEM atom for O in the correction epilogue.

    ``get_smem_store_op`` picks a vectorized ``stmatrix`` whenever the TMEM load's TV ownership
    allows it, but it cannot see the DESTINATION: our per-thread ``sO`` slice is only 32-bit aligned
    at m=64 while ``stmatrix`` needs 128-bit, so ``cute.copy`` is rejected. Fall back to the scalar
    SIMT store — which is what the helper itself returns when it cannot vectorize. Correctness
    first; re-vectorizing needs the ``sO`` layout reworked for a 64-row epilogue tile.
    """
    if const_expr(not uses_16dp(m_block_size)):
        return sm100_utils_basic.get_smem_store_op(o_layout, o_dtype, acc_dtype, tiled_tmem_load)
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), o_dtype)


def build_tmem_scale_plumbing(m_block_size: int) -> bool:
    """Should the TMEM scale tensors be constructed at all?

    They are **dead**: every ``cute.copy`` against them is commented out upstream, and the live
    row_max/row_sum exchange goes through SMEM (``sScale``). At m=64 a 1-column ``(m, 1)`` TMEM view
    has no legal 16-dp atom (the x2 op needs 2 real columns, the 32-dp op needs 32 lanes), so we
    simply do not build them — verified free: removing them left the m=128 suite bit-exact.
    """
    return not uses_16dp(m_block_size)
