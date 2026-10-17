"""SM100 dense FA4 with a fixed exp2 rounding recipe.

The softmax step is derived from FlashAttention 4.0.0b30 (BSD-3-Clause;
copyright 2022 the respective contributors; see adjacent FA4_LICENSE).
Tensor-memory loads, reductions and stores retain upstream
ordering. Head128 two-CTA tiles use emulation frequency12; masked boundary
iterations use hardware exp2. Other scheduling remains in the pinned FA4
kernel. This preserves an observable BF16 numerical recipe independently
from upstream performance tuning.
"""

# Preserve upstream CuTe fragment names to keep the port auditable.
# ruff: noqa: N803, N806

from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr, pipeline
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.utils import AuxData


class _DenseKernel(FlashAttentionForwardSm100):
    def __init__(self, query_length):
        super().__init__(
            128,
            128,
            q_stage=2 if query_length > 128 else 1,
            use_2cta_instrs=query_length > 256,
        )
        if self.use_2cta_instrs:
            self._tune = {**self._tune, "ex2_emu_freq": 12}

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: pipeline.PipelineAsync | None,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Callable | None = None,
        is_first: bool = False,
    ) -> tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Update one softmax tile in the upstream reduction/store order.

        A None mask_fn means the tcgen05.ld.red hardware max is valid.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = (
            self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        )
        tScS = thr_mma_qk.partition_C(
            cute.make_identity_tensor(self.mma_tiler_qk[:2])
        )
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        hw_row_max = Float32(-Float32.inf)
        if const_expr(self.use_ldred_rowmax):
            # ld.red returns each x32 tile's max in an extra register.
            tSrS_red = cute.make_rmem_tensor(
                ((1, 1), *tSrS_t2r.shape[1:]), self.qk_acc_dtype
            )
            cute.copy(thr_tmem_load, tStS_t2r, (tSrS_t2r, tSrS_red))
            for i in cutlass.range_constexpr(cute.size(tSrS_red.shape)):
                hw_row_max = cute.arch.fmax(hw_row_max, tSrS_red[i])
        else:
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_data,
                fastdiv_mods,
                head_divmod,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        # Masked iterations reduce over post-mask values in software.
        if const_expr(self.use_ldred_rowmax and mask_fn is None):
            row_max, acc_scale = softmax.update_row_max_precomputed(
                hw_row_max, is_first
            )
        else:
            row_max, acc_scale = softmax.update_row_max(
                tSrS_t2r.load(), is_first
            )

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(
                stage, s0_s1_sequence_phase
            )
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(
                cute.make_identity_tensor(tScP_shape)
            ).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
            tSrS_t2r.layout,
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq
            if const_expr(mask_fn is None)
            else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store,
                tSrP_r2t_f32[None, None, i],
                tStP_r2t[None, None, i],
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(
            stage, sm_stats_producer_phase
        )
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return (
            mma_si_consumer_phase ^ 1,
            sm_stats_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )


def _signature(tensor):
    if tensor is None:
        return None
    return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device)


class DenseAttention:
    """A compiled dense BSHD launch borrowing caller-owned inputs/outputs.

    Runtime owns instances and warms each shape before graph capture.
    No request tensors, streams or output buffers are retained. Inputs use
    BF16 equal-head attention with width128 on SM100; arbitrary masks,
    causal attention and paged state belong to other providers.
    """

    def __init__(self, q, k, v, out, lse, *, scale):
        if q.device.type != "cuda" or torch.cuda.get_device_capability(
            q.device
        ) != (10, 0):
            raise ValueError("this dense FA4 recipe requires SM100")
        if (
            q.ndim != 4
            or k.ndim != 4
            or v.shape != k.shape
            or q.shape[0] != k.shape[0]
            or q.shape[2:] != k.shape[2:]
            or q.shape[-1] != 128
            or min(q.shape) < 1
            or k.shape[1] < 1
            or out.shape != q.shape
            or any(
                t.dtype != torch.bfloat16 or t.device != q.device
                for t in (q, k, v, out)
            )
            or any(t.stride(-1) != 1 for t in (q, k, v, out))
        ):
            raise ValueError(
                "dense FA4 requires BF16 BSHD equal-head width128 tensors"
            )
        if lse is not None and (
            lse.shape != (q.shape[0], q.shape[2], q.shape[1])
            or lse.dtype != torch.float32
            or lse.device != q.device
        ):
            raise ValueError("dense FA4 LSE requires FP32 B,H,Q storage")
        self.signature = tuple(map(_signature, (q, k, v, out, lse)))
        self.query_length = q.shape[1]
        tensors = [to_cute_tensor(t) for t in (q, k, v, out)]
        tensors.append(to_cute_tensor(lse, assumed_align=4))
        arguments = self._arguments(*tensors, scale)
        self.launch = cute.compile(
            _DenseKernel(self.query_length),
            *arguments,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    def _arguments(self, q, k, v, out, lse, scale):
        # No prefix cache, variable lengths, sparsity, sinks, or dynamic
        # scheduling tables. The final integer bounds the complete query.
        return (
            q,
            k,
            v,
            out,
            lse,
            scale,
            *([None] * 10),
            AuxData(),
            *([None] * 7),
            self.query_length,
        )

    def __call__(self, q, k, v, out, lse, *, scale):
        if tuple(map(_signature, (q, k, v, out, lse))) != self.signature:
            raise ValueError("dense FA4 launch tensors changed shape or layout")
        self.launch(*self._arguments(q, k, v, out, lse, scale))
        return out
