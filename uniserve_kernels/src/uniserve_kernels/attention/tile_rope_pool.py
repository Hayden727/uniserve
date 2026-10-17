"""Fused interleaved RoPE, valid-prefix tile packing and stored-BF16 means.

The BT64 reduction pools dtype-rounded Q/K/V stores, preserving their original
accumulation order. Callers own every destination and all index/rotary tensors.
Transferred from Impossible research/core/kernels/vsa_tile_rope_pool_fused.py,
source SHA256 425cdf302934ccc40210860cb3f53a47e4e132e4636b11a88d3c6908d70621b9.
"""

# Preserve numerical names for comparison with the authenticated source.
# ruff: noqa: N803, N806
import torch

from uniserve_kernels.triton import tl, triton, unsupported_operands

if triton is not None:

    @triton.jit
    def _rope_pair(x, c_e, c_o, s_e, s_o, BT: tl.constexpr, BD: tl.constexpr):
        """Rotate FP32 row pairs in registers before their dtype store."""
        xp = tl.reshape(x, (BT, BD, 2))
        x_e, x_o = tl.split(xp)  # (BT, BD) even / odd lanes
        y_e = x_e * c_e - x_o * s_e
        y_o = x_o * c_o + x_e * s_o
        return tl.reshape(tl.join(y_e, y_o), (BT, 2 * BD))

    @triton.jit
    def _tile_rope_pool_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,  # (B, S, H, D) contiguous
        cos_ptr,
        sin_ptr,  # (S, D) fp32, or (S, D/2) if ROPE_DUP (unused if HAS_ROPE=0)
        src_ptr,  # (S_pad,) int64, -1 = pad
        bs_ptr,  # (nb,) int32
        qt_ptr,
        kt_ptr,
        vt_ptr,
        gt_ptr,  # (B, S_pad, H, D)
        qc_ptr,
        kc_ptr,
        vc_ptr,  # (B, H, nb, D)
        S,
        H,
        D,
        Sp,
        T: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        HAS_ROPE: tl.constexpr,
        ROPE_DUP: tl.constexpr = False,
    ):
        """Assign one program to each head, tile and batch.

        The head axis varies fastest so one tile shares rotary rows in L2.
        """
        h = tl.program_id(0)
        t = tl.program_id(1)
        b = tl.program_id(2)
        nb = tl.num_programs(1)

        lane = tl.arange(0, 2 * BD)  # full row, contiguous
        lm = lane < D

        acc_q = tl.zeros((2 * BD,), dtype=tl.float32)
        acc_k = tl.zeros((2 * BD,), dtype=tl.float32)
        acc_v = tl.zeros((2 * BD,), dtype=tl.float32)

        out_ty = qt_ptr.dtype.element_ty

        for t0 in range(0, T, BT):
            tok = t0 + tl.arange(0, BT)
            tin = tok < T
            slot = t * T + tok
            src = tl.load(src_ptr + slot, mask=tin, other=-1)
            valid = (src >= 0) & tin
            src0 = tl.where(valid, src, 0).to(tl.int64)

            # int64 base offsets against >2^31-element tensors
            ib = ((b.to(tl.int64) * S + src0) * H + h) * D
            ob = ((b.to(tl.int64) * Sp + slot.to(tl.int64)) * H + h) * D
            m2 = valid[:, None] & lm[None, :]
            sm2 = tin[:, None] & lm[None, :]  # store ALL T slots (pads = 0)

            if HAS_ROPE:
                if ROPE_DUP:
                    # Paired tables provide one value to both rotary lanes.
                    laneh = tl.arange(0, BD)
                    hm = valid[:, None] & (laneh[None, :] < (D // 2))
                    cb = src0 * (D // 2)
                    c_e = tl.load(
                        cos_ptr + cb[:, None] + laneh[None, :],
                        mask=hm,
                        other=0.0,
                    )
                    s_e = tl.load(
                        sin_ptr + cb[:, None] + laneh[None, :],
                        mask=hm,
                        other=0.0,
                    )
                    c_o, s_o = c_e, s_e
                else:
                    cb = src0 * D
                    cos_row = tl.load(
                        cos_ptr + cb[:, None] + lane[None, :],
                        mask=m2,
                        other=0.0,
                    )
                    sin_row = tl.load(
                        sin_ptr + cb[:, None] + lane[None, :],
                        mask=m2,
                        other=0.0,
                    )
                    c_e, c_o = tl.split(tl.reshape(cos_row, (BT, BD, 2)))
                    s_e, s_o = tl.split(tl.reshape(sin_row, (BT, BD, 2)))

            # ---- q (rope + pool) ----
            x = tl.load(
                q_ptr + ib[:, None] + lane[None, :], mask=m2, other=0.0
            ).to(tl.float32)
            if HAS_ROPE:
                x = _rope_pair(x, c_e, c_o, s_e, s_o, BT, BD)
            xc = x.to(out_ty)
            tl.store(qt_ptr + ob[:, None] + lane[None, :], xc, mask=sm2)
            acc_q += tl.sum(tl.where(m2, xc.to(tl.float32), 0.0), axis=0)

            # ---- k (rope + pool) ----
            x = tl.load(
                k_ptr + ib[:, None] + lane[None, :], mask=m2, other=0.0
            ).to(tl.float32)
            if HAS_ROPE:
                x = _rope_pair(x, c_e, c_o, s_e, s_o, BT, BD)
            xc = x.to(out_ty)
            tl.store(kt_ptr + ob[:, None] + lane[None, :], xc, mask=sm2)
            acc_k += tl.sum(tl.where(m2, xc.to(tl.float32), 0.0), axis=0)

            # ---- v (no rope, pooled) ----
            xc = tl.load(
                v_ptr + ib[:, None] + lane[None, :], mask=m2, other=0.0
            )
            tl.store(vt_ptr + ob[:, None] + lane[None, :], xc, mask=sm2)
            acc_v += tl.sum(tl.where(m2, xc.to(tl.float32), 0.0), axis=0)

            # ---- gate (no rope, not pooled) ----
            xc = tl.load(
                g_ptr + ib[:, None] + lane[None, :], mask=m2, other=0.0
            )
            tl.store(gt_ptr + ob[:, None] + lane[None, :], xc, mask=sm2)

        bs = tl.load(bs_ptr + t).to(tl.float32)
        cb_out = ((b.to(tl.int64) * H + h) * nb + t) * D
        tl.store(qc_ptr + cb_out + lane, (acc_q / bs).to(out_ty), mask=lm)
        tl.store(kc_ptr + cb_out + lane, (acc_k / bs).to(out_ty), mask=lm)
        tl.store(vc_ptr + cb_out + lane, (acc_v / bs).to(out_ty), mask=lm)


def tile_rope_pool(q, k, v, gate, cos, sin, source, counts, *, tiled, pooled):
    """Write four tiled projections and three stored-BF16 tile means.

    Inputs are contiguous SHD BF16, full-lane FP32 rotary tables, an int64
    padded-slot source map (-1 for padding), and int32 valid-prefix counts.
    Tiled destinations are padded SHD; pooled destinations are HTD. Full-lane
    table loads support arbitrary even/odd lanes without probing device data
    or caching pointer identities. This is inference-only computation.
    """
    operands = (q, k, v, gate, cos, sin, source, counts, *tiled, *pooled)
    reason = unsupported_operands(*operands)
    if reason is not None:
        raise ValueError(f"tile RoPE/pooling requires a CUDA kernel: {reason}")
    if (
        q.ndim != 3
        or min(q.shape) < 1
        or q.shape[-1] != 128
        or any(
            value.shape != q.shape or value.dtype != torch.bfloat16
            for value in (q, k, v, gate)
        )
        or len(tiled) != 4
        or len(pooled) != 3
        or counts.ndim != 1
        or not counts.numel()
        or counts.dtype != torch.int32
        or source.ndim != 1
        or source.dtype != torch.int64
        or source.numel() % counts.numel()
        or source.numel() // counts.numel() not in (128, 256)
        or any(
            value.shape != (q.shape[0], q.shape[2])
            or value.dtype != torch.float32
            for value in (cos, sin)
        )
        or any(
            value.shape != (source.numel(), *q.shape[1:])
            or value.dtype != q.dtype
            for value in tiled
        )
        or any(
            value.shape != (q.shape[1], counts.numel(), q.shape[2])
            or value.dtype != q.dtype
            for value in pooled
        )
        or any(not value.is_contiguous() for value in operands)
    ):
        raise ValueError(
            "tile RoPE/pooling tensors disagree with the BF16 SHD128 contract"
        )
    tokens, heads, width = q.shape
    tile = source.numel() // counts.numel()
    _tile_rope_pool_fwd[(heads, counts.numel(), 1)](
        q,
        k,
        v,
        gate,
        cos,
        sin,
        source,
        counts,
        *tiled,
        *pooled,
        tokens,
        heads,
        width,
        source.numel(),
        T=tile,
        BT=64,
        BD=64,
        HAS_ROPE=True,
        ROPE_DUP=False,
        num_warps=4,
    )
