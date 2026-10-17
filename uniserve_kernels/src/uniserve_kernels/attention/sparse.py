"""Shape-bound sparse launches borrowing runtime-owned tensor storage."""

import math

import cutlass.cute as cute
import torch

from .dense_sm100 import _signature

MAP_FIELDS = ("mask_counts", "mask_indices", "full_counts", "full_indices")


class SparseLaunch:
    """Compile before capture and retain code and scalar signatures only.

    Concrete kernels supply their numerical ABI. Every launch receives live
    maps, validity and destinations; no tensor, CUDA stream or context is
    retained by the compiled-plan owner.
    """

    logical_tiles: tuple[int, ...]
    key_tile: int
    needs_lse = False

    def __init__(self, q, k, v, out, buffers, *, tile, scale):
        if (
            tile not in self.logical_tiles
            or not math.isfinite(scale)
            or scale <= 0
            or q.device.type != "cuda"
            or torch.cuda.get_device_capability(q.device) != (10, 0)
            or q.ndim != 4
            or k.ndim != 4
            or min(q.shape) < 1
            or k.shape[1] < 1
            or q.shape[0] != 1
            or q.shape[-1] != 128
            or q.shape[1] % tile
            or k.shape[1] % tile
            or k.shape != v.shape
            or q.shape[2:] != k.shape[2:]
            or k.shape[0] != 1
            or out.shape != q.shape
            or any(t.dtype != torch.bfloat16 for t in (q, k, v, out))
            or any(t.device != q.device for t in (k, v, out))
            or any(not t.is_contiguous() for t in (q, k, v, out))
        ):
            raise ValueError(
                "sparse FA4 requires contiguous BF16 BSHD128 inputs/output "
                "and a positive finite scale on SM100"
            )
        count_shape = (1, q.shape[2], q.shape[1] // tile)
        indices = buffers["mask_indices"]
        if (
            indices.ndim != 4
            or indices.shape[:3] != count_shape
            or indices.shape[-1] < 1
            or buffers["full_indices"].shape != indices.shape
            or any(
                buffers[name].shape != count_shape
                for name in ("mask_counts", "full_counts")
            )
            or buffers["valid"].shape != (k.shape[1] // self.key_tile,)
            or any(
                value.dtype != torch.int32
                or value.device != q.device
                or not value.is_contiguous()
                for value in (buffers[name] for name in (*MAP_FIELDS, "valid"))
            )
        ):
            raise ValueError("sparse maps disagree with the compiled domain")
        lse = buffers.get("lse")
        if self.needs_lse:
            if (
                lse is None
                or lse.shape != (1, q.shape[2], q.shape[1])
                or lse.dtype != torch.float32
                or lse.device != q.device
                or not lse.is_contiguous()
            ):
                raise ValueError("hoisted sparse attention requires FP32 LSE")
        elif lse is not None:
            raise ValueError("merged sparse attention does not expose LSE")
        self.query_length, self.scale = q.shape[1], scale
        self.signature = self._signature(q, k, v, out, buffers)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare sparse FA4 shapes before capture")
        self.launch = cute.compile(
            self.kernel_class(tile),
            *self._arguments(q, k, v, out, buffers, scale, compiling=True),
            options="--enable-tvm-ffi",
        )

    @staticmethod
    def _signature(q, k, v, out, buffers):
        return tuple(
            map(
                _signature,
                (
                    q,
                    k,
                    v,
                    out,
                    *(buffers[name] for name in MAP_FIELDS),
                    buffers["valid"],
                    buffers.get("lse"),
                ),
            )
        )

    def __call__(self, q, k, v, out, buffers, *, scale):
        if (
            scale != self.scale
            or self._signature(q, k, v, out, buffers) != self.signature
        ):
            raise ValueError("sparse FA4 launch tensors or scale changed")
        self.launch(
            *self._arguments(
                q,
                k,
                v,
                out,
                buffers,
                scale,
                compiling=False,
            )
        )
