"""FP32 single-row projection reduction contracts."""

from typing import Literal, TypeAlias

GemvReduction: TypeAlias = Literal["default", "strided16"]


def validate_reduction(value: GemvReduction) -> None:
    """Require a declared order; checkpoint readers must not guess recipes."""
    if value not in {"default", "strided16"}:
        raise ValueError(f"unknown FP32 GEMV reduction {value!r}")
