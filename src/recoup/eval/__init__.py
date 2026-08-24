"""Evaluation harness. Deterministic, LLM-free, common random numbers across policies."""

from recoup.eval.harness import EvalResult, World, build_world, compare, evaluate
from recoup.eval.report import (
    RECURLY_PUBLISHED_RECOVERY,
    format_calibration_check,
    format_comparison,
    format_waste_breakdown,
)

__all__ = [
    "RECURLY_PUBLISHED_RECOVERY",
    "EvalResult",
    "World",
    "build_world",
    "compare",
    "evaluate",
    "format_calibration_check",
    "format_comparison",
    "format_waste_breakdown",
]
