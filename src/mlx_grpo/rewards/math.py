"""Math reward — exact match with LaTeX normalization and numeric tolerance.

Extracts answers from ``\\boxed{...}``, normalizes LaTeX formatting, and
compares with numeric tolerance for float answers.

    reward = MathReward()
    score = reward.compute(result, answer="\\boxed{2\\pi}")
"""

from __future__ import annotations

from typing import Any

from ..engine.result import RolloutResult
from ..types.builtins.math_helpers import _extract_boxed as _mh_extract_boxed
from ..types.builtins.math_helpers import _extract_last_boxed as _mh_extract_last_boxed
from ..types.builtins.math_helpers import _normalize_math_answer
from .base import BaseReward


class MathReward(BaseReward, name="math"):
    """Math-aware accuracy reward with LaTeX normalization.

    Extraction: ``\\boxed{...}`` with nested brace support.
    Normalization: strips LaTeX commands, converts symbols, evaluates numeric fracs.
    Comparison: normalized string match, then numeric with tolerance.
    """

    def __init__(self, rel_tolerance: float = 0.01, **kwargs: Any):
        self.rel_tolerance = rel_tolerance

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        if answer is None:
            return 0.0

        # Use LAST boxed — injected scaffold prefix may contain intermediate
        # \boxed{} answers from the reference trace; final answer is always last.
        predicted = _mh_extract_last_boxed(result.completion_text or result.generated_text)

        # Fallback: answer variable - take up to first }
        if predicted is None:
            answer_var = result.variables.get("answer")
            if answer_var and answer_var.text.strip():
                raw = answer_var.text.strip()
                # Take content before first } (closing the \boxed{)
                brace_idx = raw.find("}")
                predicted = raw[:brace_idx].strip() if brace_idx > 0 else raw.strip()

        if predicted is None:
            return 0.0

        gt = _mh_extract_boxed(answer) or answer.strip()

        pred_norm = _normalize_math_answer(predicted)
        gt_norm = _normalize_math_answer(gt)

        if pred_norm == gt_norm:
            return 1.0

        # Numeric tolerance fallback
        try:
            pred_f = float(pred_norm.replace(",", ""))
            gt_f = float(gt_norm.replace(",", ""))
            if gt_f == 0:
                return 1.0 if pred_f == 0 else 0.0
            if abs(pred_f - gt_f) / abs(gt_f) <= self.rel_tolerance:
                return 1.0
        except ValueError:
            pass

        return 0.0
