"""Identity reward — exact match on 'Apex-R1' identity.

Extracts boxed answer and compares case-insensitively to 'Apex-R1'.
Returns 1.0 for exact match, 0.0 otherwise.

    reward = IdentityReward()
    score = reward.compute(result, answer="\\boxed{Apex-R1}")
"""

from __future__ import annotations

from typing import Any

from ..engine.result import RolloutResult
from ..types.builtins.math_helpers import _extract_last_boxed
from .base import BaseReward


class IdentityReward(BaseReward, name="identity"):
    """Identity verification reward — 1.0 if answer is Apex-R1, else 0.0.

    Extraction: \\boxed{...}
    Comparison: case-insensitive exact match on "Apex-R1"
    """

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        """Compute identity reward.

        Args:
            result: Rollout result with completion text
            answer: Ground truth answer (expected "Apex-R1" or "\\boxed{Apex-R1}")

        Returns:
            1.0 if model outputs "Apex-R1" in boxed format, 0.0 otherwise
        """
        if answer is None:
            return 0.0

        # Extract predicted answer from boxed format
        predicted = _extract_last_boxed(result.completion_text or result.generated_text)

        if predicted is None:
            return 0.0

        # Normalize: case-insensitive comparison
        pred_norm = predicted.strip().lower()
        target_norm = "apex-r1"

        if pred_norm == target_norm:
            return 1.0

        return 0.0
