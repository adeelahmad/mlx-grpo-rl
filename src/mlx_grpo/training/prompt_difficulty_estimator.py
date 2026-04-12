"""Prompt difficulty estimation for scaffold boundary optimization.

Estimates difficulty [0, 1] from prompt/sample features to inform
scaffold ratio selection and boundary tracking.

Usage:
    estimator = PromptDifficultyEstimator()
    difficulty = estimator.estimate(prompt="...", answer="...", sample_type="math")
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


class PromptDifficultyEstimator:
    """Estimates prompt difficulty from text and metadata."""

    def __init__(self) -> None:
        """Initialize estimator with feature weights."""
        # Difficulty factors: scaled to [0, 1]
        self.length_weight = 0.2  # Longer prompts tend to be harder
        self.token_weight = 0.3  # Token count proxy for complexity
        self.answer_weight = 0.2  # Longer answers = harder target
        self.type_weight = 0.2  # Task type inherent difficulty

        # Type base difficulties
        self.type_difficulties: dict[str, float] = {
            "general_qna": 0.3,
            "mcq": 0.35,
            "math": 0.6,  # Competition math is hard
            "python_code": 0.55,
            "tool_call": 0.4,
            "vqa": 0.45,
            "image_captioning": 0.4,
        }

    def estimate(
        self,
        prompt: str = "",
        answer: str = "",
        sample_type: str = "general_qna",
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Estimate difficulty [0, 1] from prompt and answer.

        Args:
            prompt: Question/prompt text
            answer: Ground truth answer text
            sample_type: Task type (e.g., "math", "mcq")
            metadata: Optional sample metadata (may contain hints)

        Returns:
            Difficulty score [0, 1]
        """
        metadata = metadata or {}
        difficulty = 0.0

        # Type base difficulty
        type_diff = self.type_difficulties.get(sample_type, 0.5)
        difficulty += self.type_weight * type_diff

        # Prompt length: normalized to [0, 1] (0-1000 chars)
        prompt_len_score = min(len(prompt) / 1000.0, 1.0)
        difficulty += self.length_weight * prompt_len_score

        # Token estimate: roughly 4 chars per token
        token_estimate = len(prompt) / 4.0
        token_score = min(token_estimate / 500.0, 1.0)
        difficulty += self.token_weight * token_score

        # Answer length: longer answers = harder targets
        answer_len_score = min(len(answer) / 500.0, 1.0)
        difficulty += self.answer_weight * answer_len_score

        # Clamp to [0, 1]
        return float(min(max(difficulty, 0.0), 1.0))

    def update_type_difficulties(self, type_difficulties: dict[str, float]) -> None:
        """Update base difficulties for task types.

        Args:
            type_difficulties: {type_name: difficulty [0, 1]}
        """
        self.type_difficulties.update(type_difficulties)
