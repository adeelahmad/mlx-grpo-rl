"""Reasoning quality reward — evaluates thinking depth and coherence.

Measures reasoning quality through structural and content heuristics:
- Step progression (numbered steps, logical flow)
- Self-correction and verification
- Appropriate verbosity (not too short, not too repetitive)
"""

from __future__ import annotations

import re

from ..engine.result import RolloutResult
from .base import BaseReward


class ReasoningReward(BaseReward, name="reasoning"):
    """Reward for reasoning quality in thinking segments."""

    def __init__(
        self,
        min_thinking_tokens: int = 20,
        max_thinking_tokens: int = 2000,
        step_bonus: float = 0.15,
        verification_bonus: float = 0.2,
        repetition_penalty: float = 0.3,
    ):
        self.min_thinking_tokens = min_thinking_tokens
        self.max_thinking_tokens = max_thinking_tokens
        self.step_bonus = step_bonus
        self.verification_bonus = verification_bonus
        self.repetition_penalty = repetition_penalty

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        thinking = result.variables.get("thinking")
        if not thinking or not thinking.text.strip():
            return 0.0

        text = thinking.text
        score = 0.3  # Base score for having thinking

        # Length appropriateness
        token_count = thinking.token_count
        if token_count < self.min_thinking_tokens:
            score -= 0.2  # Too short
        elif token_count > self.max_thinking_tokens:
            score -= 0.1  # Too long

        # Step progression
        step_patterns = [
            r"(?:step|first|second|third|then|next|finally)\b",
            r"\b\d+[\.\)]\s",
            r"(?:let's|let me|we can|we need)\b",
        ]
        steps_found = sum(1 for p in step_patterns if re.search(p, text, re.IGNORECASE))
        score += min(self.step_bonus * steps_found, 0.3)

        # Verification / self-correction
        verify_patterns = [
            r"(?:check|verify|confirm|wait|actually|let me reconsider)\b",
            r"(?:double.check|sanity.check|make sure)\b",
            r"(?:therefore|thus|so the answer|hence)\b",
        ]
        verifications = sum(1 for p in verify_patterns if re.search(p, text, re.IGNORECASE))
        score += min(self.verification_bonus * verifications, 0.3)

        # Repetition penalty
        sentences = re.split(r"[.!?]\s+", text)
        if len(sentences) > 3:
            unique_ratio = len(set(s.strip().lower() for s in sentences)) / len(sentences)
            if unique_ratio < 0.5:
                score -= self.repetition_penalty

        return max(0.0, min(1.0, score))
