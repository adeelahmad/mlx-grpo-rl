"""Multiplicative reward — product of per-checkpoint scores.

Enforces that every checkpoint must be correct. A single zero
at any checkpoint zeros the total reward.

    reward = MultiplicativeReward(
        checkpoint_scorer=my_scorer_fn,
        checkpoint_tags=["checkpoint_0", "checkpoint_1"],
    )
    score = reward.compute(result, answer="42")
"""

from __future__ import annotations

from collections.abc import Callable

from ..engine.result import RolloutResult
from .base import BaseReward


def _default_scorer(segment_text: str, answer: str | None) -> float:
    """Default scorer: normalized substring match against answer."""
    if not answer or not segment_text.strip():
        return 0.0
    seg_clean = " ".join(segment_text.strip().split()).lower()
    ans_clean = " ".join(answer.strip().split()).lower()
    if seg_clean == ans_clean:
        return 1.0
    if ans_clean in seg_clean or seg_clean in ans_clean:
        return 0.5
    return 0.0


class MultiplicativeReward(BaseReward, name="multiplicative"):
    """Product of per-checkpoint scores.

    Each tagged segment is scored independently. The total reward
    is the product of all scores. One zero -> total zero.
    """

    def __init__(
        self,
        checkpoint_tags: list[str] | None = None,
        checkpoint_scorer: Callable[[str, str | None], float] | None = None,
        min_checkpoints: int = 1,
        short_circuit_penalty: float = 0.0,
    ):
        self.checkpoint_tags = checkpoint_tags
        self.scorer = checkpoint_scorer or _default_scorer
        self.min_checkpoints = min_checkpoints
        self.short_circuit_penalty = short_circuit_penalty

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        if result.metadata.get("short_circuited"):
            return self.short_circuit_penalty

        tags = self.checkpoint_tags
        if tags is None:
            tags = sorted(set(s.tag for s in result.segments if s.tag.startswith("checkpoint_")))

        if len(tags) < self.min_checkpoints:
            return 0.0

        product = 1.0
        scored_count = 0
        for tag in tags:
            text = result.segment_text(tag)
            if not text.strip():
                continue
            score = self.scorer(text, answer)
            product *= score
            scored_count += 1

        if scored_count < self.min_checkpoints:
            return 0.0

        return product
