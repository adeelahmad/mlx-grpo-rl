"""Efficiency rewards — penalize verbosity and reward concise correct thinking.

EfficiencyReward:
  General verbosity penalty (repetition, filler words, thinking/answer ratio).

ThinkingEfficiencyReward:
  Additive bonus when the model gets the answer correct AND uses fewer thinking
  tokens than the budget. Optionally stacks a tiny beauty bonus for clean
  mathematical formatting — only when correct + highly efficient.
"""

from __future__ import annotations

import re
from typing import Any

from ..engine.result import RolloutResult, SegmentSource
from .base import BaseReward

_BOXED_RE = re.compile(r"\\boxed\{")
_LATEX_CMD_RE = re.compile(
    r"\\(?:frac|sqrt|cdot|times|pi|alpha|beta|gamma|theta|sigma|sum|int|lim|infty|leq|geq|neq)\b"
)
_UGLY_RE = re.compile(r"```|<\|im_|<s>|</s>|<pad>|\[INST\]")


def _format_beauty(text: str) -> float:
    """Score answer-text formatting quality in [0, 1].

    Checks: boxed content quality, LaTeX notation richness, clean whitespace,
    absence of tokenizer/chat-template artifacts.
    """
    if not text.strip():
        return 0.0
    score = 0.0
    # \boxed{} with non-trivial content (not just a single digit)
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        inner = m.group(1).strip()
        score += 0.45 if len(inner) > 1 else 0.25
    # Rich LaTeX notation (fractions, roots, Greek letters, etc.)
    n_cmds = len(_LATEX_CMD_RE.findall(text))
    score += min(0.3, n_cmds * 0.1)
    # Clean line structure (≤ 4 newlines — terse final answer, not a wall)
    if text.count("\n") <= 4:
        score += 0.15
    # No tokenizer / template artifacts
    if not _UGLY_RE.search(text):
        score += 0.1
    return min(1.0, score)


class EfficiencyReward(BaseReward, name="efficiency"):
    """Reward for output efficiency and conciseness."""

    def __init__(
        self,
        target_ratio: float = 0.3,
        max_ratio: float = 0.8,
        repetition_threshold: float = 0.3,
    ):
        self.target_ratio = target_ratio
        self.max_ratio = max_ratio
        self.repetition_threshold = repetition_threshold

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        text = result.completion_text
        if not text.strip():
            return 0.0

        score = 0.5  # Base

        # Thinking-to-answer ratio
        thinking = result.variables.get("thinking")
        answer_var = result.variables.get("answer") or result.variables.get("code")

        if thinking and answer_var:
            think_len = len(thinking.text)
            answer_len = max(len(answer_var.text), 1)
            ratio = answer_len / max(think_len + answer_len, 1)

            if ratio < 0.05:
                score -= 0.3  # Almost no answer
            elif ratio > self.max_ratio:
                score -= 0.1  # Too little thinking

        # N-gram repetition detection
        words = text.lower().split()
        if len(words) > 20:
            trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
            unique_ratio = len(set(trigrams)) / max(len(trigrams), 1)
            if unique_ratio < self.repetition_threshold:
                score -= 0.3  # Heavy repetition

        # Filler word penalty
        fillers = len(
            re.findall(
                r"\b(basically|essentially|literally|actually|really|very|just|quite)\b",
                text,
                re.IGNORECASE,
            )
        )
        filler_ratio = fillers / max(len(words), 1) if words else 0
        if filler_ratio > 0.05:
            score -= 0.1

        return max(0.0, min(1.0, score))


class ThinkingEfficiencyReward(BaseReward, name="thinking_efficiency"):
    r"""Additive bonus for correct answers that use fewer thinking tokens.

    Conditions for the efficiency bonus:
      1. Base accuracy reward must be > 0 (correct answer).
      2. A ``\boxed{}`` answer must appear in model-GENERATED text (not scaffold).
      3. Thinking token count must be >= ``min_thinking_tokens``.

    Efficiency bonus shape (linear by default, concave when decay < 1):
      ratio    = clamp(thinking_tokens / budget, 0, 1)
      eff_bonus = max_bonus * (1 - ratio) ** decay

    Beauty bonus (optional, stacked on top):
      Activated only when correct + ratio < efficiency_threshold (highly efficient).
      Scores the answer segment for LaTeX quality, clean structure, no artifacts.
      beauty_bonus controls the maximum additional reward (default 0.0 = disabled).

    Return value:
      base_score + eff_bonus [+ beauty_bonus * beauty_score]
      May exceed 1.0 — GRPO normalizes via per-group advantage computation.

    YAML example::

      reward: thinking_efficiency
      type_overrides:
        math:
          reward_kwargs:
            budget: 400
            max_bonus: 0.2
            min_thinking_tokens: 50
            decay: 0.7
            beauty_bonus: 0.05        # tiny stacked beauty reward
            efficiency_threshold: 0.5 # beauty fires only when ratio < 0.5
            base_reward: math
    """

    def __init__(
        self,
        budget: int = 512,
        min_thinking_tokens: int = 50,
        max_bonus: float = 0.2,
        decay: float = 1.0,
        base_reward: str = "math",
        beauty_bonus: float = 0.0,
        efficiency_threshold: float = 0.5,
        length_penalty_scale: float = 0.0,
        length_penalty_budget: int = 0,
        **kwargs: Any,
    ):
        self.budget = budget
        self.min_thinking_tokens = min_thinking_tokens
        self.max_bonus = max_bonus
        self.decay = decay
        self._base_reward_name = base_reward
        self._base_reward: BaseReward | None = None
        self.beauty_bonus = beauty_bonus
        self.efficiency_threshold = efficiency_threshold
        self.length_penalty_scale = length_penalty_scale
        self.length_penalty_budget = length_penalty_budget

    def _get_base(self) -> BaseReward:
        if self._base_reward is None:
            self._base_reward = BaseReward.get(self._base_reward_name)
        return self._base_reward

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        # ── Gate 1: correctness ──
        base_score = self._get_base().compute(result, answer)
        if base_score <= 0:
            return 0.0

        # ── Gate 2: answer must be model-generated (not from scaffold injection) ──
        # generated_text excludes INJECTED segments, so \boxed{} here = model wrote it.
        if not _BOXED_RE.search(result.generated_text):
            return base_score  # correct via scaffold only — no bonus

        # ── Count thinking tokens from GENERATED segments only ──
        thinking_tokens = sum(
            len(s.tokens)
            for s in result.segments
            if s.tag == "thinking" and s.source != SegmentSource.INJECTED
        )

        # ── Gate 3: minimum thinking required ──
        if thinking_tokens < self.min_thinking_tokens:
            return base_score

        # ── Efficiency bonus ──
        ratio = min(1.0, thinking_tokens / max(self.budget, 1))
        eff_bonus = self.max_bonus * (1.0 - ratio) ** self.decay
        total = base_score + eff_bonus

        # ── Beauty bonus: only when highly efficient (ratio < threshold) ──
        if self.beauty_bonus > 0.0 and ratio < self.efficiency_threshold:
            ans_segs = [s for s in result.segments if s.tag == "answer"]
            ans_text = "".join(s.text for s in ans_segs) if ans_segs else result.generated_text
            beauty = _format_beauty(ans_text)
            total += self.beauty_bonus * beauty

        # ── Length penalty: multiplicative penalty for exceeding token budget ──
        # penalty_budget defaults to thinking budget when not set.
        # excess = (total_tokens - budget) / budget; penalty = 1 - scale * excess.
        # scale=0.5: at 2x budget → 50% penalty; at 3x budget → zeroed.
        if self.length_penalty_scale > 0.0:
            lp_budget = self.length_penalty_budget or self.budget
            total_tokens = sum(
                len(s.tokens) for s in result.segments if s.source != SegmentSource.INJECTED
            )
            if total_tokens > lp_budget:
                excess = (total_tokens - lp_budget) / max(lp_budget, 1)
                total = total * max(0.0, 1.0 - self.length_penalty_scale * excess)

        return total
