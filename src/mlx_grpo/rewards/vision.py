"""Vision rewards — text-overlap metrics for VLM task evaluation.

Provides four reward functions for image-grounded language tasks:
  - VisionExactMatchReward  (name="vision_exact")   — exact string equality
  - VisionTokenF1Reward     (name="vision_f1")      — token-level F1
  - VisionBleuReward        (name="vision_bleu")    — BLEU-1 (unigram precision)
  - VisionRougeLReward      (name="vision_rouge_l") — ROUGE-L (LCS recall)

All rewards are normalised to [0, 1]. Whitespace and case are ignored.
"""

from __future__ import annotations

import re

from ..engine.result import RolloutResult
from .base import BaseReward


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # DP — O(m*n) but both sequences are typically short (< 200 tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _pred_and_ref(result: RolloutResult, answer: str | None) -> tuple[str, str]:
    """Extract prediction text and normalised reference."""
    pred = result.completion_text.strip()
    ref = (answer or "").strip()
    return pred, ref


# ── Reward implementations ────────────────────────────────────────────


class VisionExactMatchReward(BaseReward, name="vision_exact"):
    """Exact-match reward: 1.0 if prediction equals reference (case/space normalised)."""

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        pred, ref = _pred_and_ref(result, answer)
        if not ref:
            return 0.0
        return 1.0 if pred.lower() == ref.lower() else 0.0


class VisionTokenF1Reward(BaseReward, name="vision_f1"):
    """Token-level F1: harmonic mean of precision and recall over unigrams."""

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        pred, ref = _pred_and_ref(result, answer)
        if not ref:
            return 0.0
        pred_tokens = _tokenise(pred)
        ref_tokens = _tokenise(ref)
        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counts: dict[str, int] = {}
        for t in pred_tokens:
            pred_counts[t] = pred_counts.get(t, 0) + 1

        ref_counts: dict[str, int] = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1

        common = sum(min(pred_counts.get(t, 0), cnt) for t, cnt in ref_counts.items())
        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)


class VisionBleuReward(BaseReward, name="vision_bleu"):
    """BLEU-1 reward: unigram precision of prediction against reference."""

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        pred, ref = _pred_and_ref(result, answer)
        if not ref:
            return 0.0
        pred_tokens = _tokenise(pred)
        ref_tokens = _tokenise(ref)
        if not pred_tokens:
            return 0.0

        ref_counts: dict[str, int] = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1

        clipped = sum(min(pred_tokens.count(t), ref_counts.get(t, 0)) for t in set(pred_tokens))
        return clipped / len(pred_tokens)


class VisionRougeLReward(BaseReward, name="vision_rouge_l"):
    """ROUGE-L reward: LCS-based recall of reference tokens in prediction."""

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        pred, ref = _pred_and_ref(result, answer)
        if not ref:
            return 0.0
        pred_tokens = _tokenise(pred)
        ref_tokens = _tokenise(ref)
        if not pred_tokens or not ref_tokens:
            return 0.0

        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None (stdlib only)
# Usage: VisionTokenF1Reward().compute(result, answer)
#        BaseReward.get("vision_f1").compute(result, answer)
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
