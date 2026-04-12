"""TattsLotto reward — multi-component scoring for lottery prediction.

Components (weights normalised when supplementary absent):
  value_match (30%)      — predicted numbers in ground truth
  positional_match (25%) — numbers correct AND in right sorted position
  near_miss (15%)        — numbers within +/-2 of ground truth
  supplementary (10%)    — bonus for matching supplementary numbers
  format_quality (20%)   — proper <think>...</think> + \\boxed{} structure

Hard zeros for: missing boxed, wrong count, out-of-range, duplicates.
Degenerate caps for: consecutive, arithmetic, clustered, lazy patterns.
Format gating: format score * min(1, value_matches / 2).
"""

from __future__ import annotations

import re

from ..engine.result import RolloutResult
from .base import BaseReward

_RE_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_RE_NUM = re.compile(r"\d+")
_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

_WEIGHTS = {"value": 0.30, "positional": 0.25, "near": 0.15, "supp": 0.10, "fmt": 0.20}
_VAL_MAP = {0: 0.0, 1: 0.15, 2: 0.35, 3: 0.55, 4: 0.75, 5: 0.90, 6: 1.0}
_POS_MAP = {0: 0.0, 1: 0.20, 2: 0.45, 3: 0.65, 4: 0.80, 5: 0.92, 6: 1.0}

_LAZY = [
    [1, 2, 3, 4, 5, 6],
    [5, 10, 15, 20, 25, 30],
    [7, 14, 21, 28, 35, 42],
    [1, 11, 21, 31, 41, 45],
    [40, 41, 42, 43, 44, 45],
]


class LottoReward(BaseReward, name="lotto"):
    """Reward for TattsLotto prediction tasks."""

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        text = result.completion_text
        if not text or not answer:
            return 0.0

        pred = _extract_pred(text)
        gt = _parse_nums(answer)
        if not gt:
            m = _RE_BOXED.search(answer)
            gt = _parse_nums(m.group(1)) if m else []

        # Hard zero checks
        if _hard_zero(text, pred):
            return 0.0

        pos_gt = result.metadata.get("_positional_ground_truth", gt)
        supp_gt = result.metadata.get("_supp_ground_truth", [])
        has_supp = bool(supp_gt) and any(1 <= s <= 45 for s in supp_gt)

        val_matches = len(set(pred[:6]) & set(gt[:6]))
        scores = {
            "value": _VAL_MAP.get(val_matches, 0.0),
            "positional": _score_positional(pred, pos_gt),
            "near": _score_near(pred, gt),
            "supp": _score_supp(pred, supp_gt),
            "fmt": _score_format(text) * min(1.0, val_matches / 2.0),
        }

        # Gaming penalty
        pen = _gaming_penalty(text, pred)
        if pen > 0:
            scores["value"] = max(0.0, scores["value"] - pen)

        # Weighted sum with dynamic normalisation
        total = _combine(scores, has_supp)

        # Degenerate cap
        cap = _degen_cap(pred)
        if cap is not None:
            total = min(total, cap)

        return max(0.0, min(1.0, total))


def _parse_nums(text: str) -> list[int]:
    nums = [int(n) for n in _RE_NUM.findall(text)]
    return sorted(set(n for n in nums if 1 <= n <= 45))[:6]


def _extract_pred(text: str) -> list[int]:
    m = _RE_BOXED.search(text)
    return _parse_nums(m.group(1)) if m else _parse_nums(text)


def _hard_zero(text: str, pred: list[int]) -> bool:
    m = _RE_BOXED.search(text)
    if not m:
        return True
    raw = [int(n) for n in _RE_NUM.findall(m.group(1))]
    if len(raw) != 6:
        return True
    if any(n < 1 or n > 45 for n in raw):
        return True
    return len(set(raw)) != 6


def _score_positional(pred: list[int], gt: list[int]) -> float:
    if not pred or not gt:
        return 0.0
    matches = sum(1 for p, g in zip(pred[:6], gt[:6], strict=False) if p == g)
    return _POS_MAP.get(matches, 0.0)


def _score_near(pred: list[int], gt: list[int], radius: int = 2) -> float:
    if not pred or not gt:
        return 0.0
    gt_set = set(gt[:6])
    credit = 0.0
    for p in pred[:6]:
        if p in gt_set:
            credit += 1.0
        elif any(0 < abs(p - g) <= radius for g in gt[:6]):
            credit += 0.5
    return min(1.0, credit / 6.0)


def _score_supp(pred: list[int], supp: list[int]) -> float:
    valid = [s for s in supp if 1 <= s <= 45]
    if not valid or not pred:
        return 0.0
    return min(1.0, len(set(pred[:6]) & set(valid)) * 0.5)


def _score_format(text: str) -> float:
    s = 0.0
    if "<think>" in text:
        s += 0.15
    if "</think>" in text:
        s += 0.15
    m = _RE_THINK.search(text)
    if m and len(m.group(1).strip()) >= 10:
        s += 0.20
    close = text.find("</think>")
    if close >= 0 and len(text[close + 8 :].strip()) >= 2:
        s += 0.15
    if "\\boxed{" in text:
        s += 0.20
    bm = _RE_BOXED.search(text)
    if bm and len(text[bm.end() :].strip()) > 10:
        s -= 0.15
    if re.search(r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$", text.strip()):
        s += 0.15
    return max(0.0, min(1.0, s))


def _gaming_penalty(text: str, pred: list[int]) -> float:
    pen = 0.0
    if re.search(r"\\boxed\{.*\\boxed\{", text, re.DOTALL):
        pen += 0.3
    if re.search(r"\b(maybe|perhaps|could be|might be)\b", text, re.IGNORECASE):
        pen += 0.3
    for tok in ("<|endoftext|>", "<|im_start|>", "<think><think>", "<|im_end|>"):
        if tok in text:
            pen += 0.3
    if len(pred) >= 6:
        nums = pred[:6]
        if nums == list(range(nums[0], nums[0] + 6)):
            pen += 0.5
        if max(nums) - min(nums) < 10:
            pen += 0.25
        if all(n <= 10 for n in nums) or all(n >= 35 for n in nums):
            pen += 0.35
        if nums in _LAZY:
            pen += 0.4
    return min(1.0, pen)


def _combine(scores: dict[str, float], has_supp: bool) -> float:
    applicable = {k: v for k, v in _WEIGHTS.items() if k != "supp" or has_supp}
    total_w = sum(applicable.values())
    return sum(scores.get(k, 0.0) * (w / total_w) for k, w in applicable.items())


def _degen_cap(pred: list[int]) -> float | None:
    if len(pred) < 6:
        return None
    nums = sorted(pred[:6])
    pairs = sum(1 for i in range(5) if nums[i + 1] - nums[i] == 1)
    if pairs == 5:
        return 0.02
    if pairs >= 4:
        return 0.05
    diffs = [nums[i + 1] - nums[i] for i in range(5)]
    if len(set(diffs)) == 1 and diffs[0] <= 10:
        return 0.08
    if all(n <= 15 for n in nums) or all(n >= 30 for n in nums):
        return 0.10
    if nums[-1] - nums[0] < 10:
        return 0.12
    if nums in _LAZY:
        return 0.05
    return None
