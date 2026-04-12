"""Exam reward — composite reward for MCQ/exam questions.

Weighted components: accuracy (75%), boxed (2%), format (3%), reasoning (20%).
Features: format gate, gaming detection, length/bad-word/repeat penalties,
sigmoid smoothing. New opt-in flags: require_accuracy, multiplicative_accuracy,
accuracy_weight_override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..engine.result import RolloutResult
from .base import BaseReward
from .exam_extract import (
    extract_answer_from_completion,
    extract_boxed_answer,
    normalize_answer,
    normalize_ground_truth,
)
from .exam_format import (
    check_boxed_format,
    check_format_compliance,
    check_reasoning_quality,
)
from .exam_gaming import (
    compute_bad_words_penalty,
    compute_gaming_penalties,
    compute_length_penalty,
    compute_repeat_words_penalty,
    smooth_reward,
)


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Component weights. Tuned: accuracy=0.824 gap, others mastered (<0.03)."""

    accuracy: float = 0.75
    boxed: float = 0.02
    format: float = 0.03
    reasoning: float = 0.20

    @property
    def total(self) -> float:
        return self.accuracy + self.boxed + self.format + self.reasoning


# Leading-letter extraction for answer variable text
_LEADING_RE = re.compile(
    r"\s*(?:\d+[.)\s]*)?\*{0,2}\(?([A-Ha-h])\)?\*{0,2}(?:[\s.,;:)\-}]|$)",
)


class ExamReward(BaseReward, name="exam"):
    """Composite exam reward: accuracy → format → gaming → penalties → sigmoid."""

    def __init__(
        self,
        weights: RewardWeights | None = None,
        require_accuracy: bool = False,
        wrong_max: float = 0.0,
        multiplicative_accuracy: bool = False,
        accuracy_weight_override: float | None = None,
        **kwargs: Any,
    ):
        self.weights = weights or RewardWeights()
        if accuracy_weight_override is not None:
            from dataclasses import replace as _dr

            self.weights = _dr(self.weights, accuracy=accuracy_weight_override)
        self._require_accuracy = require_accuracy
        self._wrong_max = wrong_max
        self._multiplicative_accuracy = multiplicative_accuracy

    def compute(
        self,
        result: RolloutResult,
        answer: str | None = None,
    ) -> float:
        completion = result.completion_text

        # Component scores
        acc = self._accuracy(result, answer)

        # Hard gate: wrong answer skips all other computation
        if self._require_accuracy and acc == 0.0:
            return self._wrong_max

        # Phrase-mode gate: non-MCQ targets (e.g. "Apex-R1") should not get
        # partial reward from format/reasoning when the answer is wrong.
        if acc == 0.0 and answer is not None:
            _norm_gt = normalize_ground_truth(answer)
            if not (_norm_gt and re.match(r"^[A-F]{1,4}$", _norm_gt)):
                return self._wrong_max

        boxed_s, _ = check_boxed_format(completion)
        fmt_s, _ = check_format_compliance(completion)
        reas_s, r_det = check_reasoning_quality(completion)

        # Format gate: penalize accuracy if format is poor
        combined_fmt = (boxed_s + fmt_s + reas_s) / 3.0
        gated_acc = acc * 0.3 if combined_fmt < 0.5 else acc

        len_p = compute_length_penalty(r_det.get("thinking_words", 0), r_det.get("answer_words", 0))
        bad_p = compute_bad_words_penalty(completion)
        rep_p = compute_repeat_words_penalty(completion)
        thinking_text = ""
        for tag in ("think", "thinking"):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", completion, re.I | re.DOTALL)
            if m:
                thinking_text = m.group(1)
                break
        game_p, _ = compute_gaming_penalties(
            completion, thinking_text, extract_boxed_answer(completion)
        )
        pen_mult = max(0.0, 1.0 - (len_p + bad_p + rep_p + game_p))

        w = self.weights
        if self._multiplicative_accuracy:
            raw = acc * (w.boxed * boxed_s + w.format * fmt_s + w.reasoning * reas_s) / w.total
        else:
            raw = (
                w.accuracy * gated_acc + w.boxed * boxed_s + w.format * fmt_s + w.reasoning * reas_s
            ) / w.total
        penalized = raw * pen_mult
        clamped = max(0.0, min(1.0, penalized))
        return smooth_reward(clamped, method="sigmoid", temp=4.0, center=0.5)

    # ── Accuracy computation ──

    def _accuracy(self, result: RolloutResult, answer: str | None) -> float:
        if answer is None:
            return 0.0
        gt_str = str(answer)
        _norm_gt = normalize_ground_truth(answer)
        # Short MCQ letters (A-F, 1-4 chars) → exam mode; phrases → phrase mode.
        is_exam = bool(_norm_gt and re.match(r"^[A-F]{1,4}$", _norm_gt))
        clean_gt = _norm_gt if is_exam else gt_str.strip()

        av = result.variables.get("answer")
        if av and av.text.strip():
            model_ans = self._extract_var(av.text.strip(), is_exam)
            if model_ans and self._matches(model_ans, clean_gt, is_exam):
                return 1.0

        model_ans = extract_answer_from_completion(result.generated_text, for_exam=is_exam)
        if model_ans is None:
            model_ans = extract_boxed_answer(result.generated_text)
        if model_ans is None:
            return 0.0
        if self._matches(model_ans, clean_gt, is_exam):
            return 1.0

        # Partial credit: F1 for multi-answer MCQ only
        if is_exam and len(clean_gt) > 1:
            mn = normalize_answer(model_ans)
            m_set, c_set = set(mn), set(clean_gt)
            inter = len(m_set & c_set)
            if inter > 0 and m_set and c_set:
                p = inter / len(m_set)
                r = inter / len(c_set)
                if p + r > 0:
                    return 2 * p * r / (p + r)
        return 0.0

    def _extract_var(self, text: str, for_exam: bool) -> str | None:
        if not for_exam:
            return text.strip() or None
        m = _LEADING_RE.match(text)
        if m:
            return m.group(1).upper()
        boxed = extract_boxed_answer(text)
        if boxed:
            return normalize_answer(boxed)
        return extract_answer_from_completion(text, for_exam=True)

    @staticmethod
    def _matches(extracted: str, answer: str, is_exam: bool) -> bool:
        e, a = extracted.strip(), answer.strip()
        if e.lower() == a.lower():
            return True
        if is_exam:
            en, an = normalize_answer(e), normalize_answer(a)
            if en and an and en == an:
                return True
        else:
            el, al = e.lower(), a.lower()
            if al and (al in el or el in al):
                return True
        try:
            if abs(float(e) - float(a)) < 1e-6:
                return True
        except (ValueError, OverflowError):
            pass
        return False
