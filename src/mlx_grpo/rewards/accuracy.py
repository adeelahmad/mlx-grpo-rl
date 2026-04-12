"""Accuracy reward — exact match with extraction strategies.

Handles multiple answer formats:
- Boxed: ``\\boxed{42}``
- MCQ:   ``(A)``, ``(B)``, etc.
- Exact: direct string match
- Numeric: float comparison with tolerance

    reward = AccuracyReward()
    score = reward.compute(result, answer="42")
"""

from __future__ import annotations

import re
from typing import Any

from ..engine.result import RolloutResult
from .base import BaseReward


class AccuracyReward(BaseReward, name="accuracy"):
    """Multi-strategy accuracy reward.

    Tries extraction in order: boxed → MCQ letters → numeric → exact.
    Supports multi-answer MCQ: \\boxed{ABC}, \\boxed{A,B,C}, \\boxed{A B C}.
    Returns 1.0 for match, 0.0 for mismatch.
    """

    # Letters A-H cover MCQ options up to 8 choices; guards against
    # matching words like "pi" or "no" as MCQ sets.
    _MCQ_SET_RE = re.compile(r"^[A-Ha-h](?:[,\s&/]*[A-Ha-h])*$")

    def __init__(self, tolerance: float = 1e-6, **kwargs: Any):
        self.tolerance = tolerance

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        if answer is None:
            return 0.0

        clean_answer = self._normalize_answer(answer.strip())

        # Best signal: the captured "answer" variable from structured rollouts
        # (e.g. MCQ rollout captures the generated answer after "The answer is")
        answer_var = result.variables.get("answer")
        if answer_var and answer_var.text.strip():
            av_text = answer_var.text.strip()
            if self._matches(av_text, clean_answer):
                return 1.0
            # Answer variable may contain the letter followed by verbose text
            # (e.g. " C.\n\nHere's my reasoning...") — run extractors on it
            for extract in (self._leading_letter, self._boxed, self._mcq_letter):
                extracted = extract(av_text)
                if extracted is not None and self._matches(extracted, clean_answer):
                    return 1.0

        # Fall back to extraction from model-generated text ONLY.
        # CRITICAL: must NOT use completion_text which includes injected scaffold
        # text — scaffold often contains the correct answer (\boxed{C}), giving
        # false-positive reward=1.0 for wrong model answers.
        gen_text = result.generated_text
        extractors = (
            self._boxed,
            self._mcq_letters_all,
            self._mcq_letter,
            self._leading_letter,
            self._last_number,
            self._exact,
        )
        for extract in extractors:
            extracted = extract(gen_text)
            if extracted is not None and self._matches(extracted, clean_answer):
                return 1.0

        return 0.0

    def _normalize_answer(self, answer: str) -> str:
        """Extract bare value from formatted ground truth (e.g. \\boxed{B} → B)."""
        for extract in (self._boxed, self._mcq_letter):
            extracted = extract(answer)
            if extracted is not None:
                return extracted
        return answer

    def _matches(self, extracted: str, answer: str) -> bool:
        """Compare extracted answer to ground truth."""
        e = extracted.strip()
        a = answer.strip()

        # Exact match (case-insensitive)
        if e.lower() == a.lower():
            return True

        # MCQ set comparison: A,B,C == ABC == A B C
        e_set = self._normalize_mcq_set(e)
        a_set = self._normalize_mcq_set(a)
        if e_set and a_set and e_set == a_set:
            return True

        # Numeric comparison
        try:
            return abs(float(e) - float(a)) < self.tolerance
        except (ValueError, OverflowError):
            return False

    # ── Extraction strategies ──

    @staticmethod
    def _boxed(text: str) -> str | None:
        """Extract from \\boxed{...}, handling nested braces like \\frac{}{}."""
        # Find the FIRST \\boxed{ and extract with balanced braces
        idx = text.find("\\boxed{")
        if idx == -1:
            return None
        start = idx + len("\\boxed{")
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
        return text[start:].strip() or None

    @staticmethod
    def _mcq_letters_all(text: str) -> str | None:
        """Extract all MCQ letters when multiple (X) patterns present."""
        matches = re.findall(r"\(([A-Ha-h])\)", text)
        if len(matches) >= 2:
            return "".join(sorted(set(m.upper() for m in matches)))
        return None

    @staticmethod
    def _mcq_letter(text: str) -> str | None:
        """Extract single MCQ letter answer: (A), (B), etc."""
        matches = re.findall(r"\(([A-Za-z])\)", text)
        return matches[-1].upper() if matches else None

    @classmethod
    def _normalize_mcq_set(cls, text: str) -> str | None:
        """Normalize multi-answer MCQ: 'A,B,C' / 'A B C' / 'ABC' → 'ABC'."""
        text = text.strip()
        if not text or not cls._MCQ_SET_RE.match(text):
            return None
        return "".join(sorted(set(c.upper() for c in text if c.isalpha())))

    @staticmethod
    def _leading_letter(text: str) -> str | None:
        """Extract a single MCQ letter from the start of text.

        Handles verbose model outputs after 'The answer is ' injection:
        ' C.\\nHere's my reasoning...' → 'C'
        ' (D). MERGE...' → 'D'
        '1B.\\nLet me think...' → 'B'  (digit prefix)
        ' **D**.\\n' → 'D'  (markdown bold)
        'C}' → 'C'  (\\boxed{} closure)
        """
        m = re.match(
            r"\s*(?:\d+[.)\s]*)?\*{0,2}\(?([A-Ha-h])\)?\*{0,2}(?:[\s.,;:)\-}]|$)",
            text,
        )
        return m.group(1).upper() if m else None

    @staticmethod
    def _last_number(text: str) -> str | None:
        """Extract the last number in the text."""
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        return matches[-1] if matches else None

    @staticmethod
    def _exact(text: str) -> str | None:
        """Last non-empty line as answer."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else None


class FormatReward(BaseReward, name="format"):
    """Reward for structural format compliance.

    Checks for required tags, forbidden patterns, length bounds.
    """

    def __init__(
        self,
        required_tags: list[str] | None = None,
        forbidden: list[str] | None = None,
        max_length: int | None = None,
        **kwargs: Any,
    ):
        self.required_tags = required_tags or []
        self.forbidden = forbidden or []
        self.max_length = max_length

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        text = result.completion_text
        score = 1.0
        n_checks = max(
            1,
            len(self.required_tags) + len(self.forbidden) + (1 if self.max_length else 0),
        )

        for tag in self.required_tags:
            if tag not in text:
                score -= 1.0 / n_checks

        for pattern in self.forbidden:
            if pattern in text:
                score -= 1.0 / n_checks

        if self.max_length and len(text) > self.max_length:
            score -= 1.0 / n_checks

        return max(0.0, score)
