"""Gaming detection, penalties, and reward smoothing for exam rewards.

Ported from exam_reward.py — detects hedging, filler, repetition, garbage,
inconsistency, and applies length/bad-word/repeat-word penalties.

Penalty thresholds tuned from 1396-sample rollout analysis:
- boxed_hedging: strong (0.2% correct vs 11.1% incorrect)
- inconsistency: strong (0% correct vs 0.1% incorrect)
- multiple_boxed: slight (14.6% vs 16.7%)
- answer_shopping: DISABLED (hits correct more)
- answer_repetition: DISABLED (hits correct 32x more)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

# Pre-compiled patterns for gaming detection
_HEDGING = [
    re.compile(r"\b[A-D]\s+(or|and)\s+[A-D]\b", re.I),
    re.compile(r"\b[A-D]\s*/\s*[A-D]\b", re.I),
    re.compile(r"\b[A-D]\s*,\s*[A-D]\b", re.I),
    re.compile(r"\b(either|both)\b", re.I),
]
_STRONG_CONCLUSION = [
    re.compile(
        r"(the\s+answer\s+is|correct\s+answer\s+is|therefore|thus|hence"
        r"|so\s+the\s+answer)\s+[\"']?([A-D])[\"']?",
        re.I,
    ),
    re.compile(
        r"([A-D])\s+(is\s+correct|is\s+the\s+(right|correct)"
        r"\s+(answer|choice))",
        re.I,
    ),
]
_HEDGE_WORDS = frozenset(
    [
        "maybe",
        "possibly",
        "perhaps",
        "might",
        "probably",
        "likely",
        "unlikely",
        "seems",
        "appears",
        "somewhat",
        "fairly",
        "arguably",
    ]
)
_FILLER_PHRASES = [
    "let me think",
    "let me analyze",
    "let me consider",
    "i need to think",
    "this is a great question",
    "this is an interesting question",
    "great question",
    "step by step",
    "one by one",
    "carefully consider",
    "let's break this down",
    "let's analyze",
    "let's think",
    "thinking about this",
    "hmm",
    "hmmm",
    "well let me",
    "okay so",
    "ok so",
    "alright",
    "interesting",
    "i see",
]
_BAD_WORDS = frozenset(
    [
        "obviously",
        "clearly",
        "simply",
        "basically",
        "essentially",
    ]
)
_REPEAT_IGNORE = frozenset(
    [
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "been",
        "will",
        "would",
        "could",
        "should",
        "also",
        "just",
        "more",
        "most",
        "some",
        "other",
        "than",
        "very",
        "option",
        "answer",
        "question",
        "correct",
        "using",
        "used",
        "data",
        "service",
        "need",
        "provide",
        "which",
        "what",
        "when",
        "where",
        "there",
        "their",
        "these",
        "those",
        "about",
    ]
)


def compute_gaming_penalties(
    completion: str,
    thinking: str,
    boxed_answer: str | None,
) -> tuple[float, dict[str, Any]]:
    """Detect and penalize gaming behaviors. Returns (penalty, details)."""
    penalties: dict[str, float] = {}
    comp_lower = completion.lower()
    think_lower = thinking.lower() if thinking else ""

    # 1. Multiple boxed answers (hedging)
    boxed_all = re.findall(r"\\boxed\{[^}]*\}", completion)
    if len(boxed_all) > 1:
        penalties["multiple_boxed"] = 0.25

    # 2. Hedge words (excess beyond threshold 5)
    if think_lower:
        words = comp_lower.split()
        hc = sum(1 for w in words if w.strip(".,!?;:()[]") in _HEDGE_WORDS)
        if hc > 5:
            penalties["hedge_words"] = min((hc - 5) * 0.02, 0.12)

    # 3. Repetitive n-gram phrases
    if think_lower:
        words = think_lower.split()
        if len(words) >= 4:
            ngrams: list[str] = []
            for n in range(4, min(6, len(words))):
                for i in range(len(words) - n + 1):
                    ng = " ".join(words[i : i + n])
                    if len(ng) > 10:
                        ngrams.append(ng)
            repeated = {ng: c for ng, c in Counter(ngrams).items() if c >= 4}
            if repeated:
                penalties["repetitive_phrases"] = min(len(repeated) * 0.05, 0.15)

    # 4. Garbage characters
    garbage = re.findall(
        r"[^\x20-\x7E\n\r\t\u00A0-\u00FF\u2018-\u201F\u2013-\u2014\u2026]",
        completion,
    )
    if len(garbage) > 5:
        penalties["garbage_chars"] = min((len(garbage) - 5) * 0.04, 0.20)

    # 5. Filler/fluff phrases (excess beyond threshold 4)
    if think_lower:
        fc = sum(think_lower.count(p) for p in _FILLER_PHRASES)
        if fc > 4:
            penalties["filler_phrases"] = min((fc - 4) * 0.04, 0.12)

    # 6. Boxed answer hedging (e.g., \boxed{A or B})
    if boxed_answer:
        for pat in _HEDGING:
            if pat.search(boxed_answer.strip()):
                penalties["boxed_hedging"] = 0.40
                break

    # 7. Low entropy text (very repetitive/garbage)
    if think_lower:
        words = think_lower.split()
        if len(words) >= 10:
            ratio = len(set(words)) / len(words)
            if ratio < 0.20:
                penalties["low_entropy"] = 0.15

    # 8. Reasoning-answer inconsistency
    if boxed_answer and think_lower:
        ans_up = boxed_answer.strip().upper()
        if len(ans_up) == 1 and ans_up in "ABCD":
            for opt in (o for o in "ABCD" if o != ans_up):
                for pat in _STRONG_CONCLUSION:
                    m = pat.search(think_lower)
                    if m and opt in m.group(0).upper():
                        penalties["inconsistency"] = 0.20
                        break
                if "inconsistency" in penalties:
                    break

    total = min(1.0, sum(penalties.values()))
    return total, {"total": total, "breakdown": penalties}


def compute_length_penalty(thinking_words: int, answer_words: int) -> float:
    """Soft length penalty. Min thinking: 20, max: 250, max answer: 80."""
    pens: list[float] = []
    if thinking_words < 20 and thinking_words >= 0:
        pens.append(1.0 - thinking_words / 20)
    if thinking_words > 250:
        pens.append(min(1.0, (thinking_words - 250) / 250))
    if answer_words > 80:
        pens.append(min(1.0, (answer_words - 80) / 80))
    return 0.10 * (sum(pens) / len(pens)) if pens else 0.0


def compute_bad_words_penalty(text: str) -> float:
    """Penalize filler words: obviously, clearly, simply, basically, essentially."""
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    total = sum(1 for w in words if w.lower() in _BAD_WORDS)
    return min(total * 0.025, 0.15) if total else 0.0


def compute_repeat_words_penalty(text: str) -> float:
    """Penalize excessively repeated words (threshold=4, min length=4)."""
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    counts: dict[str, int] = {}
    for w in words:
        low = w.lower()
        if len(low) >= 4 and low not in _REPEAT_IGNORE:
            counts[low] = counts.get(low, 0) + 1
    excess = sum(max(0, c - 4) for c in counts.values())
    return min(excess * 0.02, 0.20) if excess else 0.0


def smooth_reward(
    reward: float,
    method: str = "sigmoid",
    temp: float = 4.0,
    center: float = 0.5,
) -> float:
    """Apply reward smoothing: none, clip, sigmoid, or tanh."""
    if method == "none":
        return reward
    if method == "clip":
        return max(0.0, min(1.0, reward))
    if method == "sigmoid":
        x = (reward - center) * temp
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
    if method == "tanh":
        x = (reward - center) * temp
        return 0.5 * (math.tanh(x) + 1.0)
    return reward
