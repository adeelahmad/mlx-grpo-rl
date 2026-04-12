"""Format compliance & reasoning quality checks for exam rewards.

Ported from exam_reward.py — checks ``<think>`` tags, ``\\boxed{}`` format,
proper ordering, and reasoning quality indicators.
"""

from __future__ import annotations

import re
from typing import Any

_THINK_TAGS = ("think", "thinking")
_ANALYSIS_KW = (
    "because",
    "since",
    "therefore",
    "thus",
    "means",
    "implies",
    "incorrect",
    "correct",
)


def check_boxed_format(completion: str) -> tuple[float, dict[str, Any]]:
    """Check if completion uses ``\\boxed{}`` format.

    Returns: 1.0 (valid content), 0.5 (empty boxed), 0.0 (no boxed).
    """
    if not isinstance(completion, str) or not completion:
        return 0.0, {"has_boxed": False}
    m = re.search(r"\\boxed\{([^}]*)\}", completion, re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        if content:
            return 1.0, {
                "has_boxed": True,
                "boxed_content": content,
                "valid": True,
                "has_letters": bool(re.findall(r"[A-Za-z]", content)),
                "has_numeric": bool(re.search(r"[\d.\-+]", content)),
            }
        return 0.5, {
            "has_boxed": True,
            "boxed_content": "",
            "valid": False,
            "reason": "empty_content",
        }
    return 0.0, {"has_boxed": False, "reason": "no_boxed_found"}


def check_format_compliance(completion: str) -> tuple[float, dict[str, Any]]:
    """Check ``<think>...</think>`` format structure.

    Scoring: think tags (0.5) + proper order (0.3) + think bonus (0.2).
    """
    if not isinstance(completion, str) or not completion:
        return 0.0, {"has_think_tags": False}
    lower = completion.lower()
    think_valid = False
    think_tag: str | None = None
    for tag in _THINK_TAGS:
        if f"<{tag}>" in lower and f"</{tag}>" in lower:
            think_valid = True
            think_tag = tag
            break
    proper_order = True
    if think_valid and think_tag:
        close_pos = lower.rfind(f"</{think_tag}>")
        boxed_pos = lower.find("\\boxed{")
        if boxed_pos != -1 and close_pos > boxed_pos:
            proper_order = False
    score = 0.0
    if think_valid:
        score += 0.5
    if proper_order and think_valid:
        score += 0.3
    if think_valid:
        score += 0.2
    return score, {
        "has_think_tags": think_valid,
        "think_tag_found": think_tag,
        "proper_order": proper_order,
        "score": score,
    }


def check_reasoning_quality(completion: str) -> tuple[float, dict[str, Any]]:
    """Check thinking section quality: word count, analysis keywords, options.

    Returns score in [0, 1] and details with thinking_words, answer_words.
    """
    empty = {"has_thinking": False, "thinking_words": 0, "answer_words": 0}
    if not isinstance(completion, str) or not completion:
        return 0.0, empty
    thinking = ""
    for tag in _THINK_TAGS:
        m = re.search(
            rf"<{tag}>(.*?)</{tag}>",
            completion,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            thinking = m.group(1).strip()
            break
    # Count answer words (text after boxed)
    answer_words = 0
    boxed_m = re.search(r"\\boxed\{[^}]*\}", completion.lower())
    if boxed_m:
        after = completion[boxed_m.end() :].strip()
        answer_words = len(after.split()) if after else 0
    if not thinking:
        return 0.3, {
            "has_thinking": False,
            "thinking_words": 0,
            "answer_words": answer_words,
        }
    words = thinking.split()
    wc = len(words)
    lower = thinking.lower()
    has_analysis = any(kw in lower for kw in _ANALYSIS_KW)
    has_options = bool(
        re.search(r"(option [a-z]|choice [a-z]|\b[a-z]\)|\b[a-z]\.)", lower),
    )
    score = 0.0
    if wc >= 50:
        score += 0.4
    elif wc >= 30:
        score += 0.3
    elif wc >= 15:
        score += 0.2
    elif wc >= 5:
        score += 0.1
    if has_analysis:
        score += 0.3
    if has_options:
        score += 0.3
    return min(score, 1.0), {
        "has_thinking": True,
        "thinking_words": wc,
        "answer_words": answer_words,
        "has_analysis": has_analysis,
        "has_options": has_options,
    }
