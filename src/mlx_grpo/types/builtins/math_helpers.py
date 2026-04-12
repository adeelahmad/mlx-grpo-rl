"""Shared math utilities — extracted from math.py to stay under 200-line limit."""

from __future__ import annotations

import re
from typing import Any


def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    start = text.find("\\boxed{")
    if start == -1:
        return None
    start += len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    # No closing brace found
    return text[start:].strip()


def _extract_last_boxed(text: str) -> str | None:
    """Extract content from the LAST \\boxed{...}, handling nested braces.

    Use for completion_text when the injected scaffold prefix may contain
    intermediate \\boxed{} answers from the reference thinking trace.
    """
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    start += len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return text[start:].strip()


def _normalize_math_answer(text: str) -> str:
    """Normalize a math answer for comparison."""
    text = str(text).strip()
    # Remove LaTeX formatting
    text = text.replace("\\$", "").replace("$", "")
    # Strip font/style commands: \textbf{X} → X, \mathbf{X} → X, \mathrm{X} → X, etc.
    text = re.sub(
        r"\\(?:text(?:bf|it|rm|tt|sf)?|mathrm|mathbf|mathit|mathcal|mathbb|mathtt|boldsymbol)"
        r"\{([^}]*)\}",
        r"\1",
        text,
    )
    text = text.replace("\\dfrac", "frac")
    text = text.replace("\\frac", "frac")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = text.replace("\\div", "/")
    text = text.replace("\\%", "%")
    text = text.replace("\\pi", "pi")
    # Normalize infinity variants: +\infty → \infty, +inf → inf
    text = re.sub(r"^\+\\infty\b", r"\\infty", text)
    text = re.sub(r"^\+inf(inity)?$", "inf", text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = " ".join(text.split())
    # Strip surrounding parens from single MCQ letters: (B) → B
    # Covers AMC-style answers like \textbf{(B)} which normalize to (B).
    _paren = re.match(r"^\(([A-Ha-h])\)$", text)
    if _paren:
        text = _paren.group(1).upper()
    # Evaluate simple numeric fractions: frac{99}{100} → "0.99", frac{3}{4} → "0.75"
    # This lets decimal targets match LaTeX fraction predictions and vice-versa.
    _m = re.match(r"^frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$", text)
    if _m:
        try:
            num, den = float(_m.group(1)), float(_m.group(2))
            if den != 0:
                text = str(num / den)
        except (ValueError, ZeroDivisionError):
            pass
    return text


def _compute_thinking_budget(
    cfg: dict[str, Any],
    answer: str,
    prompt: str,
    max_completion_tokens: int,
    is_scaffolded: bool,
) -> tuple[int, int]:
    """Returns (thinking_budget, bonus_budget).

    Handles multi-part scaling, recovery bonus reservation, and completion cap.
    """
    thinking_budget = cfg.get("thinking_budget", 2048)
    _num_parts = _count_answer_parts(answer, prompt)
    if _num_parts > 1:
        thinking_budget = min(
            int(thinking_budget * (1.0 + (_num_parts - 1) * 0.5)),
            cfg.get("thinking_budget_ceiling", 4096),
        )
    bonus_budget = cfg.get("recovery_bonus_tokens", 0) if is_scaffolded else 0
    if bonus_budget > 0:
        thinking_budget = max(256, thinking_budget - bonus_budget)
    answer_reserve = cfg.get("answer_max_tokens", 128)
    if max_completion_tokens > 0:
        floor = max(512, int(max_completion_tokens * 0.75))
        ceiling = max(floor, max_completion_tokens - answer_reserve)
        thinking_budget = min(thinking_budget, ceiling)
    return thinking_budget, bonus_budget


def _count_answer_parts(target_answer: str, prompt: str) -> int:
    """Count answer parts for budget scaling. 'a=36,b=18' → 2, '1,000,000' → 1."""
    count_a = 1
    # Letters/= adjacent to comma → multi-part (not formatted number like 1,000,000)
    if "," in target_answer and re.search(r"[a-zA-Z=]\s*,|,\s*[a-zA-Z=]", target_answer):
        count_a = target_answer.count(",") + 1

    patterns = [
        r"[A-Z]\d+\.\d+",  # G7.1, G7.2
        r"\([a-z]\)",  # (a), (b)
        r"(?:^|\s)[a-z]\)",  # a), b)
        r"\([ivx]+\)",  # (i), (ii)
        r"[Pp]art\s+[a-z0-9]+",  # Part 1, Part a
        r"(?:^|\n)\s*\d+\.",  # 1. 2. at line start
    ]
    count_b = max((len(re.findall(p, prompt)) for p in patterns), default=0) if prompt else 0
    return max(count_a, max(count_b, 1))


def _validate_think_tags(text: str) -> str:
    """Enforce single <think>...</think> block — no reopening.

    Rules:
    - Model may close </think> once during generation
    - No reopening <think> after closing
    - System may add closing </think> if missing after token budget

    Returns: text with only the FIRST complete think block preserved
    """
    # Find the first </think>
    close_idx = text.find("</think>")
    if close_idx == -1:
        # No closing tag found, return as-is (system will add closing)
        return text

    # Everything up to and including the first </think>
    valid_part = text[:close_idx + len("</think>")]

    # Check if anything after </think> tries to reopen <think>
    after_close = text[close_idx + len("</think>"):]
    if "<think>" in after_close:
        # Model tried to reopen — drop everything after first </think>
        return valid_part

    # Also strip any leftover text that looks like it's trying to restart thinking
    if re.search(r"</?think>", after_close):
        # Any think-related tags after closing = exploit attempt
        return valid_part

    return text


def _validate_boxed_answers(text: str) -> str:
    """Keep only the FIRST boxed answer, drop subsequent ones.

    Rules:
    - Only one \\boxed{...} answer allowed
    - If model generates multiple, keep first and drop rest
    - If model tries to hack with malformed boxes, clean up

    Returns: text with only the first complete boxed answer
    """
    # Find first boxed opening
    first_open = text.find("\\boxed{")
    if first_open == -1:
        return text

    # Find the closing brace for the first boxed answer
    start = first_open + len("\\boxed{")
    depth = 1
    close_pos = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                close_pos = i
                break

    if close_pos == -1:
        # Unclosed brace — keep until end (will be closed by system)
        return text

    # Everything up to and including the first complete boxed answer
    valid_part = text[:close_pos + 1]

    # Check if there's another \boxed{ after the first one
    after_first = text[close_pos + 1:]
    if "\\boxed{" in after_first:
        # Multiple boxed answers — drop everything after first
        return valid_part

    return text


def _detect_token_repetition_hack(text: str, min_repeat_len: int = 3) -> bool:
    """Detect obvious token repetition hacking attempts.

    Returns True if suspicious repetition pattern detected (e.g., "blah blah blah...")

    Heuristic: if the same word sequence appears 3+ times consecutively,
    it's likely a hack to game token/word budgets while producing low-quality output.
    """
    words = text.split()
    if len(words) < min_repeat_len:
        return False

    # Look for repeated 1-3 word chunks appearing 3+ times in a row
    for chunk_size in [1, 2, 3]:
        if len(words) < chunk_size * 3:
            continue
        for i in range(len(words) - chunk_size * 3 + 1):
            chunk = " ".join(words[i:i + chunk_size])
            # Count consecutive repeats of this chunk
            repeats = 1
            j = i + chunk_size
            while j + chunk_size <= len(words) and " ".join(words[j:j + chunk_size]) == chunk:
                repeats += 1
                j += chunk_size

            # If same chunk repeats 3+ times consecutively, it's likely spam
            # Even a single word "answer answer answer" is suspicious
            if repeats >= 3:
                return True

    return False
