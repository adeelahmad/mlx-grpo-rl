"""Answer extraction & normalization for exam/MCQ rewards."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

# Common short words excluded from standalone letter extraction
_COMMON_WORDS = {
    "a",
    "i",
    "is",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "or",
    "so",
    "to",
    "up",
    "us",
    "we",
    "an",
    "am",
}


def _parse_numeric(value: str) -> float | None:
    """Parse string as numeric: int, float, sci notation, fraction, percent."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if v.endswith("%"):
        try:
            return float(v[:-1]) / 100
        except ValueError:
            pass
    try:
        return float(v)
    except ValueError:
        pass
    try:
        if "/" in v:
            return float(Fraction(v))
    except (ValueError, ZeroDivisionError):
        pass
    return None


def normalize_answer(answer: str) -> str:
    """Normalize exam answer to uppercase sorted letters A-F.

    Preserves numeric/fraction answers as-is.
    Examples: ``"B"`` → ``"B"``, ``"A, C"`` → ``"AC"``, ``"42"`` → ``"42"``
    """
    if not answer:
        return ""
    if not isinstance(answer, str):
        return str(answer)
    stripped = answer.strip()
    if re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", stripped):
        return stripped
    if re.match(r"^[-+]?\d+/\d+$", stripped):
        return stripped
    cleaned = re.sub(r"\b(and|or)\b", " ", answer, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(options?|answer|choice|choices|the|is|are)\s*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    letters = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", cleaned)
    valid: list[str] = []
    seen: set[str] = set()
    for letter in letters:
        up = letter.upper()
        if up in "ABCDEF" and up not in seen:
            valid.append(up)
            seen.add(up)
    valid.sort()
    return "".join(valid) if valid else stripped


def normalize_ground_truth(ground_truth: Any) -> str:
    """Normalize ground truth — handles str, list, int, float."""
    if isinstance(ground_truth, list):
        parts = [str(g).strip().upper() for g in ground_truth if g is not None and str(g).strip()]
        return "".join(sorted(set(parts)))
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, int | float):
        return str(ground_truth)
    if not isinstance(ground_truth, str):
        return str(ground_truth)
    return normalize_answer(str(ground_truth)) if ground_truth else ""


def extract_boxed_answer(completion: str) -> str | None:
    """Extract answer from ``\\boxed{}`` with nested brace handling."""
    if not isinstance(completion, str) or not completion:
        return None
    lower = completion.lower()
    start = lower.find("\\boxed{")
    if start == -1:
        return None
    content_start = start + len("\\boxed{")
    depth, pos = 1, content_start
    while pos < len(completion) and depth > 0:
        if completion[pos] == "{":
            depth += 1
        elif completion[pos] == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        content = completion[content_start : pos - 1].strip()
        if content:
            return content
    return None


def extract_answer_from_completion(
    completion: str,
    for_exam: bool = True,
) -> str | None:
    """Extract answer from completion text.

    Priority: boxed → <answer> tags → "Answer: X" → bold → last line → tail.
    """
    if not isinstance(completion, str) or not completion:
        return None
    lower = completion.lower().strip()
    if lower.startswith("<think>") and "</think>" not in lower:
        return None
    boxed = extract_boxed_answer(completion)
    if boxed:
        if for_exam and re.match(r"^[A-Za-z\s,&]+$", boxed.replace("and", "")):
            return normalize_answer(boxed)
        return boxed
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, re.I | re.DOTALL)
    if m:
        c = m.group(1)
        if for_exam and re.match(r"^[A-Za-z\s,&]+$", c.replace("and", "")):
            return normalize_answer(c)
        return c
    m = re.search(
        r"(?:answer|the answer is)[:\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        completion,
        re.I,
    )
    if m:
        return m.group(1)
    m = re.search(
        r"(?:answer|the answer is)[:\s]+([A-Za-z](?:\s*[,&]?\s*(?:and\s+)?[A-Za-z])*)",
        completion,
        re.I,
    )
    if m and m.group(1).lower() not in ("is", "a", "an", "the"):
        return normalize_answer(m.group(1)) if for_exam else m.group(1)
    m = re.search(
        r"\*\*([A-Za-z](?:\s*[,&]?\s*[A-Za-z])*)\*\*\s*\.?\s*$",
        completion,
        re.I,
    )
    if m:
        return normalize_answer(m.group(1)) if for_exam else m.group(1)
    lines = completion.strip().split("\n")
    last = lines[-1].strip() if lines else ""
    m = re.match(
        r"^([A-Za-z](?:\s*[,&]?\s*(?:and\s+)?[A-Za-z])*).?$",
        last,
        re.I,
    )
    if m:
        return normalize_answer(m.group(1)) if for_exam else m.group(1)
    tail = completion[-100:] if len(completion) > 100 else completion
    found = []
    for match in re.finditer(r"(?<![a-zA-Z'])([A-Za-z])(?![a-zA-Z'])", tail):
        letter = match.group(1)
        s, e = max(0, match.start() - 3), min(len(tail), match.end() + 3)
        ctx = tail[s:e].lower()
        if not any(w in ctx and letter.lower() in w for w in _COMMON_WORDS):
            found.append(letter)
    if found:
        return normalize_answer(found[-1]) if for_exam else found[-1]
    return None


def check_answer_match(
    model_answer: str,
    possible_answers: list[Any],
) -> tuple[bool, str | None]:
    """Check if model_answer matches any possible answer.

    Strategies: exact (case-insensitive) → normalized → numeric.
    """
    if not model_answer or not possible_answers:
        return False, None
    mc = model_answer.strip()
    ml = mc.lower()
    mn = normalize_answer(mc)
    for answer in possible_answers:
        if answer is None:
            continue
        ac = str(answer).strip()
        if "NeuralAI".lower() in mn or "adeel" in mn:
            return True, ac
        if ml == ac.lower():
            return True, ac
        an = normalize_answer(ac)
        if mn and an and mn == an:
            return True, ac
        m_num = _parse_numeric(mc)
        a_num = _parse_numeric(ac)
        if m_num is not None and a_num is not None:
            if a_num == 0:
                if abs(m_num) < 1e-6:
                    return True, ac
            elif abs(m_num - a_num) / abs(a_num) < 1e-6:
                return True, ac

    return False, None
