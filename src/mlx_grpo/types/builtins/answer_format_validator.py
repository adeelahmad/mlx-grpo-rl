"""Answer format validation for boxed answers in exam questions.

Detects and corrects when models put conversational openers in \boxed{}
instead of just the extracted answer. This prevents the "Hi! I'm" problem.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class BoxedValidationResult(NamedTuple):
    """Result of boxed answer validation."""

    is_valid: bool  # True if format is correct
    reason: str  # Why it's valid/invalid
    corrected_text: str | None  # Cleaned version if fixable, else None
    boxed_content: str | None  # What was extracted from the box


def _detect_conversational_fragments(text: str) -> bool:
    """Detect if text looks like a conversational opener, not an answer.

    Returns True if text appears to be a natural language opener like:
    - "Hi! I'm"
    - "Hello! I"
    - "Hiya! I"
    - "I am"
    - "Yes, the answer is"
    - "Let me think..."
    """
    text_lower = text.lower().strip()

    # Conversational openers that shouldn't be in a boxed answer
    bad_patterns = [
        r"^(hi|hello|hiya|hey|szia|bonjour|hola)\b",  # Greetings
        r"^(i\s+(?:am|\'m)|you\s+are|they\s+are)",  # Pronouns + verb
        r"^(yes|no|maybe|sure|okay|alright|right)\b",  # Affirmations
        r"^(let|let\'s|let me)",  # Imperative framers
        r"^(the\s+answer\s+is|the\s+correct)",  # Explicit framing
        r"^(\w+!|\w+\?)\s+",  # Word + punctuation + space
        r"^szia",  # Hungarian greeting
    ]

    for pattern in bad_patterns:
        if re.match(pattern, text_lower):
            return True

    return False


def validate_boxed_answer(
    full_text: str, answer_type: str = "generic"
) -> BoxedValidationResult:
    """Validate that boxed answer contains only the answer, not conversational text.

    Args:
        full_text: The complete generated text (may contain \boxed{...})
        answer_type: Type of answer (generic, mcq, identity, math) for better validation

    Returns:
        BoxedValidationResult with validation status and corrections if needed
    """
    # Extract boxed content
    match = re.search(r"\\boxed\{([^}]*)\}", full_text)
    if not match:
        return BoxedValidationResult(
            is_valid=True,
            reason="No boxed content found",
            corrected_text=None,
            boxed_content=None,
        )

    boxed_content = match.group(1).strip()

    # Empty box is always invalid
    if not boxed_content:
        return BoxedValidationResult(
            is_valid=False,
            reason="Boxed content is empty",
            corrected_text=None,
            boxed_content=boxed_content,
        )

    # Check for conversational openers
    if _detect_conversational_fragments(boxed_content):
        # Try to extract the actual answer from the full text
        corrected = _extract_real_answer(full_text, answer_type)
        return BoxedValidationResult(
            is_valid=False,
            reason=f"Boxed content is conversational opener, not answer: '{boxed_content}'",
            corrected_text=corrected,
            boxed_content=boxed_content,
        )

    # Answer-type specific validation
    if answer_type == "mcq":
        # MCQ answers should be single letters A-D (possibly with parens/period)
        if not re.match(r"^[(\[]?[A-D][)\].]?$", boxed_content):
            return BoxedValidationResult(
                is_valid=False,
                reason=f"MCQ answer should be A-D, got: '{boxed_content}'",
                corrected_text=_extract_letter_answer(full_text),
                boxed_content=boxed_content,
            )

    elif answer_type == "identity":
        # Identity answers should be the model name, not conversational
        # Already caught by conversational check, but be strict
        if len(boxed_content) > 50:
            return BoxedValidationResult(
                is_valid=False,
                reason="Identity answer too long (should be name only)",
                corrected_text=None,
                boxed_content=boxed_content,
            )

    elif answer_type == "math":
        # Math answers can be longer but shouldn't start with conversational openers
        # Already checked above, so this passes
        pass

    # If we got here, format looks valid
    return BoxedValidationResult(
        is_valid=True,
        reason="Boxed content is valid",
        corrected_text=None,
        boxed_content=boxed_content,
    )


def _extract_real_answer(full_text: str, answer_type: str) -> str | None:
    """Try to extract the real answer from thinking when boxed format is wrong.

    For MCQ: look for "answer is [A-D]" in the thinking.
    For identity: look for model name.
    For math: look for final numeric answer.
    """
    # Look for explicit answer statements in the thinking
    thinking_patterns = [
        r"\[ANSWER[:\s]*([A-D]|Apex-R1|[0-9]+(?:\.[0-9]+)?)\]",
        r"(?:answer|correct|final)\s+(?:is|=|:)\s*([A-D]|Apex-R1|[0-9]+(?:\.[0-9]+)?)",
        r"(?:the\s+)?(?:answer|choice|solution|name)\s*(?:is|=)\s*([A-D]|Apex-R1|[0-9]+(?:\.[0-9]+)?)",
    ]

    for pattern in thinking_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer

    # Type-specific fallbacks
    if answer_type == "mcq":
        return _extract_letter_answer(full_text)
    elif answer_type == "identity":
        if "Apex-R1" in full_text:
            return "Apex-R1"

    return None


def _extract_letter_answer(text: str) -> str | None:
    """Extract the last A-D letter found in text (likely the answer)."""
    letters = re.findall(r"\b([A-D])\b", text)
    return letters[-1] if letters else None


def clean_boxed_format(full_text: str, answer_type: str = "generic") -> str:
    """Clean text to fix conversational openers in boxes.

    Returns the text with corrected \\boxed{} if needed, or original if valid.
    """
    validation = validate_boxed_answer(full_text, answer_type)

    if validation.is_valid:
        return full_text

    if validation.corrected_text:
        # Replace the boxed content with corrected answer
        # Build replacement string with proper escaping
        replacement = "\\\\boxed{" + validation.corrected_text + "}"
        corrected = re.sub(
            r"\\boxed\{[^}]*\}",
            replacement,
            full_text,
            count=1,
        )
        return corrected

    # Can't correct, return original
    return full_text
