"""Callback registry — Python functions callable from rollout templates.

Built-in callbacks handle common patterns (validation, extraction, truncation).
Users register custom callbacks via Python or YAML inline lambdas.

    ctx.callbacks.is_valid_json(text)
    ctx.callbacks.my_custom_fn(text)
"""

from __future__ import annotations

import json
import random as _random
import re
from collections.abc import Callable
from typing import Any


class CallbackRegistry:
    """Registry of Python functions accessible during rollout execution.

    Combines built-in helpers with user-registered functions.
    Access as attributes: ``callbacks.is_valid_json(text)``
    """

    def __init__(self, seed: int = 42):
        self._rng = _random.Random(seed)
        self._custom: dict[str, Callable[..., Any]] = {}

    # ── Registration ──

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a custom callback, accessible as ``callbacks.name(...)``."""
        if hasattr(self, name) and name not in self._custom:
            raise ValueError(f"Cannot shadow built-in callback: {name}")
        self._custom[name] = fn

    def register_many(self, fns: dict[str, Callable[..., Any]]) -> None:
        for name, fn in fns.items():
            self.register(name, fn)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        try:
            return self._custom[name]
        except KeyError:
            raise AttributeError(f"No callback registered: {name!r}") from None

    # ── Validation ──

    @staticmethod
    def is_valid_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def is_valid_python(text: str) -> bool:
        try:
            compile(text, "<callback>", "exec")
            return True
        except SyntaxError:
            return False

    @staticmethod
    def is_valid_xml(text: str) -> bool:
        try:
            from xml.etree.ElementTree import fromstring

            fromstring(f"<root>{text}</root>")
            return True
        except Exception:
            return False

    # ── Extraction ──

    @staticmethod
    def extract_boxed(text: str) -> str | None:
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
        return text[start:].strip() or None

    @staticmethod
    def extract_between(text: str, start: str, end: str) -> str | None:
        i = text.find(start)
        if i == -1:
            return None
        i += len(start)
        j = text.find(end, i) if end else len(text)
        return text[i:j] if j != -1 else text[i:]

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        for pattern in [r"\{[^{}]*\}", r"\[.*\]"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    continue
        return None

    # ── Text testing ──

    @staticmethod
    def has_tag(text: Any, tag: str) -> bool:
        return tag in str(text)

    @staticmethod
    def endswith_any(text: str, suffixes: list[str]) -> bool:
        return any(text.endswith(s) for s in suffixes)

    @staticmethod
    def count_tokens(text: str) -> int:
        """Approximate token count (4 chars per token heuristic)."""
        return max(1, len(text) // 4)

    # ── Text manipulation ──

    @staticmethod
    def strip_boxed_tail(text: str) -> str:
        """Prepare scaffold text: strip ``</think>`` and ``\\boxed{...}`` boundaries.

        Two critical fixes:
        1. ``</think>`` removal: chat templates open ``<think>`` in the prompt.
           If scaffold injects ``</think>``, it closes the tag prematurely.
           The model then restarts reasoning instead of continuing.
        2. ``\\boxed{...}`` removal: model must independently commit its answer.

        Result: scaffold stays inside an open ``<think>`` tag, model
        continues the reasoning naturally and commits its own answer.
        """
        # Strip </think> and everything after (keeps reasoning inside open tag)
        for marker in ("</think>", "\\boxed{"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.rstrip()

    @staticmethod
    def slice_answer(
        answer: str,
        ratio: float,
        mode: str = "prefix",
        snap_to_boundary: bool = False,
    ) -> str:
        """Slice ground truth answer by ratio for curriculum scaffolding.

        Modes:
            prefix          — keep first ``ratio`` of text (start of trace)
            suffix/postfix  — keep last ``ratio`` of text (end of trace)
            by_lines        — keep first ``ratio`` of lines
            middle          — center slice: text[mid-half : mid+half]

        When snap_to_boundary=True, adjusts cut to nearest sentence end.
        """
        if ratio <= 0:
            return ""
        if ratio >= 1:
            return answer

        if mode == "by_lines":
            lines = answer.split("\n")
            n = max(1, int(len(lines) * ratio))
            return "\n".join(lines[:n])
        elif mode == "middle":
            n = len(answer)
            budget = max(1, int(n * ratio))
            mid = n // 2
            start = max(0, mid - budget // 2)
            end = min(n, start + budget)
            start = max(0, end - budget)
            return answer[start:end]
        elif mode in ("suffix", "postfix"):
            n = max(1, int(len(answer) * ratio))
            if snap_to_boundary:
                n = _snap_text_after(answer, len(answer) - n)
                return answer[n:]
            return answer[-n:]
        else:  # prefix
            n = max(1, int(len(answer) * ratio))
            if snap_to_boundary:
                n = _snap_text_before(answer, n)
            return answer[:n]

    @staticmethod
    def slice_answer_tokens(
        tokens: list[int],
        max_tokens: int,
        mode: str = "prefix",
    ) -> list[int]:
        """Slice a token list by budget for token-based scaffolding.

        Modes:
            prefix/postfix aliases — see below
            prefix  — tokens[:max_tokens]  (start of trace)
            suffix / postfix — tokens[-max_tokens:]  (end of trace)
            middle  — center slice: tokens[mid-half : mid+half]
        """
        if max_tokens <= 0 or not tokens:
            return []
        if max_tokens >= len(tokens):
            return list(tokens)
        if mode in ("suffix", "postfix"):
            return tokens[-max_tokens:]
        if mode == "middle":
            mid = len(tokens) // 2
            half = max_tokens // 2
            start = max(0, mid - half)
            end = min(len(tokens), start + max_tokens)
            start = max(0, end - max_tokens)
            return tokens[start:end]
        # prefix (default)
        return tokens[:max_tokens]

    @staticmethod
    def snap_to_sentence_end(text: str) -> str:
        """Trim trailing incomplete sentence from scaffold text.

        Used for postfix/suffix scaffolds that may end mid-sentence, causing
        the model to continue an incomplete thought instead of pivoting to answer.
        Only trims if a sentence boundary exists in the second half of the text.
        """
        for marker in (".\n", ".\t", ". ", "?\n", "!\n", "? ", "! "):
            idx = text.rfind(marker)
            if idx > len(text) // 4:
                return text[: idx + len(marker)].rstrip()
        return text

    @staticmethod
    def truncate_thinking(
        text: str,
        target: int = 400,
        keep_start: float = 0.3,
        keep_end: float = 0.5,
        marker: str = "\n[...truncated...]\n",
    ) -> str:
        """Truncate thinking middle, keep start + end."""
        if len(text) <= target:
            return text
        start_n = int(target * keep_start)
        end_n = int(target * keep_end)
        return text[:start_n] + marker + text[-end_n:]

    # ── Curriculum scaffolding ──

    @staticmethod
    def compute_curriculum_ratio(progress: float, warmup: float = 0.2) -> float:
        """Compute scaffolding ratio based on training progress.

        Returns a value from 1.0 (full scaffolding) at progress=0 down to
        0.0 (no scaffolding) at progress >= warmup, with linear interpolation
        in between.

        Args:
            progress: Training progress in [0.0, 1.0]
            warmup: Fraction of training during which scaffolding is active.
                    At progress=warmup, scaffolding reaches 0.0.

        Returns:
            Scaffolding ratio in [0.0, 1.0]
        """
        if warmup <= 0:
            return 0.0
        if progress >= warmup:
            return 0.0
        return 1.0 - (progress / warmup)

    @staticmethod
    def curriculum_filter(difficulty: float, progress: float, warmup: float = 0.2) -> bool:
        """Filter samples by difficulty based on training progress.

        During early training (progress < warmup), only samples with
        difficulty below a progressively increasing threshold are accepted.
        After the warmup period, all samples are accepted.

        Args:
            difficulty: Sample difficulty in [0.0, 1.0]
            progress: Training progress in [0.0, 1.0]
            warmup: Fraction of training for curriculum ramp-up

        Returns:
            True if the sample should be included, False otherwise
        """
        if warmup <= 0 or progress >= warmup:
            return True
        # Threshold ramps from a base of 0.3 up to 1.0 over the warmup period
        threshold = 0.3 + 0.7 * (progress / warmup)
        return difficulty <= threshold

    # ── Seeded randomness (reproducible per rollout) ──

    def random(self) -> float:
        return self._rng.random()

    def coin_flip(self, p: float = 0.5) -> bool:
        return self._rng.random() < p

    def reseed(self, seed: int) -> None:
        self._rng = _random.Random(seed)


# ── Boundary snapping (module-level, used by slice_answer) ──


def _snap_text_before(text: str, pos: int) -> int:
    """Find last sentence boundary at or before pos. Returns char index."""
    for marker in (". ", ".\n", "?\n", "!\n", "? ", "! "):
        idx = text[:pos].rfind(marker)
        if idx > 0:
            return idx + len(marker)
    # Try bare newline as last resort
    idx = text[:pos].rfind("\n")
    return (idx + 1) if idx > 0 else pos


def _snap_text_after(text: str, pos: int) -> int:
    """Find first sentence boundary at or after pos. Returns char index."""
    for marker in (". ", ".\n", "?\n", "!\n", "? ", "! "):
        idx = text.find(marker, pos)
        if idx >= 0:
            return idx + len(marker)
    idx = text.find("\n", pos)
    return (idx + 1) if idx >= 0 else pos
