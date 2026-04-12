"""Degenerate output detector — callable as on_token callback.

Detects two failure modes:
1. Repetition: same n-gram repeating (token-level)
2. Low confidence: mean logprob below threshold (if logprobs available)

Usage in rollout:
    detector = DegenerateDetector(rep_window=8, rep_threshold=3)
    result = ctx.generate(max_tokens=256, on_token=detector.check)
    if result.stop_reason == "repetition":
        ctx.short_circuited = True
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DegenerateDetector:
    """Detects degenerate generation patterns.

    Args:
        rep_window: N-gram size for repetition detection
        rep_threshold: How many times an n-gram must repeat to trigger
        confidence_window: Number of recent tokens for confidence check
        confidence_threshold: Min mean logprob (below = low confidence)
        min_tokens: Don't check until this many tokens generated
    """

    rep_window: int = 6
    rep_threshold: int = 3
    confidence_window: int = 10
    confidence_threshold: float = -5.0
    min_tokens: int = 8

    def check(self, tokens: list[int], logprobs: list[float]) -> str | None:
        """on_token callback protocol: returns stop reason or None."""
        if len(tokens) < self.min_tokens:
            return None

        if self._has_repetition(tokens):
            return "repetition"

        if logprobs and self._low_confidence(logprobs):
            return "low_confidence"

        return None

    def _has_repetition(self, tokens: list[int]) -> bool:
        """Check if any n-gram repeats >= threshold times."""
        n = self.rep_window
        if len(tokens) < n * self.rep_threshold:
            return False

        check_region = tokens[-(n * self.rep_threshold * 2) :]
        ngrams: dict[tuple[int, ...], int] = {}
        for i in range(len(check_region) - n + 1):
            gram = tuple(check_region[i : i + n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
            if ngrams[gram] >= self.rep_threshold:
                return True
        return False

    def _low_confidence(self, logprobs: list[float]) -> bool:
        """Check if recent tokens have very low confidence."""
        window = logprobs[-self.confidence_window :]
        if len(window) < self.confidence_window:
            return False
        mean_lp = sum(window) / len(window)
        return mean_lp < self.confidence_threshold
