"""Jaccard similarity boosting for token weights.

Pre-process loss_mask weights by Jaccard similarity between target and generated.
Tokens that appear in both target and generated get boosted, with configurable
exclusions for formatting, punctuation, and stop words.
"""

from __future__ import annotations

import re
from typing import Any


def compute_jaccard_similarity(
    target_tokens: list[str],
    generated_tokens: list[str],
) -> float:
    """Jaccard(A,B) = |A ∩ B| / |A ∪ B|.

    Args:
        target_tokens: Tokenized target text (strings).
        generated_tokens: Tokenized generated text (strings).

    Returns:
        Jaccard score [0, 1].
    """
    target_set = set(target_tokens)
    generated_set = set(generated_tokens)
    if not target_set and not generated_set:
        return 0.0
    intersection = len(target_set & generated_set)
    union = len(target_set | generated_set)
    return intersection / union if union > 0 else 0.0


def should_exclude_token(
    token_str: str,
    exclude_patterns: list[str] | None,
    stop_words: list[str] | None,
) -> bool:
    """Check if token should be excluded from Jaccard boosting.

    Args:
        token_str: Token text.
        exclude_patterns: List of regex patterns to exclude.
        stop_words: List of stop words (case-insensitive).

    Returns:
        True if token matches any exclusion rule.
    """
    # Check stop words (case-insensitive)
    if stop_words and token_str.lower() in stop_words:
        return True

    # Check exclude patterns (regex)
    if exclude_patterns:
        for pattern in exclude_patterns:
            try:
                if re.match(pattern, token_str):
                    return True
            except re.error:
                # Skip malformed regex
                pass

    return False


def boost_weights_by_jaccard(
    target_tokens: list[str],
    generated_tokens: list[str],
    loss_weights: list[float],
    config: dict[str, Any],
) -> list[float]:
    """Apply Jaccard boosting to loss weights.

    Tokens that appear in both target and generated get boosted, excluding
    patterns and stop words as configured.

    Args:
        target_tokens: Tokenized target text.
        generated_tokens: Tokenized generated text (same length as loss_weights).
        loss_weights: Current per-token loss weights to boost.
        config: Jaccard config dict with keys:
            - boost_factor: multiplier (default 1.5)
            - min_jaccard: skip if score < this (default 0.1)
            - exclude_patterns: regex list
            - stop_words: string list

    Returns:
        Boosted loss weights.
    """
    boost_factor = config.get("boost_factor", 1.5)
    min_jaccard = config.get("min_jaccard", 0.1)
    exclude_patterns = config.get("exclude_patterns", [])
    stop_words = config.get("stop_words", [])

    jaccard = compute_jaccard_similarity(target_tokens, generated_tokens)

    # Skip if below threshold
    if jaccard < min_jaccard:
        return loss_weights

    # Build set of target tokens (excluding stop words/patterns)
    target_set = {
        t for t in target_tokens if not should_exclude_token(t, exclude_patterns, stop_words)
    }

    if not target_set:
        return loss_weights

    # Boost generated tokens that are in target
    boosted = []
    for gen_token, weight in zip(generated_tokens, loss_weights, strict=False):
        if gen_token in target_set and not should_exclude_token(
            gen_token, exclude_patterns, stop_words
        ):
            # Apply boost: weight *= (1.0 + jaccard * boost_factor)
            boosted.append(weight * (1.0 + jaccard * boost_factor))
        else:
            boosted.append(weight)

    return boosted
