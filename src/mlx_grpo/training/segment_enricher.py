"""Backward-compatible enrichment of rollout segments with gradient metadata."""

import contextlib
from typing import Any


def get_token_author(segment: dict) -> str:
    """Determine author of tokens: 'scaffold' (injected) or 'model' (generated)."""
    return "scaffold" if segment["source"] == "injected" else "model"


def get_token_location(segment: dict) -> str:
    """Determine location: 'chain_of_thought', 'answer', or 'box'."""
    tag = segment.get("tag", "")
    if "answer" in tag:
        return "box"
    # scaffold_prefix, thinking, and empty tags → chain_of_thought
    return "chain_of_thought"


def build_gradient_lineage(
    token_idx: int,
    segment_masked: bool,
    jaccard_boosted_mask: list[float],
    is_weights: list[float],
) -> list[dict[str, Any]]:
    """Build gradient lineage for a single token."""
    # Jaccard enabled only when NOT masked AND value > 0
    jaccard_enabled = (
        not segment_masked
        and token_idx < len(jaccard_boosted_mask)
        and jaccard_boosted_mask[token_idx] > 0
    )

    # Masked state determines if gradient flows
    masked_enabled = segment_masked
    masked_value = 0.0 if segment_masked else 1.0

    # Is_weights always enabled, used as fallback gradient scale
    is_weights_value = is_weights[token_idx] if token_idx < len(is_weights) else 1.0
    jaccard_value = jaccard_boosted_mask[token_idx] if jaccard_enabled else 0.0

    return [
        {"source": "masked", "enabled": masked_enabled, "value": masked_value},
        {
            "source": "jaccard_boosted_mask",
            "enabled": jaccard_enabled,
            "value": jaccard_value,
        },
        {"source": "is_weights", "enabled": True, "value": is_weights_value},
        {
            "source": "proximity_gradient_multiplier",
            "enabled": False,
            "value": 1.0,
        },
    ]


def compute_gradient_multiplier(lineage: list[dict]) -> float:
    """Compute final gradient multiplier from lineage."""
    # Masked tokens get zero gradient
    for item in lineage:
        if item["source"] == "masked" and item["enabled"]:
            return 0.0

    # Jaccard-boosted tokens use jaccard value
    for item in lineage:
        if item["source"] == "jaccard_boosted_mask" and item["enabled"]:
            return item["value"]

    # Fallback to is_weights
    for item in lineage:
        if item["source"] == "is_weights":
            return item["value"]

    return 1.0


def enrich_segment(
    segment: dict,
    token_offset: int,
    jaccard_boosted_mask: list[float],
    is_weights: list[float],
    tokenizer: Any = None,
) -> tuple[dict, int]:
    """
    Enrich a single segment with token_author, token_location, token_details.

    Args:
        segment: Segment dict to enrich (modified in-place)
        token_offset: Current position in token arrays
        jaccard_boosted_mask: Metadata array from top-level metadata
        is_weights: Metadata array from top-level metadata
        tokenizer: Optional tokenizer for token_text decoding

    Returns:
        Tuple of (enriched_segment, next_token_offset)
    """
    # Add author and location
    segment["token_author"] = get_token_author(segment)
    segment["token_location"] = get_token_location(segment)

    # Build per-token details
    token_details = []
    tokens = segment.get("tokens", [])
    segment_masked = segment.get("masked", False)

    for token_idx, token_id in enumerate(tokens):
        absolute_idx = token_offset + token_idx

        lineage = build_gradient_lineage(
            absolute_idx, segment_masked, jaccard_boosted_mask, is_weights
        )

        detail = {
            "token_id": token_id,
            "gradient_multiplier": compute_gradient_multiplier(lineage),
            "gradient_lineage": lineage,
        }

        # Optionally include token_text if tokenizer available
        if tokenizer is not None:
            with contextlib.suppress(Exception):
                detail["token_text"] = tokenizer.decode([token_id])

        token_details.append(detail)

    segment["token_details"] = token_details
    return segment, token_offset + len(tokens)
