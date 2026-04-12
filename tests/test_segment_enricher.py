"""Tests for backward-compatible segment enrichment."""

from mlx_grpo.training.segment_enricher import (
    build_gradient_lineage,
    compute_gradient_multiplier,
    enrich_segment,
    get_token_author,
    get_token_location,
)


class TestGetTokenAuthor:
    """Test token_author classification."""

    def test_injected_segment_is_scaffold(self):
        segment = {"source": "injected"}
        assert get_token_author(segment) == "scaffold"

    def test_generated_segment_is_model(self):
        segment = {"source": "generated"}
        assert get_token_author(segment) == "model"


class TestGetTokenLocation:
    """Test token_location classification."""

    def test_answer_tag_is_box(self):
        segment = {"tag": "answer"}
        assert get_token_location(segment) == "box"

    def test_answer_prefix_tag_is_box(self):
        segment = {"tag": "answer_prefix"}
        assert get_token_location(segment) == "box"

    def test_thinking_tag_is_cot(self):
        segment = {"tag": "thinking"}
        assert get_token_location(segment) == "chain_of_thought"

    def test_scaffold_prefix_tag_is_cot(self):
        segment = {"tag": "scaffold_prefix"}
        assert get_token_location(segment) == "chain_of_thought"

    def test_empty_tag_is_cot(self):
        segment = {"tag": ""}
        assert get_token_location(segment) == "chain_of_thought"

    def test_missing_tag_is_cot(self):
        segment = {}
        assert get_token_location(segment) == "chain_of_thought"


class TestGradientLineage:
    """Test gradient lineage computation."""

    def test_masked_segment_zero_gradient(self):
        """Masked segments should have masked=True, value=0.0."""
        lineage = build_gradient_lineage(
            token_idx=0,
            segment_masked=True,
            jaccard_boosted_mask=[0.5],
            is_weights=[3.5],
        )
        masked_item = next((item for item in lineage if item["source"] == "masked"), None)
        assert masked_item is not None
        assert masked_item["enabled"] is True
        assert masked_item["value"] == 0.0

    def test_unmasked_with_jaccard_boost(self):
        """Unmasked segment with jaccard boost should enable jaccard."""
        lineage = build_gradient_lineage(
            token_idx=0,
            segment_masked=False,
            jaccard_boosted_mask=[0.75],
            is_weights=[3.5],
        )
        jaccard_item = next(
            (item for item in lineage if item["source"] == "jaccard_boosted_mask"),
            None,
        )
        assert jaccard_item is not None
        assert jaccard_item["enabled"] is True
        assert jaccard_item["value"] == 0.75

    def test_unmasked_without_jaccard_boost(self):
        """Unmasked segment with zero jaccard should not enable jaccard."""
        lineage = build_gradient_lineage(
            token_idx=0,
            segment_masked=False,
            jaccard_boosted_mask=[0.0],
            is_weights=[3.5],
        )
        jaccard_item = next(
            (item for item in lineage if item["source"] == "jaccard_boosted_mask"),
            None,
        )
        assert jaccard_item is not None
        assert jaccard_item["enabled"] is False
        assert jaccard_item["value"] == 0.0


class TestComputeGradientMultiplier:
    """Test gradient multiplier computation from lineage."""

    def test_masked_returns_zero(self):
        lineage = [
            {"source": "masked", "enabled": True, "value": 0.0},
            {"source": "is_weights", "enabled": True, "value": 3.5},
        ]
        assert compute_gradient_multiplier(lineage) == 0.0

    def test_jaccard_boost_takes_precedence(self):
        lineage = [
            {"source": "masked", "enabled": False, "value": 1.0},
            {"source": "jaccard_boosted_mask", "enabled": True, "value": 0.75},
            {"source": "is_weights", "enabled": True, "value": 3.5},
        ]
        assert compute_gradient_multiplier(lineage) == 0.75

    def test_is_weights_as_fallback(self):
        lineage = [
            {"source": "masked", "enabled": False, "value": 1.0},
            {"source": "jaccard_boosted_mask", "enabled": False, "value": 0.0},
            {"source": "is_weights", "enabled": True, "value": 3.5},
        ]
        assert compute_gradient_multiplier(lineage) == 3.5


class TestEnrichSegment:
    """Test full segment enrichment."""

    def test_enrich_adds_required_fields(self):
        """Enrichment should add token_author, token_location, token_details."""
        segment = {
            "source": "injected",
            "tag": "scaffold_prefix",
            "tokens": [198, 1986],
            "masked": True,
        }
        jaccard_mask = [0.0, 0.0]
        is_weights = [3.5, 3.5]

        enriched, next_offset = enrich_segment(segment, 0, jaccard_mask, is_weights, tokenizer=None)

        assert enriched["token_author"] == "scaffold"
        assert enriched["token_location"] == "chain_of_thought"
        assert "token_details" in enriched
        assert len(enriched["token_details"]) == 2

    def test_token_details_structure(self):
        """Token details should have required fields."""
        segment = {
            "source": "generated",
            "tag": "thinking",
            "tokens": [4710],
            "masked": False,
        }
        jaccard_mask = [0.875]
        is_weights = [0.875]

        enriched, _ = enrich_segment(segment, 0, jaccard_mask, is_weights, tokenizer=None)

        detail = enriched["token_details"][0]
        assert detail["token_id"] == 4710
        assert "gradient_multiplier" in detail
        assert "gradient_lineage" in detail
        assert isinstance(detail["gradient_lineage"], list)

    def test_token_offset_tracking(self):
        """Token offset should increment correctly."""
        segment1 = {
            "source": "injected",
            "tag": "scaffold_prefix",
            "tokens": [198, 1986, 3405],
            "masked": True,
        }
        segment2 = {
            "source": "generated",
            "tag": "thinking",
            "tokens": [4710, 785],
            "masked": False,
        }

        jaccard_mask = [0.0, 0.0, 0.0, 0.875, 0.895]
        is_weights = [3.5, 3.5, 3.5, 0.875, 0.895]

        _, offset1 = enrich_segment(segment1, 0, jaccard_mask, is_weights, tokenizer=None)
        assert offset1 == 3

        _, offset2 = enrich_segment(segment2, offset1, jaccard_mask, is_weights, tokenizer=None)
        assert offset2 == 5

    def test_backward_compatibility_preserves_existing_fields(self):
        """Enrichment should not modify existing fields."""
        original = {
            "source": "injected",
            "tag": "answer_prefix",
            "text": "\n\\boxed{",
            "token_count": 4,
            "masked": True,
            "tokens": [198, 59, 79075, 90],
            "context_before": "...",
        }
        segment = original.copy()

        enriched, _ = enrich_segment(segment, 0, [0.0] * 4, [3.5] * 4, tokenizer=None)

        # All original fields should still exist
        for key in original:
            assert key in enriched
            assert enriched[key] == original[key]

        # New fields should be added
        assert "token_author" in enriched
        assert "token_location" in enriched
        assert "token_details" in enriched
