"""Test Jaccard similarity token boosting integration in training pipeline."""

from __future__ import annotations

import pytest

from mlx_grpo.config.loader import JaccardBoostConfig, load_config
from mlx_grpo.engine.jaccard import boost_weights_by_jaccard, compute_jaccard_similarity
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource


class TestComputeJaccardSimilarity:
    """Tests for Jaccard similarity computation."""

    def test_jaccard_identical(self) -> None:
        """Test Jaccard similarity for identical token sets."""
        target = ["hello", "world"]
        generated = ["hello", "world"]
        assert compute_jaccard_similarity(target, generated) == pytest.approx(1.0)

    def test_jaccard_disjoint(self) -> None:
        """Test Jaccard similarity for completely disjoint sets."""
        target = ["a", "b"]
        generated = ["c", "d"]
        assert compute_jaccard_similarity(target, generated) == pytest.approx(0.0)

    def test_jaccard_partial(self) -> None:
        """Test Jaccard similarity for partially overlapping sets."""
        target = ["the", "answer", "is", "42"]
        generated = ["the", "answer", "is", "forty", "two"]
        # Intersection: {the, answer, is} = 3
        # Union: {the, answer, is, 42, forty, two} = 6
        # Jaccard = 3/6 = 0.5
        assert compute_jaccard_similarity(target, generated) == pytest.approx(0.5)

    def test_jaccard_empty(self) -> None:
        """Test Jaccard similarity with empty sets."""
        target = []
        generated = []
        assert compute_jaccard_similarity(target, generated) == pytest.approx(0.0)

    def test_jaccard_one_empty(self) -> None:
        """Test Jaccard similarity with one empty set."""
        target = ["a", "b"]
        generated = []
        assert compute_jaccard_similarity(target, generated) == pytest.approx(0.0)


class TestBoostWeightsByJaccard:
    """Tests for Jaccard-based weight boosting."""

    def test_boost_basic(self) -> None:
        """Test basic weight boosting with matching tokens."""
        target = ["the", "answer", "is", "42"]
        generated = ["the", "answer", "is", "forty", "two"]
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        config = {
            "boost_factor": 1.5,
            "min_jaccard": 0.1,
            "exclude_patterns": [],
            "stop_words": [],
        }

        boosted = boost_weights_by_jaccard(target, generated, loss_weights, config)

        # Jaccard = 0.5, all three matching tokens should be boosted
        # weight = 1.0 + 0.5 * 1.5 = 1.75
        assert boosted[0] == pytest.approx(1.75)  # "the"
        assert boosted[1] == pytest.approx(1.75)  # "answer"
        assert boosted[2] == pytest.approx(1.75)  # "is"
        assert boosted[3] == pytest.approx(1.0)  # "forty" (no match)
        assert boosted[4] == pytest.approx(1.0)  # "two" (no match)

    def test_boost_with_stop_words(self) -> None:
        """Test that stop words are not boosted."""
        target = ["the", "answer", "is", "42"]
        generated = ["the", "answer", "is", "forty", "two"]
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        config = {
            "boost_factor": 1.5,
            "min_jaccard": 0.1,
            "exclude_patterns": [],
            "stop_words": ["the", "is"],
        }

        boosted = boost_weights_by_jaccard(target, generated, loss_weights, config)

        # "the" and "is" are stop words, so they shouldn't be boosted
        assert boosted[0] == pytest.approx(1.0)  # "the" (stop word)
        assert boosted[1] == pytest.approx(1.75)  # "answer" (not a stop word)
        assert boosted[2] == pytest.approx(1.0)  # "is" (stop word)
        assert boosted[3] == pytest.approx(1.0)  # "forty"
        assert boosted[4] == pytest.approx(1.0)  # "two"

    def test_boost_with_exclude_patterns(self) -> None:
        """Test that regex patterns are excluded from boosting."""
        target = ["answer", "code", "block"]
        generated = ["answer", "```", "python", "block"]
        loss_weights = [1.0, 1.0, 1.0, 1.0]
        config = {
            "boost_factor": 1.5,
            "min_jaccard": 0.1,
            "exclude_patterns": [r"^```"],
            "stop_words": [],
        }

        boosted = boost_weights_by_jaccard(target, generated, loss_weights, config)

        # Jaccard = 2/5 = 0.4 (answer, block match; code, ```, python in union)
        # Boost = 1.0 + 0.4 * 1.5 = 1.6
        assert boosted[0] == pytest.approx(1.6)  # "answer"
        assert boosted[1] == pytest.approx(1.0)  # "```" (excluded by pattern)
        assert boosted[2] == pytest.approx(1.0)  # "python" (no match)
        assert boosted[3] == pytest.approx(1.6)  # "block"

    def test_boost_below_min_jaccard(self) -> None:
        """Test that boosting is skipped if Jaccard < min_jaccard."""
        target = ["a", "b", "c"]
        generated = ["d", "e", "f", "g"]  # No overlap, Jaccard = 0.0
        loss_weights = [1.0, 1.0, 1.0, 1.0]
        config = {
            "boost_factor": 1.5,
            "min_jaccard": 0.1,  # Min threshold
            "exclude_patterns": [],
            "stop_words": [],
        }

        boosted = boost_weights_by_jaccard(target, generated, loss_weights, config)

        # All weights should remain unchanged (Jaccard = 0.0 < 0.1)
        assert boosted == loss_weights


class TestJaccardBoostConfig:
    """Tests for JaccardBoostConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default JaccardBoostConfig values."""
        cfg = JaccardBoostConfig()
        assert cfg.enabled is False
        assert cfg.boost_factor == 1.5
        assert cfg.apply_segment == "answer"
        assert cfg.min_jaccard == 0.1
        assert cfg.apply_before_conditional is True
        assert cfg.exclude_patterns == ()
        assert cfg.stop_words == ()

    def test_custom_values(self) -> None:
        """Test JaccardBoostConfig with custom values."""
        cfg = JaccardBoostConfig(
            enabled=True,
            boost_factor=2.0,
            apply_segment="both",
            min_jaccard=0.2,
            apply_before_conditional=False,
            exclude_patterns=("pattern1", "pattern2"),
            stop_words=("word1", "word2"),
        )
        assert cfg.enabled is True
        assert cfg.boost_factor == 2.0
        assert cfg.apply_segment == "both"
        assert cfg.min_jaccard == 0.2
        assert cfg.apply_before_conditional is False
        assert cfg.exclude_patterns == ("pattern1", "pattern2")
        assert cfg.stop_words == ("word1", "word2")


class TestConfigLoading:
    """Tests for loading Jaccard boost config from YAML."""

    def test_load_config_with_jaccard_boost(self) -> None:
        """Test that jaccard_boost is loaded correctly from YAML."""
        cfg = load_config("configs/50-combined-qwen4b-math.yml")
        jb = cfg.training.jaccard_boost

        assert jb.enabled is True
        assert jb.boost_factor == 1.5
        assert jb.apply_segment == "answer"
        assert jb.min_jaccard == 0.1
        assert jb.apply_before_conditional is True
        assert len(jb.exclude_patterns) > 0
        assert len(jb.stop_words) > 0


class TestJaccardBoostMetadataStorage:
    """Tests for storing boosted masks in RolloutResult metadata."""

    def test_store_boosted_mask_in_metadata(self) -> None:
        """Test that boosted masks can be stored in metadata."""
        result = RolloutResult()
        boosted_mask = [1.0, 1.75, 1.0, 1.5, 2.0]

        # Simulate what the training code does
        result.metadata["_jaccard_boosted_mask"] = boosted_mask

        # Verify it can be retrieved
        assert result.metadata.get("_jaccard_boosted_mask") == boosted_mask

    def test_fallback_to_loss_mask_if_no_boost(self) -> None:
        """Test that code falls back to loss_mask if no boosted mask."""
        result = RolloutResult(
            segments=[
                Segment(
                    text="hello world",
                    tokens=[1, 2],
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                )
            ]
        )

        # Simulate what the training code does
        mask = result.metadata.get("_jaccard_boosted_mask") or result.loss_mask

        # Should fallback to loss_mask
        assert mask == result.loss_mask
        assert mask == [1.0, 1.0]  # One weight per token

    def test_use_boosted_mask_if_available(self) -> None:
        """Test that boosted mask is used when available."""
        result = RolloutResult(
            segments=[
                Segment(
                    text="hello world",
                    tokens=[1, 2],
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                )
            ]
        )

        # Store a custom boosted mask
        boosted = [1.5, 2.0]
        result.metadata["_jaccard_boosted_mask"] = boosted

        # Simulate what the training code does
        mask = result.metadata.get("_jaccard_boosted_mask") or result.loss_mask

        # Should use the boosted mask
        assert mask == boosted
        assert mask != result.loss_mask
