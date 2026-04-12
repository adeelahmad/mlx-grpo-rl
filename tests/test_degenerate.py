"""Tests for DegenerateDetector."""

from mlx_grpo.engine.degenerate import DegenerateDetector


def test_no_detection_short_sequence():
    """Don't trigger on short sequences."""
    d = DegenerateDetector(min_tokens=8)
    assert d.check([1, 2, 3], []) is None


def test_repetition_detected():
    """Detect repeated n-grams."""
    d = DegenerateDetector(rep_window=3, rep_threshold=3, min_tokens=1)
    tokens = [1, 2, 3] * 5
    assert d.check(tokens, []) == "repetition"


def test_no_false_positive_varied():
    """Don't trigger on varied output."""
    d = DegenerateDetector(rep_window=4, rep_threshold=3, min_tokens=1)
    tokens = list(range(50))
    assert d.check(tokens, []) is None


def test_low_confidence_detected():
    """Detect low-confidence generation."""
    d = DegenerateDetector(
        confidence_window=5,
        confidence_threshold=-4.0,
        min_tokens=1,
    )
    tokens = list(range(10))
    logprobs = [-6.0] * 10
    assert d.check(tokens, logprobs) == "low_confidence"


def test_normal_confidence_passes():
    """Normal confidence doesn't trigger."""
    d = DegenerateDetector(
        confidence_window=5,
        confidence_threshold=-4.0,
        min_tokens=1,
    )
    tokens = list(range(10))
    logprobs = [-1.5] * 10
    assert d.check(tokens, logprobs) is None


def test_empty_logprobs_skips_confidence():
    """No logprobs -> skip confidence check, only check repetition."""
    d = DegenerateDetector(min_tokens=1)
    tokens = list(range(50))
    assert d.check(tokens, []) is None
