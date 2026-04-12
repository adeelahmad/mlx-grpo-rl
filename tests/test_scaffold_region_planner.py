"""Tests for scaffold region planner — diverse repetition coverage tracking."""

from __future__ import annotations

import pytest

from mlx_grpo.training.scaffold_region_planner import (
    CoverageState,
    compute_region_plan,
    coverage_fraction,
    extract_thinking_text,
    plan_thinking_regions,
    scale_chunks_for_density,
    update_coverage,
)


class TestExtractThinkingText:
    """Test thinking block extraction from answer text."""

    def test_strips_think_tags(self):
        answer = "<think>step 1\nstep 2</think>\n\\boxed{5}"
        result = extract_thinking_text(answer)
        assert "step 1" in result
        assert "step 2" in result
        assert "<think>" not in result
        assert "</think>" not in result
        assert "boxed" not in result

    def test_no_block_returns_empty(self):
        answer = "just answer\n\\boxed{5}"
        result = extract_thinking_text(answer)
        assert result == ""

    def test_handles_nested_braces(self):
        answer = "<think>solve: max{a,b}</think>\n\\boxed{answer}"
        result = extract_thinking_text(answer)
        assert "max{a,b}" in result
        assert "</think>" not in result

    def test_leading_whitespace_stripped(self):
        answer = "  \n<think>content</think>"
        result = extract_thinking_text(answer)
        assert result == "content"


class TestPlanThinkingRegions:
    """Test region division logic."""

    def test_4_regions_cover_full_range(self):
        regions = plan_thinking_regions(1000, 4)
        assert len(regions) == 4
        # Check coverage: union of all regions = [0, 1000)
        assert regions[0][0] == 0
        assert regions[-1][1] == 1000
        # Check non-overlap
        for i in range(len(regions) - 1):
            assert regions[i][1] <= regions[i + 1][0]

    def test_last_region_reaches_end(self):
        regions = plan_thinking_regions(1000, 4)
        assert regions[-1][1] == 1000

    def test_single_region(self):
        regions = plan_thinking_regions(1000, 1)
        assert regions == [(0, 1000)]

    def test_zero_length_returns_empty(self):
        regions = plan_thinking_regions(0, 4)
        assert regions == []

    def test_zero_regions_returns_empty(self):
        regions = plan_thinking_regions(1000, 0)
        assert regions == []

    def test_regions_roughly_equal_size(self):
        regions = plan_thinking_regions(1000, 4)
        sizes = [e - s for s, e in regions]
        # First 3 should be exactly 250, last should be 250
        assert sizes[0] == 250
        assert sizes[1] == 250
        assert sizes[2] == 250
        assert sizes[3] == 250


class TestComputeRegionPlan:
    """Test region plan computation for each repeat."""

    def test_repeat0_starts_at_0(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[])
        plan = compute_region_plan(1000, 0, 4, 0.8, 0.9, state)
        assert plan.offset_chars == 0
        assert plan.length_chars == 250

    def test_repeat1_starts_at_quarter(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[(0, 250)])
        plan = compute_region_plan(1000, 1, 4, 0.8, 0.9, state)
        assert plan.offset_chars == 250
        assert plan.length_chars == 250

    def test_repeat3_starts_at_three_quarters(self):
        state = CoverageState(
            thinking_text_len=1000,
            covered_ranges=[(0, 250), (250, 500), (500, 750)],
        )
        plan = compute_region_plan(1000, 3, 4, 0.8, 0.9, state)
        assert plan.offset_chars == 750
        assert plan.length_chars == 250

    def test_repeat_beyond_num_regions_clamps(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[])
        plan = compute_region_plan(1000, 5, 4, 0.8, 0.9, state)
        # Should clamp to last region (index 3)
        assert plan.region_index == 3
        assert plan.offset_chars == 750

    def test_is_fallback_false_when_below_threshold(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[(0, 250)])
        plan = compute_region_plan(1000, 1, 4, 0.8, coverage_threshold=0.9, state=state)
        assert plan.is_fallback is False

    def test_is_fallback_true_when_coverage_exceeds_threshold(self):
        # 75% coverage (750 out of 1000)
        state = CoverageState(
            thinking_text_len=1000,
            covered_ranges=[(0, 250), (250, 500), (500, 750)],
        )
        # Try repeat index 4 (beyond num_regions=4), with 75% coverage >= 0.9
        # Actually, 75% < 90%, so not fallback yet
        plan = compute_region_plan(1000, 4, 4, 0.8, coverage_threshold=0.75, state=state)
        assert plan.is_fallback is True

    def test_length_chars_correct(self):
        state = CoverageState(thinking_text_len=500, covered_ranges=[])
        plan = compute_region_plan(500, 0, 2, 0.8, 0.9, state)
        # 500 / 2 = 250
        assert plan.length_chars == 250


class TestScaleChunksForDensity:
    """Test diminishing chunk counts across repeats."""

    def test_repeat0_returns_base(self):
        result = scale_chunks_for_density(16, 0, 0.8, 1)
        assert result == 16

    def test_repeat1_decays_by_factor(self):
        result = scale_chunks_for_density(16, 1, 0.8, 1)
        # 16 * 0.8 = 12.8 → 13
        assert result == 13

    def test_repeat2_further_decay(self):
        result = scale_chunks_for_density(16, 2, 0.8, 1)
        # 16 * 0.8^2 = 16 * 0.64 = 10.24 → 10
        assert result == 10

    def test_never_below_min_chunks(self):
        result = scale_chunks_for_density(1, 10, 0.5, min_chunks=2)
        # 1 * 0.5^10 ≈ 0.0009 → 0, but floor is 2
        assert result >= 2

    def test_zero_base_returns_min(self):
        result = scale_chunks_for_density(0, 0, 0.8, min_chunks=3)
        assert result == 3


class TestUpdateCoverage:
    """Test coverage state updates."""

    def test_adds_new_range(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[])
        update_coverage(state, 0, 100)
        assert state.covered_ranges == [(0, 100)]

    def test_merges_overlapping_ranges(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[(0, 100)])
        update_coverage(state, 50, 100)  # (50, 150)
        # Should merge to (0, 150)
        assert state.covered_ranges == [(0, 150)]

    def test_merges_adjacent_ranges(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[(0, 100), (100, 200)])
        # They're already adjacent, should stay merged
        assert state.covered_ranges == [(0, 100), (100, 200)]

    def test_coverage_fraction_correct(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[])
        frac = update_coverage(state, 0, 500)
        assert frac == pytest.approx(0.5)

    def test_full_coverage_is_one(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[(0, 500)])
        frac = update_coverage(state, 500, 500)
        assert frac == pytest.approx(1.0)

    def test_zero_length_no_change(self):
        state = CoverageState(thinking_text_len=1000, covered_ranges=[(0, 100)])
        frac = update_coverage(state, 200, 0)
        assert frac == pytest.approx(0.1)
        assert state.covered_ranges == [(0, 100)]


class TestCoverageFraction:
    """Test coverage computation."""

    def test_empty_coverage_is_zero(self):
        frac = coverage_fraction([], 1000)
        assert frac == 0.0

    def test_full_coverage_is_one(self):
        frac = coverage_fraction([(0, 1000)], 1000)
        assert frac == 1.0

    def test_partial_coverage(self):
        frac = coverage_fraction([(0, 250), (250, 500)], 1000)
        assert frac == pytest.approx(0.5)

    def test_zero_total_len_handles_gracefully(self):
        frac = coverage_fraction([(0, 100)], 0)
        assert frac == 0.0


class TestRegionPlanIntegration:
    """Integration tests for full region planning flow."""

    def test_sequential_regions_tile_block(self):
        """Simulate 4 sequential repeats covering a 1000-char thinking block."""
        state = CoverageState(thinking_text_len=1000, covered_ranges=[])
        coverage_per_repeat = []

        for repeat_idx in range(4):
            plan = compute_region_plan(
                thinking_len=1000,
                repeat_index=repeat_idx,
                num_regions=4,
                density_decay=0.8,
                coverage_threshold=0.9,
                state=state,
            )
            # After each region is used, update coverage
            cov_frac = update_coverage(state, plan.offset_chars, plan.length_chars)
            coverage_per_repeat.append(cov_frac)

        # Coverage should grow: ~0.25 → ~0.5 → ~0.75 → ~1.0
        assert coverage_per_repeat[0] == pytest.approx(0.25)
        assert coverage_per_repeat[1] == pytest.approx(0.5)
        assert coverage_per_repeat[2] == pytest.approx(0.75)
        assert coverage_per_repeat[3] == pytest.approx(1.0)

    def test_diminishing_density_chunks(self):
        """Verify chunk counts decrease across repeats."""
        chunks_per_repeat = []
        for rep_idx in range(5):
            n_chunks = scale_chunks_for_density(
                base_chunks=16, repeat_index=rep_idx, decay=0.8, min_chunks=1
            )
            chunks_per_repeat.append(n_chunks)

        # Should be monotonically non-increasing
        for i in range(1, len(chunks_per_repeat)):
            assert chunks_per_repeat[i] <= chunks_per_repeat[i - 1]
