"""Extended tests for scaffold.py — ScaffoldConfig and effective_tokens."""

from __future__ import annotations

import pytest

from mlx_grpo.scaffold import (
    ScaffoldConfig,
    ViewConfig,
    build_scaffold_config,
    get_view_scaffold_ratio,
)

# ── ScaffoldConfig.effective_ratio ────────────────────────────────────────────


class TestEffectiveRatio:
    def test_before_end_progress(self):
        sc = ScaffoldConfig(initial_ratio=1.0, end_progress=0.8)
        ratio = sc.effective_ratio(0.0)
        assert ratio > 0.0

    def test_at_end_progress(self):
        sc = ScaffoldConfig(initial_ratio=1.0, end_progress=0.8)
        ratio = sc.effective_ratio(0.8)
        assert ratio == pytest.approx(0.0)

    def test_past_end_progress(self):
        sc = ScaffoldConfig(initial_ratio=1.0, end_progress=0.8)
        ratio = sc.effective_ratio(0.9)
        assert ratio == pytest.approx(0.0)

    def test_zero_end_progress(self):
        sc = ScaffoldConfig(initial_ratio=1.0, end_progress=0.0)
        ratio = sc.effective_ratio(0.0)
        assert ratio == pytest.approx(0.0)

    def test_with_levels(self):
        sc = ScaffoldConfig(levels=(1.0, 0.5, 0.0), initial_ratio=1.0, end_progress=0.8)
        r0 = sc.effective_ratio(0.0, group_index=0)
        r1 = sc.effective_ratio(0.0, group_index=1)
        r2 = sc.effective_ratio(0.0, group_index=2)
        assert r0 > r1 > r2
        assert r2 == pytest.approx(0.0)

    def test_levels_cycles(self):
        sc = ScaffoldConfig(levels=(1.0, 0.5), initial_ratio=1.0, end_progress=0.8)
        r0 = sc.effective_ratio(0.0, group_index=0)
        r2 = sc.effective_ratio(0.0, group_index=2)
        # group 2 % 2 = 0 → same as group 0
        assert r0 == pytest.approx(r2)


# ── ScaffoldConfig.effective_tokens ──────────────────────────────────────────


class TestEffectiveTokens:
    def test_basic_answer_count(self):
        sc = ScaffoldConfig(initial_ratio=0.5, end_progress=0.8)
        tokens = sc.effective_tokens(0.0, 100)
        assert tokens == pytest.approx(50, abs=5)

    def test_max_scaffolded_tokens_priority(self):
        sc = ScaffoldConfig(max_scaffolded_tokens=50, initial_ratio=1.0, end_progress=0.8)
        tokens = sc.effective_tokens(0.0, 1000)
        assert tokens <= 50

    def test_max_tokens_fallback(self):
        sc = ScaffoldConfig(max_tokens=30, initial_ratio=1.0, end_progress=0.8)
        tokens = sc.effective_tokens(0.0, 5)
        # max_tokens=30 applies when max_scaffolded_tokens is None
        assert tokens <= 30

    def test_zero_when_past_end_progress(self):
        sc = ScaffoldConfig(initial_ratio=1.0, end_progress=0.5)
        tokens = sc.effective_tokens(0.9, 100)
        assert tokens == 0

    def test_group_index_affects_tokens(self):
        sc = ScaffoldConfig(levels=(1.0, 0.0), initial_ratio=1.0, end_progress=0.8)
        t0 = sc.effective_tokens(0.0, 100, group_index=0)
        t1 = sc.effective_tokens(0.0, 100, group_index=1)
        assert t0 > 0
        assert t1 == 0


# ── ScaffoldConfig properties ────────────────────────────────────────────────


class TestRatioFactors:
    def test_thinking_ratio_factor_default(self):
        sc = ScaffoldConfig(target="both")
        assert sc.thinking_ratio_factor == 1.0

    def test_thinking_ratio_factor_answer_only(self):
        sc = ScaffoldConfig(target="answer")
        assert sc.thinking_ratio_factor == 0.0

    def test_answer_ratio_factor_default(self):
        sc = ScaffoldConfig(target="both")
        assert sc.answer_ratio_factor == 1.0

    def test_answer_ratio_factor_thinking_only(self):
        sc = ScaffoldConfig(target="thinking")
        assert sc.answer_ratio_factor == 0.0


# ── ScaffoldConfig.dynamic_budget ────────────────────────────────────────────


class TestDynamicBudget:
    def test_no_dynamic_returns_base(self):
        sc = ScaffoldConfig(dynamic_max_tokens=False)
        result = sc.dynamic_budget(0.0, 2048, 100, 0)
        assert result == 2048

    def test_dynamic_adds_headroom(self):
        sc = ScaffoldConfig(
            dynamic_max_tokens=True,
            max_scaffolded_tokens=200,
            initial_ratio=0.5,
            end_progress=0.8,
        )
        # Less scaffold → more generation room
        result = sc.dynamic_budget(0.0, 2048, 100, 0)
        assert result >= 2048


# ── build_scaffold_config ────────────────────────────────────────────────────


class TestBuildScaffoldConfig:
    def test_none_data_returns_defaults(self):
        defaults = ScaffoldConfig(initial_ratio=0.5)
        result = build_scaffold_config(None, defaults)
        assert result.initial_ratio == pytest.approx(0.5)

    def test_overrides_mode(self):
        result = build_scaffold_config({"mode": "suffix"})
        assert result.mode == "suffix"

    def test_overrides_initial_ratio(self):
        result = build_scaffold_config({"initial_ratio": 0.7})
        assert result.initial_ratio == pytest.approx(0.7)

    def test_levels_from_list(self):
        result = build_scaffold_config({"levels": [1.0, 0.5, 0.0]})
        assert result.levels == (1.0, 0.5, 0.0)

    def test_max_scaffolded_tokens(self):
        result = build_scaffold_config({"max_scaffolded_tokens": 128})
        assert result.max_scaffolded_tokens == 128

    def test_per_group_config(self):
        data = {
            "per_group_config": {
                "target": "thinking",
                "views": [
                    {
                        "name": "v1",
                        "scaffold_position": "prefix",
                        "truncate_from": "prefix",
                        "used_tokens": 64,
                    },
                ],
            }
        }
        result = build_scaffold_config(data)
        assert result.per_group_config is not None
        assert result.per_group_config.target == "thinking"
        assert len(result.per_group_config.views) == 1

    def test_empty_data_returns_base(self):
        base = ScaffoldConfig()
        result = build_scaffold_config({}, base)
        assert result.mode == base.mode


# ── get_view_scaffold_ratio ──────────────────────────────────────────────────


class TestGetViewScaffoldRatio:
    def test_prefix_before_end(self):
        v = ViewConfig(
            name="v1",
            scaffold_position="prefix",
            truncate_from="prefix",
            used_tokens=64,
            end_progress=0.5,
        )
        ratio = get_view_scaffold_ratio(v, progress=0.0)
        assert ratio == pytest.approx(1.0)

    def test_none_position_returns_zero(self):
        v = ViewConfig(
            name="v1",
            scaffold_position="none",
            truncate_from="prefix",
            used_tokens=64,
        )
        ratio = get_view_scaffold_ratio(v, progress=0.0)
        assert ratio == pytest.approx(0.0)

    def test_past_end_returns_zero(self):
        v = ViewConfig(
            name="v1",
            scaffold_position="prefix",
            truncate_from="prefix",
            used_tokens=64,
            end_progress=0.3,
        )
        ratio = get_view_scaffold_ratio(v, progress=0.5)
        assert ratio == pytest.approx(0.0)

    def test_decay_rate(self):
        v = ViewConfig(
            name="v1",
            scaffold_position="prefix",
            truncate_from="prefix",
            used_tokens=64,
            end_progress=0.8,
            decay_rate=2.0,
        )
        ratio = get_view_scaffold_ratio(v, progress=0.4)
        assert 0.0 < ratio < 1.0
