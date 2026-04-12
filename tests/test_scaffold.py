"""Tests for multi-view contrastive scaffold.

Covers: ViewConfig, PerGroupConfig, get_view_scaffold_ratio.
"""

import tempfile
import textwrap
from pathlib import Path

import pytest

from mlx_grpo.scaffold import (
    ScaffoldConfig,
    ViewConfig,
    build_scaffold_config,
    get_view_scaffold_ratio,
)

# ── Helpers ──


def _make_view(scaffold_position: str = "prefix", end_progress: float = 0.3) -> ViewConfig:
    return ViewConfig(
        name="test",
        scaffold_position=scaffold_position,
        truncate_from="prefix",
        used_tokens=64,
        end_progress=end_progress,
    )


# ── get_view_scaffold_ratio ──


class TestGetViewScaffoldRatio:
    def test_progress_zero_returns_one(self):
        v = _make_view("prefix", end_progress=0.3)
        assert get_view_scaffold_ratio(v, 0.0) == pytest.approx(1.0)

    def test_progress_at_end_returns_zero(self):
        v = _make_view("prefix", end_progress=0.3)
        assert get_view_scaffold_ratio(v, 0.3) == pytest.approx(0.0)

    def test_progress_beyond_end_returns_zero(self):
        v = _make_view("prefix", end_progress=0.3)
        assert get_view_scaffold_ratio(v, 0.5) == pytest.approx(0.0)

    def test_none_position_returns_zero(self):
        v = _make_view("none", end_progress=0.3)
        assert get_view_scaffold_ratio(v, 0.0) == pytest.approx(0.0)

    def test_halfway_decay_rate_one(self):
        v = _make_view("prefix", end_progress=0.4)
        # At progress=0.2 with decay_rate=1.0: (1 - 0.2/0.4)^1 = 0.5
        assert get_view_scaffold_ratio(v, 0.2) == pytest.approx(0.5)

    def test_decay_rate_two(self):
        v = ViewConfig("t", "prefix", "prefix", 64, decay_rate=2.0, end_progress=0.4)
        # At progress=0.2 with decay_rate=2.0: (1 - 0.2/0.4)^2 = 0.25
        assert get_view_scaffold_ratio(v, 0.2) == pytest.approx(0.25)

    def test_zero_end_progress_returns_zero(self):
        v = ViewConfig("t", "prefix", "prefix", 64, end_progress=0.0)
        assert get_view_scaffold_ratio(v, 0.0) == pytest.approx(0.0)

    def test_postfix_position_decays_normally(self):
        v = _make_view("postfix", end_progress=0.6)
        ratio = get_view_scaffold_ratio(v, 0.0)
        assert ratio == pytest.approx(1.0)
        ratio_mid = get_view_scaffold_ratio(v, 0.3)
        assert 0.0 < ratio_mid < 1.0


# ── Baked-in ratio technique ──


class TestBakedInRatio:
    def test_fixed_ratio_unchanged_by_progress(self):
        """ScaffoldConfig(initial_ratio=0.5, end_progress=2.0, decay_rate=0.0) returns 0.5."""
        sc = ScaffoldConfig(initial_ratio=0.5, end_progress=2.0, decay_rate=0.0, min_ratio=0.0)
        # At any progress from 0 to 1, ratio should stay 0.5
        # (1 - progress/2.0)^0.0 = 1.0, so base = 0.5 * 1.0 = 0.5
        assert sc.effective_ratio(0.0) == pytest.approx(0.5)
        assert sc.effective_ratio(0.5) == pytest.approx(0.5)
        assert sc.effective_ratio(0.99) == pytest.approx(0.5)

    def test_zero_ratio_is_zero(self):
        sc = ScaffoldConfig(initial_ratio=0.0, end_progress=2.0, decay_rate=0.0, min_ratio=0.0)
        assert sc.effective_ratio(0.0) == pytest.approx(0.0)
        assert sc.effective_ratio(0.5) == pytest.approx(0.0)


# ── build_scaffold_config parses per_group_config ──


class TestBuildScaffoldConfigPerGroup:
    def _pgc_data(self) -> dict:
        return {
            "per_group_config": {
                "target": "thinking",
                "views": [
                    {
                        "name": "entry",
                        "scaffold_position": "prefix",
                        "truncate_from": "prefix",
                        "used_tokens": 96,
                        "end_progress": 0.3,
                    },
                    {
                        "name": "naked",
                        "scaffold_position": "none",
                        "truncate_from": "postfix",
                        "used_tokens": 128,
                        "end_progress": 0.3,
                    },
                ],
            }
        }

    def test_parses_per_group_config(self):
        sc = build_scaffold_config(self._pgc_data())
        assert sc.per_group_config is not None
        pgc = sc.per_group_config
        assert pgc.target == "thinking"
        assert len(pgc.views) == 2

    def test_view_fields(self):
        sc = build_scaffold_config(self._pgc_data())
        v0 = sc.per_group_config.views[0]  # type: ignore[union-attr]
        assert v0.name == "entry"
        assert v0.scaffold_position == "prefix"
        assert v0.truncate_from == "prefix"
        assert v0.used_tokens == 96
        assert v0.end_progress == pytest.approx(0.3)

    def test_naked_view(self):
        sc = build_scaffold_config(self._pgc_data())
        v1 = sc.per_group_config.views[1]  # type: ignore[union-attr]
        assert v1.name == "naked"
        assert v1.scaffold_position == "none"
        assert get_view_scaffold_ratio(v1, 0.0) == pytest.approx(0.0)

    def test_none_when_missing(self):
        sc = build_scaffold_config({})
        assert sc.per_group_config is None

    def test_defaults_propagate(self):
        defaults = ScaffoldConfig(mode="middle")
        sc = build_scaffold_config(self._pgc_data(), defaults=defaults)
        # mode not overridden — inherits from defaults? No, per_group_config doesn't touch mode.
        # Actually build_scaffold_config uses data.get("mode", base.mode),
        # so if "mode" not in data, it inherits base.mode.
        assert sc.mode == "middle"
        assert sc.per_group_config is not None


# ── Config validation ──


class TestConfigValidation:
    def _write_yml(self, content: str) -> Path:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(textwrap.dedent(content))
            return Path(f.name)

    def test_mismatched_views_raises(self):
        from mlx_grpo.config.loader import load_config

        yml = self._write_yml("""
            training:
              group_size: 4
              gradient_accumulation_steps: 4
              samples_per_step: 1
            scaffold:
              per_group_config:
                target: thinking
                views:
                  - name: v1
                    scaffold_position: prefix
                    truncate_from: prefix
                    used_tokens: 64
                  - name: v2
                    scaffold_position: none
                    truncate_from: postfix
                    used_tokens: 128
        """)
        with pytest.raises(ValueError, match="views has 2 entries but training.group_size=4"):
            load_config(yml)

    def test_any_accum_steps_allowed_with_per_group_config(self):
        """gas != group_size is now permitted — restriction was removed."""
        from mlx_grpo.config.loader import load_config

        yml = self._write_yml("""
            training:
              group_size: 2
              gradient_accumulation_steps: 1
              samples_per_step: 1
            scaffold:
              per_group_config:
                target: thinking
                views:
                  - name: v1
                    scaffold_position: prefix
                    truncate_from: prefix
                    used_tokens: 64
                  - name: v2
                    scaffold_position: none
                    truncate_from: postfix
                    used_tokens: 128
        """)
        cfg = load_config(yml)  # must NOT raise
        assert cfg.training.gradient_accumulation_steps == 1

    def test_wrong_samples_per_step_raises(self):
        from mlx_grpo.config.loader import load_config

        yml = self._write_yml("""
            training:
              group_size: 2
              gradient_accumulation_steps: 2
              samples_per_step: 2
            scaffold:
              per_group_config:
                target: thinking
                views:
                  - name: v1
                    scaffold_position: prefix
                    truncate_from: prefix
                    used_tokens: 64
                  - name: v2
                    scaffold_position: none
                    truncate_from: postfix
                    used_tokens: 128
        """)
        with pytest.raises(ValueError, match="samples_per_step"):
            load_config(yml)

    def test_valid_config_passes(self):
        from mlx_grpo.config.loader import load_config

        yml = self._write_yml("""
            training:
              group_size: 2
              gradient_accumulation_steps: 2
              samples_per_step: 1
            scaffold:
              per_group_config:
                target: thinking
                views:
                  - name: v1
                    scaffold_position: prefix
                    truncate_from: prefix
                    used_tokens: 64
                  - name: v2
                    scaffold_position: none
                    truncate_from: postfix
                    used_tokens: 128
        """)
        config = load_config(yml)
        assert config.scaffold.per_group_config is not None
        assert len(config.scaffold.per_group_config.views) == 2
