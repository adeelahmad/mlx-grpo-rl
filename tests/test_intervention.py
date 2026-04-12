"""Tests for the pluggable rollout intervention hook (training/intervention.py)."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from mlx_grpo.config.escalation import InterventionConfig
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training.intervention import (
    GroupMemberRecord,
    _handler_cache,
    _load_handler,
    build_group_progress,
    call_intervention,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _seg(tag: str = "answer") -> Segment:
    return Segment(
        tokens=[1, 2], text="text", source=SegmentSource.GENERATED, masked=False, tag=tag
    )


def _rollout(**meta) -> RolloutResult:
    r = RolloutResult(segments=[_seg()])
    r.metadata.update(meta)
    return r


def _clear_cache():
    """Clear the module-level handler cache between tests."""
    _handler_cache.clear()


# ── InterventionConfig defaults ─────────────────────────────────────────────


def test_intervention_config_defaults():
    cfg = InterventionConfig()
    assert cfg.enabled is False
    assert cfg.lambda_expr == ""
    assert cfg.script == ""
    assert cfg.callable_path == ""
    assert cfg.trigger == "last_member"
    assert cfg.masked is True


def test_intervention_config_custom():
    cfg = InterventionConfig(
        enabled=True,
        lambda_expr="lambda p, a, g, prog: 'hint'",
        trigger="always",
        masked=False,
    )
    assert cfg.enabled is True
    assert cfg.trigger == "always"
    assert cfg.masked is False


# ── _load_handler: disabled ─────────────────────────────────────────────────


def test_load_handler_disabled_returns_none():
    cfg = InterventionConfig(enabled=False, lambda_expr="lambda p,a,g,prog: 'x'")
    assert _load_handler(cfg) is None


def test_load_handler_no_source_returns_none():
    """enabled=True but no source → None."""
    cfg = InterventionConfig(enabled=True)
    assert _load_handler(cfg) is None


# ── _load_handler: lambda_expr ──────────────────────────────────────────────


def test_load_handler_lambda_expr():
    _clear_cache()
    cfg = InterventionConfig(enabled=True, lambda_expr="lambda p, a, g, prog: 'lambda-hint'")
    handler = _load_handler(cfg)
    assert handler is not None
    result = handler("prompt", "answer", "gt", [])
    assert result == "lambda-hint"


def test_load_handler_lambda_with_progress():
    _clear_cache()
    expr = "lambda p, a, g, prog: f'members={len(prog)}'"
    cfg = InterventionConfig(enabled=True, lambda_expr=expr)
    handler = _load_handler(cfg)
    assert handler is not None
    assert handler("p", None, None, ["x", "y"]) == "members=2"


def test_load_handler_lambda_none_return():
    _clear_cache()
    cfg = InterventionConfig(enabled=True, lambda_expr="lambda p, a, g, prog: None")
    handler = _load_handler(cfg)
    assert handler is not None
    assert handler("p", "a", "g", []) is None


# ── _load_handler: script ───────────────────────────────────────────────────


def test_load_handler_script(tmp_path):
    _clear_cache()
    script = tmp_path / "my_handler.py"
    script.write_text(
        textwrap.dedent("""
        def handle(prompt, answer, ground_truth, group_progress):
            return f"script:{prompt}"
    """)
    )
    cfg = InterventionConfig(enabled=True, script=str(script))
    handler = _load_handler(cfg)
    assert handler is not None
    assert handler("Q1", None, None, []) == "script:Q1"


def test_load_handler_script_cached(tmp_path):
    """Second call with same script path returns cached handler."""
    _clear_cache()
    script = tmp_path / "cached_handler.py"
    script.write_text("def handle(p, a, g, prog): return 'cached'")
    cfg = InterventionConfig(enabled=True, script=str(script))
    h1 = _load_handler(cfg)
    h2 = _load_handler(cfg)
    assert h1 is h2  # same object from cache


def test_load_handler_script_missing_handle_raises(tmp_path):
    _clear_cache()
    script = tmp_path / "no_handle.py"
    script.write_text("def not_handle(): pass")
    cfg = InterventionConfig(enabled=True, script=str(script))
    with pytest.raises(AttributeError, match="handle"):
        _load_handler(cfg)


def test_load_handler_script_nonexistent_raises():
    _clear_cache()
    cfg = InterventionConfig(enabled=True, script="/nonexistent/path/handler.py")
    with pytest.raises((ImportError, FileNotFoundError)):
        _load_handler(cfg)


# ── _load_handler: callable_path ────────────────────────────────────────────


def test_load_handler_callable_path_colon(monkeypatch):
    """callable_path using colon separator: 'module:function'."""
    _clear_cache()
    mock_fn = MagicMock(return_value="callable-result")
    mock_mod = MagicMock()
    mock_mod.my_fn = mock_fn

    with patch("importlib.import_module", return_value=mock_mod) as mock_import:
        cfg = InterventionConfig(enabled=True, callable_path="my.module:my_fn")
        handler = _load_handler(cfg)
        mock_import.assert_called_once_with("my.module")
        assert handler is mock_fn


def test_load_handler_callable_path_cached(monkeypatch):
    """Second call with same callable_path returns cached handler."""
    _clear_cache()
    mock_fn = MagicMock(return_value="x")
    mock_mod = MagicMock()
    mock_mod.fn = mock_fn

    with patch("importlib.import_module", return_value=mock_mod):
        cfg = InterventionConfig(enabled=True, callable_path="some.module:fn")
        h1 = _load_handler(cfg)
        h2 = _load_handler(cfg)
    assert h1 is h2


# ── call_intervention ───────────────────────────────────────────────────────


def test_call_intervention_disabled_returns_none():
    cfg = InterventionConfig(enabled=False)
    result = call_intervention(cfg, "p", "a", "g", [])
    assert result is None


def test_call_intervention_success():
    _clear_cache()
    cfg = InterventionConfig(enabled=True, lambda_expr="lambda p, a, g, prog: 'hint'")
    result = call_intervention(cfg, "prompt", "answer", "gt", [])
    assert result == "hint"


def test_call_intervention_handler_exception_returns_none(capsys):
    """Handler raising an exception → None returned, not propagated."""
    _clear_cache()
    cfg = InterventionConfig(enabled=True, lambda_expr="lambda p, a, g, prog: 1/0")
    result = call_intervention(cfg, "p", "a", "g", [])
    assert result is None
    # Should print a warning
    captured = capsys.readouterr()
    assert "handler raised" in captured.out or "ZeroDivisionError" in captured.out


def test_call_intervention_passes_all_args():
    _clear_cache()
    expr = "lambda p, a, g, prog: str({'p': p, 'a': a, 'g': g, 'n': len(prog)})"
    cfg = InterventionConfig(enabled=True, lambda_expr=expr)
    result = call_intervention(cfg, "myprompt", "myanswer", "mygt", ["x", "y"])
    assert "myprompt" in result
    assert "myanswer" in result
    assert "mygt" in result
    assert "2" in result  # len(progress) = 2


def test_call_intervention_no_handler_returns_none():
    """enabled=True but no source configured → None (no handler to call)."""
    _clear_cache()
    cfg = InterventionConfig(enabled=True)  # no lambda/script/callable
    result = call_intervention(cfg, "p", "a", "g", [])
    assert result is None


# ── build_group_progress ─────────────────────────────────────────────────────


def test_build_group_progress_empty():
    result = build_group_progress([])
    assert result == []


def test_build_group_progress_single():
    r = _rollout(_esc_reward=0.75, _esc_member_idx=0, _view_is_naked=False)
    records = build_group_progress([r])
    assert len(records) == 1
    rec = records[0]
    assert rec.reward == pytest.approx(0.75)
    assert rec.member_idx == 0
    assert rec.is_naked is False
    assert rec.rollout is r


def test_build_group_progress_naked_rollout():
    r = _rollout(_esc_reward=1.0, _esc_member_idx=2, _view_is_naked=True)
    records = build_group_progress([r])
    assert records[0].is_naked is True
    assert records[0].reward == pytest.approx(1.0)
    assert records[0].member_idx == 2


def test_build_group_progress_multiple():
    rollouts = [
        _rollout(_esc_reward=0.0, _esc_member_idx=0, _view_is_naked=False),
        _rollout(_esc_reward=0.5, _esc_member_idx=1, _view_is_naked=False),
        _rollout(_esc_reward=1.0, _esc_member_idx=2, _view_is_naked=True),
    ]
    records = build_group_progress(rollouts)
    assert len(records) == 3
    assert records[0].reward == pytest.approx(0.0)
    assert records[1].reward == pytest.approx(0.5)
    assert records[2].is_naked is True


def test_build_group_progress_missing_metadata_defaults():
    """Rollouts without _esc_* metadata → defaults: reward=0.0, idx=0, naked=False."""
    r = _rollout()  # no _esc_* keys
    records = build_group_progress([r])
    assert records[0].reward == pytest.approx(0.0)
    assert records[0].member_idx == 0
    assert records[0].is_naked is False


def test_build_group_progress_reward_type():
    """Reward must be float (coerced from metadata int or str)."""
    r = _rollout(_esc_reward=1)  # int in metadata
    records = build_group_progress([r])
    assert isinstance(records[0].reward, float)


# ── GroupMemberRecord ────────────────────────────────────────────────────────


def test_group_member_record_fields():
    r = _rollout()
    rec = GroupMemberRecord(rollout=r, reward=0.42, member_idx=3, is_naked=True)
    assert rec.rollout is r
    assert rec.reward == pytest.approx(0.42)
    assert rec.member_idx == 3
    assert rec.is_naked is True
