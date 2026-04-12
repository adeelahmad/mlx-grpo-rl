"""Tests for VQA task type (task 7.3)."""

from __future__ import annotations

import pytest

from mlx_grpo.types.base import TaskType


def _ensure_builtins():
    import mlx_grpo.rewards.vision  # noqa: F401
    import mlx_grpo.types.builtins  # noqa: F401


@pytest.fixture(autouse=True)
def _register():
    _ensure_builtins()


# ── Registration ──────────────────────────────────────────────────────


def test_vqa_registered():
    assert "vqa" in TaskType.list_types()


def test_vqa_instantiable():
    t = TaskType.get("vqa")
    assert t is not None


# ── Config & ChatParams ───────────────────────────────────────────────


def test_vqa_config():
    t = TaskType.get("vqa")
    cfg = t.config()
    assert "temperature" in cfg
    assert "answer_max_tokens" in cfg
    assert cfg["answer_max_tokens"] > 0


def test_vqa_chat_params():
    t = TaskType.get("vqa")
    cp = t.chat_params()
    assert not cp.enable_thinking  # VQA answers are short — no thinking
    assert cp.system_prompt


# ── Rollout ───────────────────────────────────────────────────────────


def _make_ctx(answer_text: str = "cat"):
    """Build a minimal mock RolloutContext."""
    from unittest.mock import MagicMock

    gen_out = MagicMock()
    gen_out.text = answer_text
    gen_out.stop_reason = "eos"

    ctx = MagicMock()
    ctx.type_config = {"temperature": 0.3, "answer_max_tokens": 64}
    ctx.generate.return_value = gen_out
    return ctx


def test_vqa_rollout_calls_generate():
    t = TaskType.get("vqa")
    ctx = _make_ctx()
    t.rollout(ctx)
    ctx.generate.assert_called_once()


def test_vqa_rollout_tags_answer():
    t = TaskType.get("vqa")
    ctx = _make_ctx()
    t.rollout(ctx)
    call_kwargs = ctx.generate.call_args[1]
    assert call_kwargs.get("tag") == "answer"


def test_vqa_rollout_captures_answer_var():
    t = TaskType.get("vqa")
    ctx = _make_ctx()
    t.rollout(ctx)
    ctx.capture.assert_called_once()
    capture_args = ctx.capture.call_args[0]
    assert capture_args[0] == "answer_var"


# ── Reward ────────────────────────────────────────────────────────────


def _make_result(text: str):
    from unittest.mock import MagicMock

    r = MagicMock()
    r.completion_text = text
    return r


def test_vqa_reward_exact_single_token():
    """Single-token reference → uses exact match reward."""
    t = TaskType.get("vqa")
    result = _make_result("cat")
    assert t.reward(result, "cat") == pytest.approx(1.0)


def test_vqa_reward_exact_mismatch():
    t = TaskType.get("vqa")
    result = _make_result("dog")
    assert t.reward(result, "cat") == 0.0


def test_vqa_reward_multi_token_f1():
    """Multi-token reference → uses token F1 reward."""
    t = TaskType.get("vqa")
    result = _make_result("a black cat")
    score = t.reward(result, "a black cat")
    assert score == pytest.approx(1.0)


def test_vqa_reward_no_answer_zero():
    t = TaskType.get("vqa")
    result = _make_result("cat")
    assert t.reward(result, None) == 0.0
    assert t.reward(result, "") == 0.0


def test_vqa_reward_partial_match():
    t = TaskType.get("vqa")
    result = _make_result("black cat")
    score = t.reward(result, "big black cat")
    assert 0.0 < score < 1.0
