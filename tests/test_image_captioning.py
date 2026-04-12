"""Tests for ImageCaptioning task type (task 8.3)."""

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


def test_image_captioning_registered():
    assert "image_captioning" in TaskType.list_types()


def test_image_captioning_instantiable():
    assert TaskType.get("image_captioning") is not None


# ── Config & ChatParams ───────────────────────────────────────────────


def test_image_captioning_config():
    t = TaskType.get("image_captioning")
    cfg = t.config()
    assert "temperature" in cfg
    assert "caption_max_tokens" in cfg
    assert cfg["caption_max_tokens"] > 0


def test_image_captioning_no_thinking():
    t = TaskType.get("image_captioning")
    cp = t.chat_params()
    assert not cp.enable_thinking
    assert cp.system_prompt


# ── Rollout ───────────────────────────────────────────────────────────


def _make_ctx(caption_text: str = "A black cat sits on a mat."):
    from unittest.mock import MagicMock

    gen_out = MagicMock()
    gen_out.text = caption_text
    gen_out.stop_reason = "eos"

    ctx = MagicMock()
    ctx.type_config = {"temperature": 0.6, "caption_max_tokens": 128}
    ctx.generate.return_value = gen_out
    return ctx


def test_captioning_rollout_calls_generate():
    t = TaskType.get("image_captioning")
    ctx = _make_ctx()
    t.rollout(ctx)
    ctx.generate.assert_called_once()


def test_captioning_rollout_tags_caption():
    t = TaskType.get("image_captioning")
    ctx = _make_ctx()
    t.rollout(ctx)
    call_kwargs = ctx.generate.call_args[1]
    assert call_kwargs.get("tag") == "caption"


def test_captioning_rollout_captures_caption_var():
    t = TaskType.get("image_captioning")
    ctx = _make_ctx()
    t.rollout(ctx)
    ctx.capture.assert_called_once()
    assert ctx.capture.call_args[0][0] == "caption_var"


# ── Reward ────────────────────────────────────────────────────────────


def _make_result(text: str):
    from unittest.mock import MagicMock

    r = MagicMock()
    r.completion_text = text
    return r


def test_captioning_reward_perfect():
    t = TaskType.get("image_captioning")
    result = _make_result("a black cat")
    assert t.reward(result, "a black cat") == pytest.approx(1.0)


def test_captioning_reward_no_answer_zero():
    t = TaskType.get("image_captioning")
    result = _make_result("a cat")
    assert t.reward(result, None) == 0.0
    assert t.reward(result, "") == 0.0


def test_captioning_reward_partial():
    t = TaskType.get("image_captioning")
    result = _make_result("black cat")
    score = t.reward(result, "a big black cat on a mat")
    assert 0.0 < score < 1.0


def test_captioning_reward_multiple_references():
    """Semicolon-separated references → max ROUGE-L used."""
    t = TaskType.get("image_captioning")
    result = _make_result("a sleeping cat")
    # Second reference matches better
    score = t.reward(result, "a sitting dog; a sleeping cat")
    assert score == pytest.approx(1.0)


def test_captioning_reward_best_of_multiple():
    """Best-matching reference wins."""
    t = TaskType.get("image_captioning")
    result = _make_result("big red apple on table")
    ref1 = "small green apple"
    ref2 = "big red apple on the table"
    score_single = t.reward(result, ref2)
    score_multi = t.reward(result, f"{ref1}; {ref2}")
    assert score_multi >= score_single - 1e-9  # multi ≥ single best


def test_captioning_reward_semicolon_empty_refs():
    """Whitespace-only semicolon entries are ignored."""
    t = TaskType.get("image_captioning")
    result = _make_result("a cat")
    score = t.reward(result, "a cat;  ; ")
    assert score == pytest.approx(1.0)
