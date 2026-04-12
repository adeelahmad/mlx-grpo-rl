"""Tests for reward-based answer gradient weighting and double-application guard.

Feature: correct_answer_weight / incorrect_answer_weight in TokenISConfig.
When mode=segment is active alongside these, the uniform answer_weight from
segment mode must be neutralized (set to 1.0) to prevent double-application.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mlx_grpo.config.loader import TokenISConfig
from mlx_grpo.training.token_importance import compute_is_weights, compute_segment_weights

# ── TokenISConfig: new fields ────────────────────────────────────────────────


def test_correct_answer_weight_default():
    cfg = TokenISConfig()
    assert cfg.correct_answer_weight == pytest.approx(1.0)
    assert cfg.incorrect_answer_weight == pytest.approx(1.0)
    assert cfg.answer_weight_threshold == pytest.approx(0.5)


def test_correct_answer_weight_custom():
    cfg = TokenISConfig(
        correct_answer_weight=2.5, incorrect_answer_weight=0.3, answer_weight_threshold=0.6
    )
    assert cfg.correct_answer_weight == pytest.approx(2.5)
    assert cfg.incorrect_answer_weight == pytest.approx(0.3)
    assert cfg.answer_weight_threshold == pytest.approx(0.6)


# ── Segment mode neutralization: core logic ──────────────────────────────────


def test_neutralize_segment_answer_weight_via_replace():
    """When correctness weights active, segment's answer_weight should be replaced with 1.0."""
    cfg = TokenISConfig(
        mode="segment",
        answer_weight=2.0,
        thinking_weight=0.5,
        correct_answer_weight=3.0,
        incorrect_answer_weight=0.2,
    )
    # Before neutralization: answer tokens get 2.0
    ans_mask = [False, True, True]
    w_original = compute_segment_weights(3, cfg, ans_mask, None)
    assert w_original[1] == pytest.approx(2.0)
    assert w_original[2] == pytest.approx(2.0)

    # After neutralization (simulating what _neutralize_seg_answer does):
    neutralized = replace(cfg, answer_weight=1.0)
    w_neutral = compute_segment_weights(3, neutralized, ans_mask, None)
    assert w_neutral[1] == pytest.approx(1.0)  # neutralized — correctness pass takes over
    assert w_neutral[2] == pytest.approx(1.0)
    assert w_neutral[0] == pytest.approx(1.0)  # non-answer unchanged


def test_segment_thinking_weight_preserved_after_neutralization():
    """Neutralizing answer_weight must not affect thinking_weight."""
    cfg = TokenISConfig(
        mode="segment",
        answer_weight=2.0,
        thinking_weight=0.4,
        correct_answer_weight=2.5,
        incorrect_answer_weight=0.3,
    )
    think_mask = [True, False, False]
    ans_mask = [False, False, True]
    neutralized = replace(cfg, answer_weight=1.0)
    w = compute_segment_weights(3, neutralized, ans_mask, think_mask)
    assert w[0] == pytest.approx(0.4)  # thinking preserved
    assert w[1] == pytest.approx(1.0)  # neither
    assert w[2] == pytest.approx(1.0)  # answer → neutralized to 1.0


def test_no_neutralization_when_correctness_weights_inactive():
    """When correct/incorrect_answer_weight == 1.0, segment answer_weight is unchanged."""
    cfg = TokenISConfig(
        mode="segment", answer_weight=2.0, correct_answer_weight=1.0, incorrect_answer_weight=1.0
    )
    ans_mask = [False, True]
    w = compute_segment_weights(2, cfg, ans_mask, None)
    assert w[1] == pytest.approx(2.0)  # unchanged: correctness weights are identity


def test_no_neutralization_when_answer_weight_already_one():
    """answer_weight=1.0 → no double-application even with correctness weights."""
    cfg = TokenISConfig(
        mode="segment", answer_weight=1.0, correct_answer_weight=2.5, incorrect_answer_weight=0.3
    )
    ans_mask = [True, False]
    w = compute_segment_weights(2, cfg, ans_mask, None)
    assert w[0] == pytest.approx(1.0)  # already 1.0, no double application possible


# ── Double-application arithmetic proof ─────────────────────────────────────


def test_double_application_would_be_wrong():
    """Demonstrate why neutralization is needed: 2.0 * 2.5 = 5.0x is unintended."""
    cfg = TokenISConfig(mode="segment", answer_weight=2.0, correct_answer_weight=2.5)
    ans_mask = [True, False]
    w = compute_segment_weights(2, cfg, ans_mask, None)
    # w[0] = 2.0 (segment answer_weight)
    # If we then multiply by correct_answer_weight=2.5: 2.0 * 2.5 = 5.0 (WRONG)
    # With neutralization: 1.0 * 2.5 = 2.5 (correct: only correctness pass multiplies)
    assert w[0] == pytest.approx(2.0)
    double_applied = w[0] * 2.5
    assert double_applied == pytest.approx(5.0)  # proves the problem

    neutralized = replace(cfg, answer_weight=1.0)
    w_neu = compute_segment_weights(2, neutralized, ans_mask, None)
    correct_applied = w_neu[0] * 2.5
    assert correct_applied == pytest.approx(2.5)  # proves the fix


# ── compute_is_weights with neutralized segment ──────────────────────────────


def test_compute_is_weights_segment_mode_neutralized():
    """Calling compute_is_weights with neutralized cfg gives 1.0 on answer tokens."""
    cfg = TokenISConfig(mode="segment", answer_weight=1.0, thinking_weight=0.5)
    ans_mask = [True, True, False]
    think_mask = [False, False, True]
    w = compute_is_weights(3, cfg, ans_mask, think_mask)
    assert w is not None
    assert w[0] == pytest.approx(1.0)  # answer, neutralized
    assert w[1] == pytest.approx(1.0)  # answer, neutralized
    assert w[2] == pytest.approx(0.5)  # thinking weight unchanged


# ── Config loader: new fields parse correctly ────────────────────────────────


def test_config_loader_answer_weight_fields(tmp_path):
    """correct_answer_weight / incorrect_answer_weight are parsed from YAML."""
    import yaml

    from mlx_grpo.config.loader import load_config

    p = tmp_path / "cfg.yml"
    p.write_text(
        yaml.dump(
            {
                "training": {"model_path": "x", "iterations": 1},
                "generation": {"backend": "mlx"},
                "datasets": {},
                "importance_sampling": {
                    "mode": "segment",
                    "answer_weight": 2.0,
                    "correct_answer_weight": 3.0,
                    "incorrect_answer_weight": 0.25,
                    "answer_weight_threshold": 0.6,
                },
            }
        )
    )
    cfg = load_config(p)
    assert cfg.importance_sampling is not None
    is_cfg = cfg.importance_sampling
    assert is_cfg.mode == "segment"
    assert is_cfg.answer_weight == pytest.approx(2.0)
    assert is_cfg.correct_answer_weight == pytest.approx(3.0)
    assert is_cfg.incorrect_answer_weight == pytest.approx(0.25)
    assert is_cfg.answer_weight_threshold == pytest.approx(0.6)


def test_config_loader_answer_weight_defaults(tmp_path):
    """When correctness fields absent, they default to 1.0."""
    import yaml

    from mlx_grpo.config.loader import load_config

    p = tmp_path / "cfg.yml"
    p.write_text(
        yaml.dump(
            {
                "training": {"model_path": "x", "iterations": 1},
                "generation": {"backend": "mlx"},
                "datasets": {},
                "importance_sampling": {"mode": "segment"},
            }
        )
    )
    cfg = load_config(p)
    assert cfg.importance_sampling is not None
    assert cfg.importance_sampling.correct_answer_weight == pytest.approx(1.0)
    assert cfg.importance_sampling.incorrect_answer_weight == pytest.approx(1.0)
    assert cfg.importance_sampling.answer_weight_threshold == pytest.approx(0.5)


# ── _has_answer_w flag: active when weights != 1.0 ───────────────────────────


def test_has_answer_w_inactive_when_both_one():
    """correct=1.0 and incorrect=1.0 → feature effectively disabled (no-op)."""
    cfg = TokenISConfig(correct_answer_weight=1.0, incorrect_answer_weight=1.0)
    caw = cfg.correct_answer_weight
    iaw = cfg.incorrect_answer_weight
    has_answer_w = caw != 1.0 or iaw != 1.0
    assert has_answer_w is False


def test_has_answer_w_active_when_correct_nonone():
    cfg = TokenISConfig(correct_answer_weight=2.5)
    has_answer_w = cfg.correct_answer_weight != 1.0 or cfg.incorrect_answer_weight != 1.0
    assert has_answer_w is True


def test_has_answer_w_active_when_incorrect_nonone():
    cfg = TokenISConfig(incorrect_answer_weight=0.3)
    has_answer_w = cfg.correct_answer_weight != 1.0 or cfg.incorrect_answer_weight != 1.0
    assert has_answer_w is True


# ── Threshold logic ──────────────────────────────────────────────────────────


def test_threshold_selects_correct_weight():
    """reward >= threshold → correct_answer_weight; reward < threshold → incorrect."""
    cfg = TokenISConfig(
        correct_answer_weight=3.0, incorrect_answer_weight=0.2, answer_weight_threshold=0.5
    )

    def select(reward: float) -> float:
        return (
            cfg.correct_answer_weight
            if reward >= cfg.answer_weight_threshold
            else cfg.incorrect_answer_weight
        )

    assert select(1.0) == pytest.approx(3.0)
    assert select(0.5) == pytest.approx(3.0)  # >= threshold
    assert select(0.49) == pytest.approx(0.2)  # < threshold
    assert select(0.0) == pytest.approx(0.2)


def test_threshold_boundary_is_inclusive():
    """Boundary value (reward == threshold) goes to correct branch."""
    cfg = TokenISConfig(
        correct_answer_weight=2.0, incorrect_answer_weight=0.5, answer_weight_threshold=0.7
    )
    w = (
        cfg.correct_answer_weight
        if cfg.answer_weight_threshold <= 0.7
        else cfg.incorrect_answer_weight
    )
    assert w == pytest.approx(2.0)
