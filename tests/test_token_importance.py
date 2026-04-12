"""Tests for token-level importance sampling weight computation."""

from __future__ import annotations

import pytest

from mlx_grpo.config.loader import TokenISConfig
from mlx_grpo.training.token_importance import (
    compute_is_weights,
    compute_length_scale,
    compute_position_weights,
    compute_scaffold_proximity_weights,
    compute_segment_weights,
)

# ── compute_length_scale ─────────────────────────────────────────────


def test_length_scale_at_ref():
    """n == ref_length → scale = 1.0."""
    cfg = TokenISConfig(length_exponent=0.5, ref_length=256)
    assert compute_length_scale(256, cfg) == pytest.approx(1.0)


def test_length_scale_shorter():
    """n < ref_length → scale > 1 (shorter sequences upweighted)."""
    cfg = TokenISConfig(length_exponent=0.5, ref_length=256)
    ls = compute_length_scale(64, cfg)
    assert ls > 1.0  # (256/64)^0.5 = 4^0.5 = 2.0
    assert ls == pytest.approx(2.0)


def test_length_scale_longer():
    """n > ref_length → scale < 1 (longer sequences downweighted)."""
    cfg = TokenISConfig(length_exponent=0.5, ref_length=256)
    ls = compute_length_scale(1024, cfg)
    assert ls < 1.0  # (256/1024)^0.5 = 0.25^0.5 = 0.5
    assert ls == pytest.approx(0.5)


def test_length_scale_zero_exponent():
    """length_exponent=0.0 → always 1.0 (disabled)."""
    cfg = TokenISConfig(length_exponent=0.0, ref_length=256)
    assert compute_length_scale(64, cfg) == 1.0
    assert compute_length_scale(1024, cfg) == 1.0


def test_length_scale_zero_n():
    """n=0 → 1.0 (guard against division by zero)."""
    cfg = TokenISConfig(length_exponent=0.5, ref_length=256)
    assert compute_length_scale(0, cfg) == 1.0


# ── compute_position_weights ─────────────────────────────────────────


def test_position_weights_length():
    """Returns exactly n weights."""
    cfg = TokenISConfig(decay_base=0.5)
    w = compute_position_weights(10, cfg)
    assert len(w) == 10


def test_position_weights_first_is_one():
    """First token always has weight 1.0."""
    cfg = TokenISConfig(decay_base=0.5)
    w = compute_position_weights(8, cfg)
    assert w[0] == pytest.approx(1.0)


def test_position_weights_decreasing():
    """Weights are strictly decreasing with decay_base < 1."""
    cfg = TokenISConfig(decay_base=0.5)
    w = compute_position_weights(8, cfg)
    for i in range(len(w) - 1):
        assert w[i] > w[i + 1]


def test_position_weights_no_decay():
    """decay_base=1.0 → all 1.0 (no decay)."""
    cfg = TokenISConfig(decay_base=1.0)
    w = compute_position_weights(5, cfg)
    assert all(wi == pytest.approx(1.0) for wi in w)


def test_position_weights_empty():
    """n=0 → empty list."""
    cfg = TokenISConfig(decay_base=0.5)
    assert compute_position_weights(0, cfg) == []


def test_position_weights_last_approx_decay_base():
    """Last weight ≈ decay_base (exponent → 1.0 as t → n)."""
    cfg = TokenISConfig(decay_base=0.5)
    # last weight = 0.5^((n-1)/n), approaches 0.5 as n→∞
    w = compute_position_weights(1000, cfg)
    assert w[-1] == pytest.approx(0.5, abs=0.01)


# ── compute_segment_weights ──────────────────────────────────────────


def test_segment_weights_all_default():
    """No masks → all 1.0."""
    cfg = TokenISConfig()
    w = compute_segment_weights(5, cfg, None, None)
    assert w == [1.0] * 5


def test_segment_weights_answer():
    """answer_mask True → answer_weight."""
    cfg = TokenISConfig(answer_weight=3.0)
    ans = [False, False, True, True, False]
    w = compute_segment_weights(5, cfg, ans, None)
    assert w[0] == pytest.approx(1.0)
    assert w[2] == pytest.approx(3.0)
    assert w[3] == pytest.approx(3.0)
    assert w[4] == pytest.approx(1.0)


def test_segment_weights_thinking():
    """thinking_mask True → thinking_weight."""
    cfg = TokenISConfig(thinking_weight=0.25)
    think = [True, True, False, False, False]
    w = compute_segment_weights(5, cfg, None, think)
    assert w[0] == pytest.approx(0.25)
    assert w[1] == pytest.approx(0.25)
    assert w[2] == pytest.approx(1.0)


def test_segment_weights_both_masks():
    """Answer mask takes priority over thinking (applied last)."""
    cfg = TokenISConfig(thinking_weight=0.5, answer_weight=2.0)
    ans = [False, False, True]
    think = [True, False, False]
    w = compute_segment_weights(3, cfg, ans, think)
    assert w[0] == pytest.approx(0.5)  # thinking
    assert w[1] == pytest.approx(1.0)  # neither
    assert w[2] == pytest.approx(2.0)  # answer


# ── compute_is_weights ───────────────────────────────────────────────


def test_is_weights_none_mode():
    """mode='none' → returns None (backward compat, no allocation)."""
    cfg = TokenISConfig(mode="none")
    assert compute_is_weights(100, cfg) is None


def test_is_weights_token_mode_length():
    """mode='token' → list of length n."""
    cfg = TokenISConfig(mode="token")
    w = compute_is_weights(10, cfg)
    assert w is not None
    assert len(w) == 10


def test_is_weights_token_mode_incorporates_length_scale():
    """mode='token', short sequence → all weights > 1 (length scale > 1)."""
    cfg = TokenISConfig(mode="token", length_exponent=0.5, ref_length=256, decay_base=1.0)
    # n=64, ref=256 → ls=(256/64)^0.5=2.0; decay_base=1.0 → all 1.0 → final = 2.0
    w = compute_is_weights(64, cfg)
    assert w is not None
    assert all(wi == pytest.approx(2.0) for wi in w)


def test_is_weights_token_mode_combined():
    """mode='token' with decay: first > last, all scaled by length_scale."""
    cfg = TokenISConfig(mode="token", decay_base=0.5, length_exponent=0.0)
    # length_exponent=0 → ls=1.0; decay_base=0.5 → decreasing
    w = compute_is_weights(8, cfg)
    assert w is not None
    assert w[0] > w[-1]  # positional decay


def test_is_weights_segment_mode():
    """mode='segment' → uses segment weights."""
    cfg = TokenISConfig(mode="segment", thinking_weight=0.5, answer_weight=2.0)
    ans = [True, False]
    think = [False, True]
    w = compute_is_weights(2, cfg, ans, think)
    assert w is not None
    assert w[0] == pytest.approx(2.0)  # answer
    assert w[1] == pytest.approx(0.5)  # thinking


def test_is_weights_zero_length():
    """n=0 → None for any mode."""
    cfg_t = TokenISConfig(mode="token")
    cfg_s = TokenISConfig(mode="segment")
    cfg_p = TokenISConfig(mode="scaffold_proximity")
    assert compute_is_weights(0, cfg_t) is None
    assert compute_is_weights(0, cfg_s) is None
    assert compute_is_weights(0, cfg_p) is None


# ── compute_scaffold_proximity_weights ───────────────────────────────


def test_proximity_no_scaffold_all_ones():
    """No scaffold in completion → all weights 1.0."""
    cfg = TokenISConfig(proximity_decay=0.9)
    scaffold = [False] * 6
    w = compute_scaffold_proximity_weights(6, cfg, scaffold)
    assert w == [1.0] * 6


def test_proximity_decay_1_all_ones():
    """proximity_decay=1.0 → all weights 1.0 (disabled)."""
    cfg = TokenISConfig(proximity_decay=1.0)
    scaffold = [True, False, False, False, False]
    w = compute_scaffold_proximity_weights(5, cfg, scaffold)
    assert w == [1.0] * 5


def test_proximity_prefix_scaffold():
    """Prefix scaffold [T,T,F,F,F] → first generated token gets dist=0 (max weight)."""
    cfg = TokenISConfig(proximity_decay=0.9)
    # scaffold at indices 0,1; generated at 2,3,4.
    # Forward-only: d primed to -1 at scaffold → next token starts at dist=0.
    # dist: [0, 0, 0, 1, 2]
    scaffold = [True, True, False, False, False]
    w = compute_scaffold_proximity_weights(5, cfg, scaffold)
    assert w[0] == pytest.approx(1.0)  # scaffold token (dist=0, masked in loss)
    assert w[1] == pytest.approx(1.0)  # scaffold token (dist=0, masked in loss)
    assert w[2] == pytest.approx(1.0)  # 1st generated: dist=0, weight=decay^0=1.0
    assert w[3] == pytest.approx(0.9)  # 2nd generated: dist=1 → 0.9^1
    assert w[4] == pytest.approx(0.81)  # 3rd generated: dist=2 → 0.9^2


def test_proximity_postfix_scaffold():
    """Postfix scaffold [F,F,F,T,T]: tokens before scaffold get large distance (no look-ahead)."""
    cfg = TokenISConfig(proximity_decay=0.9)
    scaffold = [False, False, False, True, True]
    # Forward-only: no backward pass. Tokens before scaffold have d=n (very far).
    # dist: [5,5,5,0,0]
    w = compute_scaffold_proximity_weights(5, cfg, scaffold)
    assert w[4] == pytest.approx(1.0)  # scaffold (dist=0)
    assert w[3] == pytest.approx(1.0)  # scaffold (dist=0)
    assert w[2] == pytest.approx(0.9**5)  # before scaffold: dist=n=5, no look-ahead
    assert w[1] == pytest.approx(0.9**5)
    assert w[0] == pytest.approx(0.9**5)


def test_proximity_middle_scaffold():
    """Middle scaffold [F,F,T,F,F]: tokens after scaffold decay; tokens before get large dist."""
    cfg = TokenISConfig(proximity_decay=0.9)
    scaffold = [False, False, True, False, False]
    # Forward-only: pos 0,1 have dist=n=5 (no preceding scaffold).
    # Scaffold at pos 2 primes d=-1; pos 3 gets dist=0, pos 4 gets dist=1.
    # dist: [5, 5, 0, 0, 1]
    w = compute_scaffold_proximity_weights(5, cfg, scaffold)
    assert w[2] == pytest.approx(1.0)  # scaffold (dist=0)
    assert w[3] == pytest.approx(1.0)  # 1st generated after scaffold: dist=0
    assert w[4] == pytest.approx(0.9)  # 2nd generated: dist=1
    assert w[0] == pytest.approx(0.9**5)  # before scaffold: dist=n, no look-ahead
    assert w[1] == pytest.approx(0.9**5)


def test_proximity_forward_reset_at_each_injection():
    """Two injections: weight resets to 1.0 at each scaffold start."""
    cfg = TokenISConfig(proximity_decay=0.8)
    # scaffold: positions 0-1 and 5-6; generated: 2-4 and 7-9
    scaffold = [True, True, False, False, False, True, True, False, False, False]
    w = compute_scaffold_proximity_weights(10, cfg, scaffold)
    # After first injection (pos 0-1): pos 2=1.0, pos 3=0.8, pos 4=0.64
    assert w[2] == pytest.approx(1.0)
    assert w[3] == pytest.approx(0.8)
    assert w[4] == pytest.approx(0.64)
    # After second injection (pos 5-6): weight RESETS to 1.0 at pos 7
    assert w[7] == pytest.approx(1.0)
    assert w[8] == pytest.approx(0.8)
    assert w[9] == pytest.approx(0.64)


def test_is_weights_scaffold_proximity_mode():
    """compute_is_weights routes to scaffold proximity correctly."""
    cfg = TokenISConfig(mode="scaffold_proximity", proximity_decay=0.5)
    # scaffold at index 0, generated at 1,2,3
    # Forward-only: pos 1 gets dist=0 → weight=1.0, then decays.
    scaffold = [True, False, False, False]
    w = compute_is_weights(4, cfg, scaffold_mask=scaffold)
    assert w is not None
    assert w[0] == pytest.approx(1.0)  # scaffold, dist=0 (masked in loss)
    assert w[1] == pytest.approx(1.0)  # 1st generated: dist=0 → 0.5^0=1.0
    assert w[2] == pytest.approx(0.5)  # dist=1 → 0.5^1
    assert w[3] == pytest.approx(0.25)  # dist=2 → 0.5^2


# ── TokenISConfig defaults ───────────────────────────────────────────


def test_tokenisconfig_defaults():
    cfg = TokenISConfig()
    assert cfg.mode == "none"
    assert cfg.decay_base == 0.5
    assert cfg.length_exponent == 0.5
    assert cfg.ref_length == 256
    assert cfg.thinking_weight == 0.5
    assert cfg.answer_weight == 2.0


# ── Config loader integration ────────────────────────────────────────


def test_config_importance_sampling_absent(tmp_path):
    """Config without importance_sampling: → importance_sampling is None."""
    import yaml

    from mlx_grpo.config.loader import load_config

    p = tmp_path / "cfg.yml"
    p.write_text(
        yaml.dump(
            {
                "training": {"model_path": "x", "iterations": 1},
                "generation": {"backend": "mlx"},
                "datasets": {},
            }
        )
    )
    cfg = load_config(p)
    assert cfg.importance_sampling is None


def test_proximity_member_split_defaults():
    """proximity_member_split=1.0 and proximity_decay_far=None are defaults."""
    cfg = TokenISConfig()
    assert cfg.proximity_member_split == 1.0
    assert cfg.proximity_decay_far is None


def test_proximity_near_vs_far_weight_shape():
    """Near-proximity weights decrease with distance; far-proximity weights increase."""
    scaffold = [True, False, False, False, False]
    cfg_near = TokenISConfig(proximity_decay=0.9, invert_proximity=False)
    cfg_far = TokenISConfig(proximity_decay=0.3, invert_proximity=True)
    w_near = compute_scaffold_proximity_weights(5, cfg_near, scaffold)
    w_far = compute_scaffold_proximity_weights(5, cfg_far, scaffold)
    # Near: first generated token (idx 2) has max weight, decays further out
    assert w_near[2] > w_near[3] > w_near[4]
    # Far: first generated token (idx 2) has min weight, increases further out
    assert w_far[2] < w_far[3] < w_far[4]


def test_proximity_decay_far_override():
    """proximity_decay_far is applied via _dc_replace before compute_scaffold_proximity_weights.

    Training dispatch: replace(cfg, invert_proximity=True, proximity_decay=proximity_decay_far)
    This test verifies the effective far-cfg produces the expected floor weight.
    """
    from dataclasses import replace

    scaffold = [True, False, False, False, False]
    base_cfg = TokenISConfig(proximity_decay=0.9, proximity_decay_far=0.2)
    # Simulate what training/__init__.py does for a far member
    far_cfg = replace(base_cfg, invert_proximity=True, proximity_decay=base_cfg.proximity_decay_far)
    w = compute_scaffold_proximity_weights(5, far_cfg, scaffold)
    # invert_proximity=True, proximity_decay=0.2: floor at scaffold-adjacent = 0.2
    # dist[1]=0 (scaffold-adjacent): w = 0.2 + (1-0.2)*(0/max_d) = 0.2
    assert w[1] == pytest.approx(0.2)


def test_proximity_member_split_round_trip():
    """proximity_member_split field round-trips through TokenISConfig."""
    cfg = TokenISConfig(proximity_member_split=0.6, proximity_decay_far=0.4)
    assert cfg.proximity_member_split == pytest.approx(0.6)
    assert cfg.proximity_decay_far == pytest.approx(0.4)


def test_proximity_member_split_use_far_logic():
    """Simulate the _use_far dispatch: near members get near-proximity, far get inverted."""
    from dataclasses import replace

    scaffold = [True, False, False, False, False]
    n = 5
    # Base cfg: near-proximity (invert_proximity=False)
    base_cfg = TokenISConfig(
        mode="scaffold_proximity", proximity_decay=0.8, proximity_member_split=0.5
    )
    # Simulate far override (what training/__init__.py applies for _use_far members)
    far_cfg = replace(base_cfg, invert_proximity=True)

    w_near = compute_scaffold_proximity_weights(n, base_cfg, scaffold)
    w_far = compute_scaffold_proximity_weights(n, far_cfg, scaffold)

    # Near member: weights decrease (near scaffold = higher gradient)
    assert w_near[2] > w_near[4]
    # Far member: weights increase (far from scaffold = higher gradient)
    assert w_far[2] < w_far[4]
    # Both produce n weights
    assert len(w_near) == n
    assert len(w_far) == n


def test_proximity_chunk_split_default():
    """proximity_chunk_split=1.0 is default (all chunks near, unchanged behavior)."""
    cfg = TokenISConfig()
    assert cfg.proximity_chunk_split == 1.0


def test_proximity_chunk_split_near_only():
    """chunk_split=1.0 → all chunks near-proximity; same as original behavior."""
    # scaffold at 0-1, gen at 2-4, scaffold at 5-6, gen at 7-9
    scaffold = [True, True, False, False, False, True, True, False, False, False]
    cfg_split = TokenISConfig(proximity_decay=0.8, proximity_chunk_split=1.0)
    cfg_orig = TokenISConfig(proximity_decay=0.8, invert_proximity=False)
    w_split = compute_scaffold_proximity_weights(10, cfg_split, scaffold)
    w_orig = compute_scaffold_proximity_weights(10, cfg_orig, scaffold)
    assert w_split == pytest.approx(w_orig)


def test_proximity_chunk_split_two_chunks_half():
    """With 2 chunks and split=0.5: chunk 0 near (decreasing), chunk 1 far (increasing)."""
    # chunk 0: scaffold at 0-1; gen at 2-4 (dist=0,1,2)
    # chunk 1: scaffold at 5-6; gen at 7-9 (dist=0,1,2)
    scaffold = [True, True, False, False, False, True, True, False, False, False]
    cfg = TokenISConfig(proximity_decay=0.8, proximity_chunk_split=0.5)
    w = compute_scaffold_proximity_weights(10, cfg, scaffold)
    # Chunk 0 (near): weights decrease with distance
    assert w[2] > w[3] > w[4]  # dist=0,1,2 → 1.0, 0.8, 0.64
    # Chunk 1 (far): weights increase with distance
    assert w[7] < w[8] < w[9]  # dist=0→min, dist=2→max


def test_proximity_chunk_split_all_far():
    """chunk_split=0.0 → all chunks far-proximity (increasing weights)."""
    scaffold = [True, True, False, False, False]
    cfg = TokenISConfig(proximity_decay=0.5, proximity_chunk_split=0.0)
    w = compute_scaffold_proximity_weights(5, cfg, scaffold)
    # Far: weights increase with distance. dist at idx2=0, idx3=1, idx4=2
    assert w[2] < w[3] < w[4]


def test_proximity_chunk_split_three_chunks():
    """With 3 chunks and split=0.67: first 2 near, last 1 far."""
    # Three chunks of [T, F, F] each: indices 0,3,6 are scaffold; 1-2,4-5,7-8 generated
    scaffold = [True, False, False, True, False, False, True, False, False]
    cfg = TokenISConfig(proximity_decay=0.9, proximity_chunk_split=0.67)
    w = compute_scaffold_proximity_weights(9, cfg, scaffold)
    # Chunk 0 and 1 (near): w[1] > w[2], w[4] > w[5]
    assert w[1] > w[2]
    assert w[4] > w[5]
    # Chunk 2 (far): w[7] < w[8]
    assert w[7] < w[8]


def test_proximity_naked_member_unaffected_by_split():
    """Naked member (no scaffold) produces flat weights regardless of invert_proximity."""
    naked_mask = [False] * 6
    cfg_near = TokenISConfig(proximity_decay=0.8, invert_proximity=False)
    cfg_far = TokenISConfig(proximity_decay=0.8, invert_proximity=True)
    w_near = compute_scaffold_proximity_weights(6, cfg_near, naked_mask)
    w_far = compute_scaffold_proximity_weights(6, cfg_far, naked_mask)
    # Both return flat 1.0 (no scaffold → naked path, bypasses near/far logic)
    assert w_near == [1.0] * 6
    assert w_far == [1.0] * 6


def test_config_importance_sampling_present(tmp_path):
    """Config with importance_sampling: block parses correctly."""
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
                    "mode": "token",
                    "decay_base": 0.7,
                    "length_exponent": 0.3,
                    "ref_length": 512,
                },
            }
        )
    )
    cfg = load_config(p)
    assert cfg.importance_sampling is not None
    is_cfg = cfg.importance_sampling
    assert is_cfg.mode == "token"
    assert is_cfg.decay_base == pytest.approx(0.7)
    assert is_cfg.length_exponent == pytest.approx(0.3)
    assert is_cfg.ref_length == 512
