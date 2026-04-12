"""Tests for the ICL-Guided Dynamic Escalation Engine (v5 + v6.1).

96 tests covering:
- _count_n_eff_max: boundary counting, empty/single/multi-sentence traces
- _plan_adaptive_scaffold: boundary-snapped planning, token fallback, N_eff, B_eff
- _two_phase_generate: Phase A clean stop, Phase B triggered, overflow
- _build_tri_factor_weights: naked early exit, answer tokens, tri-factor decay
- _apply_group_shaping: group-level shaping, annealing, lambda values
- _vertical_rollout: interleaved scaffold+generation cycles, answer capture
- run_escalation: horizontal loop, inline reward, early-stop, metadata schema
- collect_escalation_stats: stat aggregation with new schema
- EscalationConfig / build_escalation_config: YAML parsing incl. new sub-configs
- IS weights integration: _esc_weights detection, shaped reward for advantage
- v6.1: adaptive cap (n_eff_max+1), allocation spans n_eff_max, c_max=n_eff_max
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock

import pytest

from mlx_grpo.backends.base import GenerateOutput, GenerateParams
from mlx_grpo.config.escalation import (
    EscalationConfig,
    EscalationScaffoldConfig,
    EscalationTrackingConfig,
    HorizontalConfig,
    RewardShapingConfig,
    TriFactorConfig,
    VerticalConfig,
    build_escalation_config,
)
from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.engine.context import BatchContext, RolloutContext
from mlx_grpo.engine.executor import RolloutExecutor
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training.escalation import (
    _vertical_rollout,
    collect_escalation_stats,
    run_escalation,
)
from mlx_grpo.training.escalation_plan import (
    _answer_prefix_text,
    _apply_group_shaping,
    _build_tri_factor_weights,
    _count_n_eff_max,
    _jaccard_anchor_from_naked,
    _plan_adaptive_scaffold,
    _strip_answer_close,
    _two_phase_generate,
    merge_esc_cfg,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────


class _ControlledBackend:
    """Backend where each generate() call returns a pre-determined response."""

    def __init__(self, responses: list[str] | None = None, stop_at: str | None = "</think>"):
        self._responses = responses or []
        self._call_idx = 0
        self._stop_at = stop_at

    def generate(self, prefix_tokens: list[int], params: GenerateParams) -> GenerateOutput:
        text = (
            self._responses[self._call_idx]
            if self._call_idx < len(self._responses)
            else f"gen_{self._call_idx}"
        )
        self._call_idx += 1
        stop_reason, stop_sequence = "eos", None
        if self._stop_at and self._stop_at in params.stop and self._stop_at in text:
            idx = text.index(self._stop_at)
            text = text[:idx]
            stop_reason, stop_sequence = "stop_sequence", self._stop_at
        tokens = self.encode(text)
        if len(tokens) > params.max_tokens:
            tokens = tokens[: params.max_tokens]
            text = self.decode(tokens)
            stop_reason = "max_tokens"
        return GenerateOutput(
            text=text, tokens=tokens, stop_reason=stop_reason, stop_sequence=stop_sequence
        )

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _make_ctx(backend, segments=None, vert_cfg=None, type_cfg=None, n_eff=None, b_eff=0, b_ext=0):
    """Build a minimal RolloutContext for _vertical_rollout tests."""
    segs = segments or []
    prompt_tokens = backend.encode("prompt")
    batch = BatchContext(
        iteration=0,
        total_iters=10,
        group_index=0,
        group_size=2,
        custom={
            "segments": segs,
            "vert_cfg": vert_cfg,
            "N_eff": n_eff if n_eff is not None else len(segs),
            "B_eff": b_eff,
            "b_ext": b_ext,
        },
    )
    return RolloutContext(
        backend=backend,
        prompt="prompt",
        prompt_tokens=prompt_tokens,
        answer="ans",
        type_name="test",
        type_config=type_cfg or {"thinking_budget": 64, "answer_max_tokens": 8},
        batch=batch,
    )


def _make_trainer(backend, config=None):
    """Build a minimal Trainer-like object for run_escalation tests."""
    _ans = "<think>step1.\nstep2.\n</think>A"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "q", "answer": _ans, "type": "mcq"}) + "\n")

    cfg = config or TrainingConfig(iterations=10, group_size=2, save_every=0, eval_every=0)
    trainer = MagicMock()
    trainer.backend = backend
    trainer.config = cfg
    trainer.executor = RolloutExecutor(backend)
    trainer._build_callbacks = MagicMock(return_value=None)
    return trainer


def _make_sample(answer="<think>step one.\nstep two.\n</think>\\boxed{42}"):
    from mlx_grpo.data import Sample

    return Sample(
        prompt="what is 6×7?",
        answer=answer,
        type_name="math",
        metadata={"ground_truth": "42"},
    )


def _make_reward_fn(fixed_reward=0.0):
    rf = MagicMock()
    rf.compute = MagicMock(return_value=fixed_reward)
    return rf


def _make_result_with_segments(
    injected_toks: list[int] | None = None,
    generated_toks: list[int] | None = None,
    answer_toks: list[int] | None = None,
) -> RolloutResult:
    """Build a RolloutResult with explicit injected/generated/answer segments."""
    r = RolloutResult(type_name="test")
    if injected_toks:
        r.segments.append(
            Segment(
                tokens=injected_toks,
                text="scaffold",
                source=SegmentSource.INJECTED,
                masked=True,
                tag="scaffold_0",
            )
        )
    if generated_toks:
        r.segments.append(
            Segment(
                tokens=generated_toks,
                text="generated",
                source=SegmentSource.GENERATED,
                masked=False,
                tag="cycle_0",
            )
        )
    if answer_toks:
        r.segments.append(
            Segment(
                tokens=answer_toks,
                text="answer",
                source=SegmentSource.GENERATED,
                masked=False,
                tag="answer",
            )
        )
    return r


# ── TestPlanAdaptiveScaffold ───────────────────────────────────────────────────


class TestPlanAdaptiveScaffold:
    def test_naked_returns_empty(self):
        """c_k=0 → ([], 0, 0) immediately."""
        backend = _ControlledBackend()
        segs, n_eff, b_eff = _plan_adaptive_scaffold("<think>abc</think>", 0, backend, 512)
        assert segs == []
        assert n_eff == 0
        assert b_eff == 0

    def test_basic_splits_on_boundaries(self):
        """Trace with sentence boundaries → segments split at boundaries."""
        backend = _ControlledBackend()
        trace = "<think>First sentence.\nSecond sentence.\nThird sentence.\n</think>"
        segs, n_eff, b_eff = _plan_adaptive_scaffold(trace, 3, backend, 512)
        assert len(segs) >= 1
        assert n_eff >= 1
        assert b_eff >= 1

    def test_strips_think_tags(self):
        """<think> tag is stripped; </think> and answer not included."""
        backend = _ControlledBackend()
        segs, _, _ = _plan_adaptive_scaffold("<think>hello.\n</think>\\boxed{X}", 1, backend, 512)
        joined = "".join(segs)
        assert "<think>" not in joined
        assert "boxed" not in joined
        assert "X" not in joined

    def test_stops_at_close_think(self):
        """Content after </think> is excluded."""
        backend = _ControlledBackend()
        segs, _, _ = _plan_adaptive_scaffold(
            "<think>thinking.\n</think>\\boxed{answer}", 2, backend, 512
        )
        joined = "".join(segs)
        assert "answer" not in joined
        assert "boxed" not in joined

    def test_b_eff_floor(self):
        """B_eff is always >= 32."""
        backend = _ControlledBackend()
        _, _, b_eff = _plan_adaptive_scaffold("<think>a.\n</think>", 1, backend, 512)
        assert b_eff >= 32

    def test_empty_text_returns_empty(self):
        """Empty trace → ([], 0, 0)."""
        backend = _ControlledBackend()
        segs, n_eff, b_eff = _plan_adaptive_scaffold("", 4, backend, 512)
        assert segs == []

    def test_short_trace_fewer_segments(self):
        """Short trace with few boundaries → N_eff <= C_k."""
        backend = _ControlledBackend()
        trace = "<think>one.\n</think>"
        segs, n_eff, _ = _plan_adaptive_scaffold(trace, 10, backend, 512)
        assert n_eff <= 10
        assert len(segs) <= 10

    def test_max_tokens_cap(self):
        """Each segment is capped at max_tokens."""
        backend = _ControlledBackend()
        trace = "<think>" + "x" * 200 + ".\n" + "y" * 200 + ".\n</think>"
        segs, _, _ = _plan_adaptive_scaffold(trace, 2, backend, max_tokens=5)
        for seg in segs:
            assert len(backend.encode(seg)) <= 5


# ── TestTwoPhasGenerate ───────────────────────────────────────────────────────


class TestTwoPhasGenerate:
    def test_phase_a_stops_cleanly(self):
        """Phase A hits </think> → returns immediately, no Phase B."""
        backend = _ControlledBackend(["thinking</think>"])
        ctx = _make_ctx(backend, segments=["s"])
        result, closed = _two_phase_generate(ctx, b_eff=64, b_ext=32, temp=0.7, cycle_idx=0)
        assert closed is True
        # Only one generate call (Phase A), no Phase B
        assert backend._call_idx == 1

    def test_phase_b_triggered_when_phase_a_hits_max(self):
        """Phase A hits max_tokens → Phase B is called."""
        backend = _ControlledBackend(["aaaa", "ends.\n"])
        ctx = _make_ctx(backend, segments=["s"])
        # b_eff=2: only 2 tokens for Phase A → max_tokens hit, no </think>
        _two_phase_generate(ctx, b_eff=2, b_ext=8, temp=0.7, cycle_idx=0)
        assert backend._call_idx == 2  # Phase A + Phase B

    def test_overflow_accepted_when_no_close(self):
        """Both Phase A and Phase B hit max_tokens → closed_think=False."""
        backend = _ControlledBackend(["thinking", "more thinking"])
        ctx = _make_ctx(backend, segments=["s"])
        _, closed = _two_phase_generate(ctx, b_eff=4, b_ext=4, temp=0.7, cycle_idx=0)
        assert closed is False

    def test_no_phase_b_when_b_ext_zero(self):
        """b_ext=0 → Phase B disabled even if A hits max_tokens."""
        backend = _ControlledBackend(["thinking"])
        ctx = _make_ctx(backend, segments=["s"])
        _two_phase_generate(ctx, b_eff=4, b_ext=0, temp=0.7, cycle_idx=0)
        assert backend._call_idx == 1


# ── TestBuildTriFactorWeights ─────────────────────────────────────────────────


class TestBuildTriFactorWeights:
    def _make_tf(self, alpha_h=0.7, alpha_v=0.8, alpha_p=0.9, b_ext=64):
        return TriFactorConfig(alpha_h=alpha_h, alpha_v=alpha_v, alpha_p=alpha_p, b_ext=b_ext)

    def test_naked_early_exit_all_ones(self):
        """c_k=0 → all weights 1.0, same length as loss_mask."""
        r = _make_result_with_segments(injected_toks=[1, 2], generated_toks=[3, 4, 5])
        tf = self._make_tf()
        w = _build_tri_factor_weights(r, c_k=0, c_max=4, n_eff=2, tf_cfg=tf)
        assert w == [1.0] * len(r.loss_mask)

    def test_answer_tokens_weight_one(self):
        """Tokens in 'answer'-tagged segment always get weight 1.0."""
        r = _make_result_with_segments(injected_toks=[1], generated_toks=[2, 3], answer_toks=[4, 5])
        tf = self._make_tf()
        w = _build_tri_factor_weights(r, c_k=2, c_max=4, n_eff=1, tf_cfg=tf)
        # Last 2 weights correspond to the answer segment
        assert w[-2] == pytest.approx(1.0)
        assert w[-1] == pytest.approx(1.0)

    def test_weight_array_length_equals_mask(self):
        """len(weights) == len(result.loss_mask) always (C1–C4 regression)."""
        r = _make_result_with_segments(injected_toks=[1, 2, 3], generated_toks=[4, 5])
        tf = self._make_tf()
        w = _build_tri_factor_weights(r, c_k=1, c_max=2, n_eff=1, tf_cfg=tf)
        assert len(w) == len(r.loss_mask)

    def test_weight_array_injected_positions_present(self):
        """Injected segment tokens have weight entries (value 1.0); total = injected+generated."""
        inj = [1, 2, 3]
        gen = [4, 5]
        r = _make_result_with_segments(injected_toks=inj, generated_toks=gen)
        tf = self._make_tf()
        w = _build_tri_factor_weights(r, c_k=1, c_max=2, n_eff=1, tf_cfg=tf)
        # 3 injected + 2 generated = 5 total weights
        assert len(w) == 5
        # Injected positions get 1.0
        assert w[:3] == [1.0, 1.0, 1.0]

    def test_tri_factor_horizontal_decay(self):
        """Higher C_k/C_max → lower base weight for generated tokens."""
        r1 = _make_result_with_segments(injected_toks=[1], generated_toks=[2])
        r2 = _make_result_with_segments(injected_toks=[1], generated_toks=[2])
        tf = self._make_tf(alpha_h=0.7)
        w_low = _build_tri_factor_weights(r1, c_k=1, c_max=4, n_eff=1, tf_cfg=tf)
        w_high = _build_tri_factor_weights(r2, c_k=4, c_max=4, n_eff=1, tf_cfg=tf)
        # More scaffold → lower weight
        assert w_high[-1] < w_low[-1]

    def test_tri_factor_proximity_decay(self):
        """Tokens further from scaffold have lower weight than closer tokens."""
        gen_toks = [10, 11, 12, 13, 14]
        r = _make_result_with_segments(injected_toks=[1], generated_toks=gen_toks)
        tf = self._make_tf(alpha_p=0.9)
        w = _build_tri_factor_weights(r, c_k=1, c_max=2, n_eff=1, tf_cfg=tf)
        gen_weights = w[1:]  # skip injected weight
        # Weights should decrease as causal distance increases
        assert gen_weights[0] > gen_weights[-1]

    def test_cycle_idx_no_double_increment(self):
        """Issue A fix: Phase A (cycle_0) and Phase B (cycle_0_ext) use the SAME cycle_idx."""
        r = RolloutResult(type_name="test")
        # scaffold_0 injection → cycle_idx becomes 0
        r.segments.append(
            Segment(
                tokens=[1], text="s", source=SegmentSource.INJECTED, masked=True, tag="scaffold_0"
            )
        )
        # Phase A of cycle 0
        r.segments.append(
            Segment(
                tokens=[2, 3],
                text="pa",
                source=SegmentSource.GENERATED,
                masked=False,
                tag="cycle_0",
            )
        )
        # Phase B of cycle 0 (same cycle, just extension)
        r.segments.append(
            Segment(
                tokens=[4, 5],
                text="pb",
                source=SegmentSource.GENERATED,
                masked=False,
                tag="cycle_0_ext",
            )
        )
        # scaffold_1 injection → cycle_idx becomes 1
        r.segments.append(
            Segment(
                tokens=[6], text="s2", source=SegmentSource.INJECTED, masked=True, tag="scaffold_1"
            )
        )
        # Phase A of cycle 1
        r.segments.append(
            Segment(
                tokens=[7, 8],
                text="pa2",
                source=SegmentSource.GENERATED,
                masked=False,
                tag="cycle_1",
            )
        )

        tf = TriFactorConfig(alpha_h=0.7, alpha_v=0.8, alpha_p=1.0, b_ext=64)
        w = _build_tri_factor_weights(r, c_k=2, c_max=4, n_eff=2, tf_cfg=tf)

        # Index mapping: inj[0]=w[0], pa[0]=w[1], pa[1]=w[2], pb[0]=w[3], pb[1]=w[4],
        #                inj[1]=w[5], pa2[0]=w[6], pa2[1]=w[7]
        # cycle_idx=0 for pa and pb (same cycle), cycle_idx=1 for pa2
        # alpha_p=1.0 so proximity factor is 1.0 everywhere

        alpha_h = 0.7 ** (2 / 4)
        # Both Phase A and Phase B of cycle 0 use cycle_idx=0
        v_cycle0 = 0.8 ** (0 / max(2 - 1, 1))
        v_cycle1 = 0.8 ** (1 / max(2 - 1, 1))

        assert w[1] == pytest.approx(alpha_h * v_cycle0)  # pa[0]
        assert w[3] == pytest.approx(alpha_h * v_cycle0)  # pb[0] — same cycle as pa
        assert w[6] == pytest.approx(alpha_h * v_cycle1)  # pa2[0] — next cycle

    def test_tri_factor_vertical_decay(self):
        """Later cycles get lower vertical factor."""
        r = RolloutResult(type_name="test")
        # Cycle 0
        r.segments.append(
            Segment(
                tokens=[1], text="s0", source=SegmentSource.INJECTED, masked=True, tag="scaffold_0"
            )
        )
        r.segments.append(
            Segment(
                tokens=[2], text="c0", source=SegmentSource.GENERATED, masked=False, tag="cycle_0"
            )
        )
        # Cycle 1
        r.segments.append(
            Segment(
                tokens=[3], text="s1", source=SegmentSource.INJECTED, masked=True, tag="scaffold_1"
            )
        )
        r.segments.append(
            Segment(
                tokens=[4], text="c1", source=SegmentSource.GENERATED, masked=False, tag="cycle_1"
            )
        )
        tf = TriFactorConfig(alpha_h=1.0, alpha_v=0.8, alpha_p=1.0, b_ext=64)
        w = _build_tri_factor_weights(r, c_k=2, c_max=4, n_eff=2, tf_cfg=tf)
        # w[0]=1.0 (inj), w[1]=cycle0_weight, w[2]=1.0 (inj), w[3]=cycle1_weight
        assert w[3] < w[1]  # cycle 1 gets lower weight than cycle 0


# ── TestApplyGroupShaping ─────────────────────────────────────────────────────


class TestApplyGroupShaping:
    def _make_rs(self, lambda_s=0.05, lambda_f=0.02, failure_floor=0.0):
        return RewardShapingConfig(
            lambda_s_max=lambda_s, lambda_f_max=lambda_f, failure_floor=failure_floor
        )

    def _make_rollout(self, reward, allocated_chunks, is_naked=False):
        r = RolloutResult(type_name="test")
        r.metadata["_esc_reward"] = reward
        r.metadata["_esc_allocated_chunks"] = allocated_chunks
        r.metadata["_view_is_naked"] = is_naked
        return r

    def test_shaping_config_defaults(self):
        """Default lambda values are small (tiebreakers, not primary signals)."""
        rs = RewardShapingConfig()
        assert rs.lambda_s_max <= 0.10
        assert rs.lambda_f_max <= 0.05

    def test_lambda_defaults_small(self):
        """lambda_s_max ≤ 0.10, lambda_f_max ≤ 0.05 (C1 regression)."""
        rs = RewardShapingConfig()
        assert rs.lambda_s_max <= 0.10
        assert rs.lambda_f_max <= 0.05

    def test_annealing_lambda_s_increases(self):
        """progress=0 → lambda_s=0; progress=1 → lambda_s=lambda_s_max (C2 regression)."""
        rs = self._make_rs(lambda_s=0.05)
        r = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)

        # progress=0 → lambda_s=0 → no shaping on success
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=0, total_iters=10)
        shaped_at_0 = r.metadata["_esc_shaped_reward"]
        assert shaped_at_0 == pytest.approx(1.0)  # lambda_s=0 → no discount

        r2 = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked2 = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        # progress=1 → lambda_s=max → max discount
        _apply_group_shaping([r2, naked2], rs, c_max=4, iteration=10, total_iters=10)
        shaped_at_1 = r2.metadata["_esc_shaped_reward"]
        assert shaped_at_1 < 1.0  # discount applied

    def test_annealing_lambda_f_decreases(self):
        """progress=0 → lambda_f=max; progress=1 → lambda_f=0 (C2 regression)."""
        rs = self._make_rs(lambda_f=0.02, failure_floor=1.0)  # opt-in to nudge
        r = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)

        # All wrong group. progress=0 → lambda_f=max → nudge
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=0, total_iters=10)
        nudge_at_0 = r.metadata["_esc_shaped_reward"]
        assert nudge_at_0 > 0.0

        r2 = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked2 = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        # progress=1 → lambda_f=0 → no nudge
        _apply_group_shaping([r2, naked2], rs, c_max=4, iteration=10, total_iters=10)
        nudge_at_1 = r2.metadata["_esc_shaped_reward"]
        assert nudge_at_1 == pytest.approx(0.0)

    def test_annealing_opposite_directions(self):
        """At progress=0.5: lambda_s and lambda_f are at half-max, moving opposite (C2)."""
        rs = self._make_rs(lambda_s=0.05, lambda_f=0.02, failure_floor=1.0)  # opt-in to nudge
        # Test via the actual shaping: success path at progress=0.5
        r_fail = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)

        # All-wrong group to test failure path
        _apply_group_shaping([r_fail, naked], rs, c_max=4, iteration=5, total_iters=10)
        nudge_half = r_fail.metadata["_esc_shaped_reward"]
        # lambda_f at mid-point = 0.5 * lambda_f_max
        expected_nudge = 0.02 * 0.5 * (2 / 4)
        assert nudge_half == pytest.approx(expected_nudge)

    def test_shaping_is_group_level(self):
        """Wrong member in correct group gets success path (C3 regression)."""
        rs = self._make_rs(lambda_s=0.05)
        r_correct = self._make_rollout(reward=1.0, allocated_chunks=1)
        r_wrong = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)

        _apply_group_shaping([r_correct, r_wrong, naked], rs, c_max=4, iteration=5, total_iters=10)
        # any_correct=True → success path for all
        # r_wrong: raw=0 → shaped = 0 * (1 - lambda_s * 2/4) = 0.0
        assert r_wrong.metadata["_esc_shaped_reward"] == pytest.approx(0.0)
        # NOT the failure path (which would give lambda_f * 2/4 > 0)

    def test_shaping_all_wrong_group(self):
        """All members wrong → failure path nudge when failure_floor allows (C3)."""
        rs = self._make_rs(lambda_f=0.02, failure_floor=1.0)  # opt-in to nudge
        r1 = self._make_rollout(reward=0.0, allocated_chunks=1)
        r2 = self._make_rollout(reward=0.0, allocated_chunks=3)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([r1, r2, naked], rs, c_max=4, iteration=0, total_iters=10)
        # lambda_f_max * 1.0 (progress=0) * C_k/C_max
        assert r1.metadata["_esc_shaped_reward"] == pytest.approx(0.02 * 1 / 4)
        assert r2.metadata["_esc_shaped_reward"] == pytest.approx(0.02 * 3 / 4)
        # More scaffold → larger nudge
        assert r2.metadata["_esc_shaped_reward"] > r1.metadata["_esc_shaped_reward"]

    def test_shaping_per_member_would_fail(self):
        """Wrong member in correct group must NOT get lambda_f nudge (C3 regression guard)."""
        rs = self._make_rs(lambda_s=0.05, lambda_f=0.02)
        r_correct = self._make_rollout(reward=1.0, allocated_chunks=1)
        r_wrong = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([r_correct, r_wrong, naked], rs, c_max=4, iteration=5, total_iters=10)
        wrong_shaped = r_wrong.metadata["_esc_shaped_reward"]
        # Per-member failure path would give: 0.02 * 0.5 * 2/4 = 0.005
        failure_path_value = 0.02 * 0.5 * (2 / 4)
        # Actual should be 0.0 (success path, raw=0)
        assert wrong_shaped != pytest.approx(failure_path_value)
        assert wrong_shaped == pytest.approx(0.0)

    def test_correct_scaffolded_beats_wrong_naked(self):
        """shaped(correct, k=max) >> shaped(wrong, k=0) — no reward inversion (C1)."""
        rs = self._make_rs(lambda_s=0.05)
        r_correct_max = self._make_rollout(reward=1.0, allocated_chunks=4)
        r_naked_wrong = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping(
            [r_correct_max, r_naked_wrong], rs, c_max=4, iteration=10, total_iters=10
        )
        # Correct but heavily scaffolded: 1.0 * (1 - 0.05 * 1.0) = 0.95
        assert r_correct_max.metadata["_esc_shaped_reward"] == pytest.approx(0.95)
        # Naked wrong: raw=0, _view_is_naked → shaped=0
        assert r_naked_wrong.metadata["_esc_shaped_reward"] == pytest.approx(0.0)

    def test_reward_shaping_success_path(self):
        """Success path formula: raw * (1 - lambda_s * C_k / C_max)."""
        rs = self._make_rs(lambda_s=0.05)
        r = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked = self._make_rollout(reward=1.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=10, total_iters=10)
        # lambda_s = 0.05 * 1.0 (progress=1), C_k/C_max = 2/4 = 0.5
        expected = 1.0 * (1.0 - 0.05 * 0.5)
        assert r.metadata["_esc_shaped_reward"] == pytest.approx(expected)

    def test_reward_shaping_failure_path(self):
        """Failure path formula: lambda_f * C_k / C_max (with failure_floor opt-in)."""
        rs = self._make_rs(lambda_f=0.02, failure_floor=1.0)  # opt-in to nudge
        r = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=0, total_iters=10)
        # lambda_f = 0.02 * 1.0 (progress=0), C_k/C_max = 2/4 = 0.5
        expected = 0.02 * 0.5
        assert r.metadata["_esc_shaped_reward"] == pytest.approx(expected)

    def test_reward_shaping_fully_annealed(self):
        """progress=1.0 on failure path → lambda_f=0 → shaped=0."""
        rs = self._make_rs(lambda_f=0.02)
        r = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=10, total_iters=10)
        assert r.metadata["_esc_shaped_reward"] == pytest.approx(0.0)

    def test_reward_shaping_uses_allocated_not_n_eff(self):
        """Shaping uses _esc_allocated_chunks (C_k), not N_eff."""
        rs = self._make_rs(lambda_s=0.05)
        r = self._make_rollout(reward=1.0, allocated_chunks=4)
        naked = self._make_rollout(reward=1.0, allocated_chunks=0, is_naked=True)
        # Even though N_eff might be 2, shaping uses _esc_allocated_chunks=4
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=10, total_iters=10)
        # c_k=4, c_max=4 → full fraction
        expected = 1.0 * (1.0 - 0.05 * 1.0)
        assert r.metadata["_esc_shaped_reward"] == pytest.approx(expected)

    def test_annealing_progress_bounds(self):
        """progress is clamped to [0, 1]."""
        rs = self._make_rs(lambda_s=0.05)
        r = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked = self._make_rollout(reward=1.0, allocated_chunks=0, is_naked=True)
        # iteration > total_iters → progress clamped to 1.0
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=999, total_iters=10)
        expected = 1.0 * (1.0 - 0.05 * 0.5)
        assert r.metadata["_esc_shaped_reward"] == pytest.approx(expected)

    def test_naked_view_no_shaping(self):
        """Naked view (_view_is_naked=True) → shaped == raw."""
        rs = self._make_rs(lambda_f=0.02)
        r_wrong = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.7, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([r_wrong, naked], rs, c_max=4, iteration=0, total_iters=10)
        assert naked.metadata["_esc_shaped_reward"] == pytest.approx(0.7)

    def test_prefer_scaffolded_config_defaults(self):
        """prefer_scaffolded and prefer_scaffolded_bonus have sensible defaults."""
        rs = RewardShapingConfig()
        assert rs.prefer_scaffolded is False
        assert rs.prefer_scaffolded_bonus == pytest.approx(0.1)

    def test_prefer_scaffolded_parsed_from_yaml(self):
        cfg = build_escalation_config(
            {"reward_shaping": {"prefer_scaffolded": True, "prefer_scaffolded_bonus": 0.2}}
        )
        assert cfg.reward_shaping.prefer_scaffolded is True
        assert cfg.reward_shaping.prefer_scaffolded_bonus == pytest.approx(0.2)

    def test_prefer_scaffolded_adds_bonus_when_both_pass(self):
        """Both naked and scaffolded reward=1.0: scaffolded shaped += bonus."""
        rs = RewardShapingConfig(prefer_scaffolded=True, prefer_scaffolded_bonus=0.1)
        scaffolded = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked = self._make_rollout(reward=1.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([scaffolded, naked], rs, c_max=4, iteration=0, total_iters=10)
        # lambda_s=0 at progress=0 → shaped=1.0 before bonus, then +0.1
        assert scaffolded.metadata["_esc_shaped_reward"] == pytest.approx(1.1)
        # Naked unchanged
        assert naked.metadata["_esc_shaped_reward"] == pytest.approx(1.0)

    def test_prefer_scaffolded_no_bonus_when_naked_fails(self):
        """Naked fails (reward=0): bonus does NOT apply even if scaffolded passes."""
        rs = RewardShapingConfig(prefer_scaffolded=True, prefer_scaffolded_bonus=0.1)
        scaffolded = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked = self._make_rollout(reward=0.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([scaffolded, naked], rs, c_max=4, iteration=0, total_iters=10)
        assert scaffolded.metadata["_esc_shaped_reward"] == pytest.approx(1.0)

    def test_prefer_scaffolded_no_bonus_when_scaffolded_fails(self):
        """Naked passes but scaffolded fails: bonus not applied (scaffolded raw < 1.0)."""
        rs = RewardShapingConfig(prefer_scaffolded=True, prefer_scaffolded_bonus=0.1)
        scaffolded = self._make_rollout(reward=0.0, allocated_chunks=2)
        naked = self._make_rollout(reward=1.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([scaffolded, naked], rs, c_max=4, iteration=0, total_iters=10)
        assert scaffolded.metadata["_esc_shaped_reward"] == pytest.approx(0.0)

    def test_prefer_scaffolded_disabled_by_default(self):
        """prefer_scaffolded=False (default): no bonus applied when both pass."""
        rs = RewardShapingConfig()  # prefer_scaffolded=False
        scaffolded = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked = self._make_rollout(reward=1.0, allocated_chunks=0, is_naked=True)
        _apply_group_shaping([scaffolded, naked], rs, c_max=4, iteration=0, total_iters=10)
        assert scaffolded.metadata["_esc_shaped_reward"] == pytest.approx(1.0)

    def test_prefer_scaffolded_bonus_after_lambda_shaping(self):
        """Bonus is applied AFTER λ_s shaping: shaped = raw*(1-λ_s*c_k/c_max) + bonus."""
        rs = RewardShapingConfig(
            lambda_s_max=0.1, prefer_scaffolded=True, prefer_scaffolded_bonus=0.1
        )
        scaffolded = self._make_rollout(reward=1.0, allocated_chunks=2)
        naked = self._make_rollout(reward=1.0, allocated_chunks=0, is_naked=True)
        # progress=1.0 → lambda_s=0.1; shaped before bonus = 1.0*(1-0.1*2/4)=0.95; +0.1=1.05
        _apply_group_shaping([scaffolded, naked], rs, c_max=4, iteration=10, total_iters=10)
        assert scaffolded.metadata["_esc_shaped_reward"] == pytest.approx(1.05)


# ── TestVerticalRollout ───────────────────────────────────────────────────────


class TestVerticalRollout:
    def test_naked_generates_thinking_and_answer(self):
        backend = _ControlledBackend(["my thinking</think>", "A"])
        ctx = _make_ctx(backend, segments=[])
        _vertical_rollout(ctx)
        result = ctx.build_result()
        assert "my thinking" in result.completion_text
        assert "\\boxed{" in result.completion_text

    def test_single_segment_inject_then_generate(self):
        backend = _ControlledBackend(["continuation</think>", "B"])
        ctx = _make_ctx(backend, segments=["prefix chunk"])
        _vertical_rollout(ctx)
        result = ctx.build_result()
        assert "prefix chunk" in result.completion_text
        assert "continuation" in result.completion_text
        assert "\\boxed{" in result.completion_text

    def test_stops_scaffold_cycle_on_think_close(self):
        backend = _ControlledBackend(["done</think>", "C", "unused", "unused"])
        vert = VerticalConfig(n_chunks=3, stop_on_close=True, close_inject=False)
        ctx = _make_ctx(backend, segments=["s1", "s2", "s3"], vert_cfg=vert)
        _vertical_rollout(ctx)
        result = ctx.build_result()
        assert "s2" not in result.completion_text
        assert "s3" not in result.completion_text

    def test_close_inject_when_no_think_close(self):
        backend = _ControlledBackend(["still thinking", "D"])
        vert = VerticalConfig(n_chunks=1, stop_on_close=True, close_inject=True)
        ctx = _make_ctx(backend, segments=["s1"], vert_cfg=vert)
        _vertical_rollout(ctx)
        result = ctx.build_result()
        assert "</think>" in result.completion_text

    def test_no_close_inject_when_disabled(self):
        backend = _ControlledBackend(["still thinking", "E"])
        vert = VerticalConfig(n_chunks=1, stop_on_close=True, close_inject=False)
        ctx = _make_ctx(backend, segments=["s1"], vert_cfg=vert)
        _vertical_rollout(ctx)
        result = ctx.build_result()
        assert "</think>" not in result.completion_text

    def test_answer_phase_always_runs(self):
        for segments in [[], ["s1"], ["s1", "s2"]]:
            backend = _ControlledBackend(["thinking</think>", "X", "X"])
            ctx = _make_ctx(backend, segments=segments)
            _vertical_rollout(ctx)
            result = ctx.build_result()
            assert "\\boxed{" in result.completion_text

    def test_answer_captured_in_variables(self):
        """_vertical_rollout sets result.variables['answer']."""
        backend = _ControlledBackend(["thinking</think>", "D"])
        ctx = _make_ctx(backend, segments=[])
        _vertical_rollout(ctx)
        result = ctx.build_result()
        assert "answer" in result.variables

    def test_answer_capture_reward_extraction(self):
        """With ctx.capture, answer variable is set (reward extractor can find it)."""
        backend = _ControlledBackend(["thinking</think>", "D"])
        ctx = _make_ctx(backend, segments=[])
        _vertical_rollout(ctx)
        result = ctx.build_result()
        av = result.variables.get("answer")
        assert av is not None
        assert "D" in av.text


# ── TestEscalationConfig ──────────────────────────────────────────────────────


class TestEscalationConfig:
    def test_defaults(self):
        cfg = EscalationConfig()
        assert cfg.enabled is False
        assert isinstance(cfg.vertical, VerticalConfig)
        assert isinstance(cfg.horizontal, HorizontalConfig)
        assert isinstance(cfg.scaffold, EscalationScaffoldConfig)
        assert isinstance(cfg.tracking, EscalationTrackingConfig)
        assert isinstance(cfg.reward_shaping, RewardShapingConfig)
        assert isinstance(cfg.tri_factor, TriFactorConfig)

    def test_build_none_returns_default(self):
        cfg = build_escalation_config(None)
        assert cfg.enabled is False

    def test_build_empty_dict_returns_default(self):
        cfg = build_escalation_config({})
        assert cfg.enabled is False

    def test_build_enabled(self):
        cfg = build_escalation_config({"enabled": True})
        assert cfg.enabled is True

    def test_build_vertical_subconfig(self):
        cfg = build_escalation_config({"enabled": True, "vertical": {"n_chunks": 8}})
        assert cfg.vertical.n_chunks == 8
        assert cfg.vertical.stop_on_close is True

    def test_build_horizontal_subconfig(self):
        cfg = build_escalation_config({"horizontal": {"reward_threshold": 0.9, "max_members": 4}})
        assert cfg.horizontal.reward_threshold == 0.9
        assert cfg.horizontal.max_members == 4
        assert cfg.horizontal.naked_always is True

    def test_build_scaffold_subconfig(self):
        cfg = build_escalation_config({"scaffold": {"target": "answer", "max_tokens": 128}})
        assert cfg.scaffold.target == "answer"
        assert cfg.scaffold.max_tokens == 128

    def test_build_tracking_subconfig(self):
        cfg = build_escalation_config({"tracking": {"wandb_prefix": "myesc"}})
        assert cfg.tracking.wandb_prefix == "myesc"

    def test_build_reward_shaping_subconfig(self):
        """build_escalation_config parses reward_shaping block (v5)."""
        cfg = build_escalation_config(
            {"reward_shaping": {"lambda_s_max": 0.03, "lambda_f_max": 0.01}}
        )
        assert cfg.reward_shaping.lambda_s_max == pytest.approx(0.03)
        assert cfg.reward_shaping.lambda_f_max == pytest.approx(0.01)

    def test_build_tri_factor_subconfig(self):
        """build_escalation_config parses tri_factor block (v5)."""
        cfg = build_escalation_config(
            {"tri_factor": {"alpha_h": 0.6, "alpha_v": 0.7, "alpha_p": 0.8, "b_ext": 32}}
        )
        assert cfg.tri_factor.alpha_h == pytest.approx(0.6)
        assert cfg.tri_factor.alpha_v == pytest.approx(0.7)
        assert cfg.tri_factor.b_ext == 32

    def test_tri_factor_config_defaults(self):
        """TriFactorConfig defaults match design spec."""
        tf = TriFactorConfig()
        assert tf.alpha_h == pytest.approx(0.7)
        assert tf.alpha_v == pytest.approx(0.8)
        assert tf.alpha_p == pytest.approx(0.9)
        assert tf.b_ext == 64

    def test_unknown_keys_ignored(self):
        cfg = build_escalation_config({"enabled": True, "future_key": "future_value"})
        assert cfg.enabled is True

    def test_unknown_subconfig_keys_ignored(self):
        cfg = build_escalation_config({"vertical": {"n_chunks": 3, "unknown": 99}})
        assert cfg.vertical.n_chunks == 3


# ── TestMergeEscCfg ───────────────────────────────────────────────────────────


class TestMergeEscCfg:
    def _base(self) -> EscalationConfig:
        return build_escalation_config({"enabled": True, "horizontal": {"max_members": 4}})

    def test_none_overrides_returns_base(self):
        base = self._base()
        assert merge_esc_cfg(base, None) is base

    def test_empty_dict_returns_base(self):
        base = self._base()
        assert merge_esc_cfg(base, {}) is base

    def test_override_horizontal_field(self):
        base = self._base()
        result = merge_esc_cfg(base, {"horizontal": {"forced_answer_on_last_member": True}})
        assert result.horizontal.forced_answer_on_last_member is True
        assert result.horizontal.max_members == 4  # preserved from base

    def test_override_horizontal_multiple_fields(self):
        base = self._base()
        result = merge_esc_cfg(
            base,
            {"horizontal": {"forced_answer_inject_chars": 5, "forced_answer_end_prob": 0.3}},
        )
        assert result.horizontal.forced_answer_inject_chars == 5
        assert result.horizontal.forced_answer_end_prob == pytest.approx(0.3)

    def test_override_scaffold_max_tokens(self):
        base = self._base()
        result = merge_esc_cfg(base, {"scaffold": {"max_tokens": 256}})
        assert result.scaffold.max_tokens == 256
        assert result.scaffold.boundary_mode == "sentence"  # preserved

    def test_override_vertical_field(self):
        base = self._base()
        result = merge_esc_cfg(base, {"vertical": {"n_chunks": 8}})
        assert result.vertical.n_chunks == 8

    def test_override_top_level_enabled(self):
        base = self._base()
        result = merge_esc_cfg(base, {"enabled": False})
        assert result.enabled is False

    def test_override_tri_factor(self):
        base = self._base()
        result = merge_esc_cfg(base, {"tri_factor": {"alpha_h": 0.5}})
        assert result.tri_factor.alpha_h == pytest.approx(0.5)
        assert result.tri_factor.b_ext == 64  # preserved

    def test_unknown_keys_ignored(self):
        base = self._base()
        result = merge_esc_cfg(base, {"future_field": 999})
        assert result.horizontal.max_members == 4  # unchanged

    def test_split_boundaries_list_coerced_to_tuple(self):
        base = self._base()
        result = merge_esc_cfg(base, {"scaffold": {"split_boundaries": ["\n\n", "\n"]}})
        assert result.scaffold.split_boundaries == ("\n\n", "\n")

    def test_non_dict_value_for_sub_ignored(self):
        base = self._base()
        result = merge_esc_cfg(base, {"horizontal": "not-a-dict"})
        assert result is base  # no-op, non-dict sub ignored


# ── TestCollectEscalationStats ────────────────────────────────────────────────


class TestCollectEscalationStats:
    def _make_result(self, member_idx, n_eff, reward, early_stop=False):
        r = RolloutResult(type_name="test")
        r.metadata["_esc_member_idx"] = member_idx
        r.metadata["_esc_n_eff"] = n_eff  # updated key
        r.metadata["_esc_reward"] = reward
        if early_stop:
            r.metadata["_esc_early_stop"] = True
        return r

    def test_empty_returns_empty(self):
        assert collect_escalation_stats([]) == {}

    def test_no_esc_metadata_returns_empty(self):
        r = RolloutResult(type_name="test")
        assert collect_escalation_stats([r]) == {}

    def test_basic_stats(self):
        members = [
            self._make_result(0, 0, 0.5),
            self._make_result(1, 1, 1.0, early_stop=True),
            self._make_result(2, 0, 0.3),
        ]
        stats = collect_escalation_stats(members)
        assert stats["esc_n_members"] == 3
        assert stats["esc_member_rewards"] == [0.5, 1.0, 0.3]
        assert abs(stats["esc_mean_reward"] - (0.5 + 1.0 + 0.3) / 3) < 1e-9
        assert stats["esc_max_reward"] == 1.0
        assert stats["esc_chunks_used"] == [0, 1, 0]
        assert stats["esc_early_stop"] is True

    def test_no_early_stop(self):
        members = [self._make_result(0, 0, 0.2), self._make_result(1, 1, 0.4)]
        stats = collect_escalation_stats(members)
        assert stats["esc_early_stop"] is False


# ── TestRunEscalation ─────────────────────────────────────────────────────────


class TestRunEscalation:
    def _make_esc_cfg(self, **kw) -> EscalationConfig:
        horiz_kw = {k: kw.pop(k) for k in list(kw) if k in HorizontalConfig.__dataclass_fields__}
        horiz = HorizontalConfig(**horiz_kw) if horiz_kw else HorizontalConfig()
        return EscalationConfig(enabled=True, horizontal=horiz, **kw)

    def test_returns_list_of_rollout_results(self):
        backend = _ControlledBackend([f"gen{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, naked_always=True),
            vertical=VerticalConfig(n_chunks=2),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 32, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("prompt"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, RolloutResult) for r in results)

    def test_naked_always_appended(self):
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True, horizontal=HorizontalConfig(max_members=2, naked_always=True)
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        naked = [r for r in results if r.metadata.get("_view_is_naked")]
        assert len(naked) >= 1
        assert results[-1].metadata.get("_view_name") == "naked"

    def test_early_stop_on_reward_threshold(self):
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(1.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=4, reward_threshold=1.0, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert len(results) == 2
        assert results[-1].metadata.get("_view_name") == "naked"
        assert results[-1].metadata.get("_esc_early_stop") is True

    def test_no_early_stop_when_reward_below_threshold(self):
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=3, reward_threshold=1.0, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert len(results) == 4  # 3 members + 1 naked (no early-stop, reward=0 < 1.0)

    def test_member_metadata_stamped(self):
        """New metadata schema: _esc_allocated_chunks and _esc_n_eff present (v5)."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.5)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, reward_threshold=1.0, naked_always=False),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert "_esc_member_idx" in r.metadata
            assert "_esc_allocated_chunks" in r.metadata  # new key
            assert "_esc_n_eff" in r.metadata  # new key
            assert "_esc_reward" in r.metadata
            assert "_view_name" in r.metadata

    def test_no_esc_n_chunks_in_metadata(self):
        """Old _esc_n_chunks key must NOT appear in any member metadata (v5 regression)."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert "_esc_n_chunks" not in r.metadata

    def test_n_eff_stored(self):
        """_esc_n_eff is stored per member."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, naked_always=False),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert "_esc_n_eff" in r.metadata
            assert isinstance(r.metadata["_esc_n_eff"], int)

    def test_allocated_vs_n_eff_metadata(self):
        """_esc_allocated_chunks (C_k) and _esc_n_eff (N_eff) are separate fields."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=3, naked_always=False),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert "_esc_allocated_chunks" in r.metadata
            assert "_esc_n_eff" in r.metadata
            # Both are integers
            assert isinstance(r.metadata["_esc_allocated_chunks"], int)
            assert isinstance(r.metadata["_esc_n_eff"], int)

    def test_reward_stores_in_metadata(self):
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.5)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, reward_threshold=1.0, naked_always=False),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert r.metadata.get("_esc_reward") == pytest.approx(0.5)

    def test_shaped_reward_in_metadata(self):
        """_esc_shaped_reward is set on all members after group-level shaping."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.5)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=5,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert "_esc_shaped_reward" in r.metadata

    def test_view_name_stored(self):
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, reward_threshold=1.0, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        names = [r.metadata.get("_view_name") for r in results]
        assert "esc_0" in names
        assert "naked" in names

    def test_naked_false_no_naked_view(self):
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, reward_threshold=1.0, naked_always=False),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert len(results) == 2
        assert not any(r.metadata.get("_view_name") == "naked" for r in results)

    def test_esc_weights_in_metadata(self):
        """_esc_weights list is stored per member."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert "_esc_weights" in r.metadata
            assert isinstance(r.metadata["_esc_weights"], list)


# ── TestTrainerEscalationDispatch ─────────────────────────────────────────────


class TestTrainerEscalationDispatch:
    def _make_dataset(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "q", "answer": "a", "type": "math"}) + "\n")
            f.flush()
            return DatasetLoader(DatasetConfig(path=f.name, shuffle=False)).load()

    def test_trainer_accepts_esc_cfg(self):
        from mlx_grpo.training import Trainer

        backend = _ControlledBackend()
        dataset = self._make_dataset()
        config = TrainingConfig(iterations=0, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=backend, esc_cfg=None)
        assert trainer._esc_cfg is None

    def test_trainer_stores_esc_cfg(self):
        from mlx_grpo.training import Trainer

        backend = _ControlledBackend()
        dataset = self._make_dataset()
        config = TrainingConfig(iterations=0, group_size=2, save_every=0, eval_every=0)
        esc_cfg = EscalationConfig(enabled=True)
        trainer = Trainer(config=config, dataset=dataset, backend=backend, esc_cfg=esc_cfg)
        assert trainer._esc_cfg is esc_cfg
        assert trainer._esc_cfg.enabled is True


# ── TestConfigYAMLEscalation ──────────────────────────────────────────────────


class TestConfigYAMLEscalation:
    def test_load_config_without_escalation(self):
        from mlx_grpo.config import load_config

        yaml_content = "training:\n  model_path: test\n  iterations: 10\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        assert cfg.escalation is None

    def test_load_config_with_escalation_disabled(self):
        from mlx_grpo.config import load_config

        yaml_content = "training:\n  model_path: test\nescalation:\n  enabled: false\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        assert cfg.escalation is not None
        assert cfg.escalation.enabled is False

    def test_load_config_with_escalation_enabled(self):
        from mlx_grpo.config import load_config

        yaml_content = (
            "training:\n  model_path: test\nescalation:\n  enabled: true\n"
            "  vertical:\n    n_chunks: 6\n  horizontal:\n    reward_threshold: 0.9\n"
            "    max_members: 4\n  scaffold:\n    max_tokens: 256\n"
            "  tracking:\n    wandb_prefix: mytest\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        esc = cfg.escalation
        assert esc is not None
        assert esc.enabled is True
        assert esc.vertical.n_chunks == 6
        assert esc.horizontal.reward_threshold == 0.9
        assert esc.horizontal.max_members == 4
        assert esc.scaffold.max_tokens == 256
        assert esc.tracking.wandb_prefix == "mytest"


# ── TestISWeightsIntegration ──────────────────────────────────────────────────


class TestISWeightsIntegration:
    """Verify IS weight assembly logic for escalation rollouts."""

    def _make_rollout_with_esc_weights(self, weights: list[float]) -> RolloutResult:
        r = RolloutResult(type_name="test")
        r.metadata["_esc_weights"] = weights
        return r

    def test_esc_weights_detected_without_is_config(self):
        """_has_is must be True when _esc_weights present, even without _is_cfg (B fix)."""
        rollout = self._make_rollout_with_esc_weights([1.0, 1.0, 0.8])
        rollouts = [rollout]
        is_cfg = None  # no traditional IS config

        _has_is = (
            is_cfg is not None
            or any(r.metadata.get("_is_override") for r in rollouts)
            or any(r.metadata.get("_esc_weights") for r in rollouts)
        )
        assert _has_is is True

    def test_esc_weights_not_detected_without_flag(self):
        """Without _esc_weights, _has_is=False when no IS config."""
        rollout = RolloutResult(type_name="test")  # no _esc_weights
        rollouts = [rollout]
        is_cfg = None

        _has_is = (
            is_cfg is not None
            or any(r.metadata.get("_is_override") for r in rollouts)
            or any(r.metadata.get("_esc_weights") for r in rollouts)
        )
        assert _has_is is False

    def test_is_weights_use_esc_weights_when_present(self):
        """Rollout with _esc_weights → those weights used directly."""
        weights = [0.5, 0.8, 1.0]
        rollout = self._make_rollout_with_esc_weights(weights)
        masks = [[True, True, True]]
        rollouts = [rollout]

        _sw: list[list[float] | None] = []
        for i, r in enumerate(rollouts):
            esc_w = r.metadata.get("_esc_weights")
            if esc_w is not None:
                n = len(masks[i])
                _sw.append(esc_w[:n])
                continue
            _sw.append(None)

        assert _sw[0] == [0.5, 0.8, 1.0]

    def test_is_weights_fallback_without_esc_weights(self):
        """Rollout without _esc_weights → _sw entry is None (falls through to IS config)."""
        rollout = RolloutResult(type_name="test")  # no _esc_weights
        masks = [[True, True]]
        rollouts = [rollout]

        _sw: list[list[float] | None] = []
        for i, r in enumerate(rollouts):
            esc_w = r.metadata.get("_esc_weights")
            if esc_w is not None:
                n = len(masks[i])
                _sw.append(esc_w[:n])
                continue
            _sw.append(None)  # IS config path handles this

        assert _sw[0] is None

    def test_shaped_reward_used_for_advantage(self):
        """run_escalation sets _esc_shaped_reward; training loop should use it."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(1.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, naked_always=True),
            reward_shaping=RewardShapingConfig(lambda_s_max=0.05),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=5,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        # All members with reward=1.0: shaped < 1.0 on success path (non-naked)
        scaffold_members = [r for r in results if not r.metadata.get("_view_is_naked")]
        for r in scaffold_members:
            shaped = r.metadata["_esc_shaped_reward"]
            raw = r.metadata["_esc_reward"]
            assert shaped <= raw  # success path: discount applied (progress=0.5)


# ── TestAdaptiveMemberCount (v6) ──────────────────────────────────────────────


class TestAdaptiveMemberCount:
    """v6.1: adaptive cap (n_eff_max+1), allocation spans n_eff_max, c_max=n_eff_max."""

    # ── _count_n_eff_max unit tests ──

    def test_count_n_eff_max_empty(self):
        """Empty trace → 0."""
        assert _count_n_eff_max("") == 0
        assert _count_n_eff_max("   ") == 0

    def test_count_n_eff_max_no_boundaries(self):
        """Trace with no sentence boundaries → 1 (floor)."""
        assert _count_n_eff_max("<think>no boundaries here</think>") == 1

    def test_count_n_eff_max_counts_boundaries(self):
        """Trace with 3 boundaries → 3."""
        trace = "<think>A.\nB.\nC.\n</think>\\boxed{3}"
        assert _count_n_eff_max(trace) == 3

    def test_count_n_eff_max_stops_at_close_think(self):
        """Content after </think> not counted."""
        trace = "<think>A.\n</think>B.\nC.\nD.\n"
        assert _count_n_eff_max(trace) == 1

    def test_count_n_eff_max_line_mode_counts_newlines(self):
        """Line mode counts every \\n, not just sentence-ending ones."""
        # 4 newlines inside <think> → 4 boundaries (no .\n needed)
        trace = "<think>Step 1\nStep 2\nStep 3\nStep 4\n</think>"
        assert _count_n_eff_max(trace, boundary_mode="line") == 4

    def test_count_n_eff_max_line_mode_no_sentence_boundaries(self):
        """Line mode succeeds on structured steps that have no .\n patterns."""
        trace = "<think>[Step 1: LLN]\nanalysis here\n[Step 2: Pos]\nmore here\n</think>"
        sentence = _count_n_eff_max(trace, boundary_mode="sentence")
        line = _count_n_eff_max(trace, boundary_mode="line")
        assert sentence == 1  # no .\n → falls back to 1
        assert line == 4  # 4 newlines

    def test_count_n_eff_max_line_mode_empty(self):
        """Line mode: empty trace → 0."""
        assert _count_n_eff_max("", boundary_mode="line") == 0
        assert _count_n_eff_max("<think></think>", boundary_mode="line") == 0

    def test_plan_adaptive_scaffold_line_mode_splits_at_newlines(self):
        """Line mode splits a structured trace at \\n boundaries, not .\\n."""
        trace = "<think>[Step 1: LLN]\nanalysis\n[Step 2: Pos]\nresult\n</think>"
        backend = _ControlledBackend()
        segs, n_eff, b_eff = _plan_adaptive_scaffold(trace, 2, backend, 512, boundary_mode="line")
        assert n_eff == 2
        assert len(segs) == 2
        assert all(s.strip() for s in segs)

    def test_plan_adaptive_scaffold_line_mode_vs_sentence_mode(self):
        """Line mode finds more boundaries than sentence mode on step-format traces."""
        trace = "<think>[Step 1: LLN]\nanalysis\n[Step 2: Pos]\nresult\n</think>"
        backend = _ControlledBackend()
        # Sentence mode: no .\n → token fallback
        segs_s, n_s, _ = _plan_adaptive_scaffold(trace, 2, backend, 512, boundary_mode="sentence")
        # Line mode: snaps to \n boundaries
        segs_l, n_l, _ = _plan_adaptive_scaffold(trace, 2, backend, 512, boundary_mode="line")
        assert n_l == 2
        assert n_s >= 1  # token fallback still produces segments

    def test_plan_adaptive_scaffold_line_mode_respects_max_tokens(self):
        """Line mode still caps each chunk at max_tokens."""
        trace = "<think>" + "word\n" * 20 + "</think>"
        backend = _ControlledBackend()
        segs, n_eff, _ = _plan_adaptive_scaffold(
            trace, 3, backend, max_tokens=3, boundary_mode="line"
        )
        assert all(len(backend.encode(s)) <= 3 for s in segs)

    # ── Adaptive member cap integration tests ──

    def test_adaptive_cap_short_trace(self):
        """n_eff_max=1 → scaffolded_max=min(max_members, 2): at most 2 scaffold members."""
        sample = _make_sample(answer="<think>One sentence only.\n</think>\\boxed{1}")
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=6, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        # n_eff_max=1 → scaffolded_max=min(6, 1+1)=2 → 2 scaffolded + 1 naked = 3
        assert len(results) == 3
        scaffolded = [r for r in results if not r.metadata.get("_view_is_naked")]
        assert len(scaffolded) == 2

    def test_adaptive_cap_no_reduction_rich_trace(self):
        """n_eff_max >= max_members → no cap: all configured members run."""
        trace = "<think>" + "Sentence.\n" * 5 + "</think>\\boxed{5}"  # 5 boundaries
        sample = _make_sample(answer=trace)
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=4, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        # n_eff_max=5 → scaffolded_max=min(4, 5+1)=4 → no cap: 4 scaffolded + 1 naked = 5
        assert len(results) == 5

    def test_adaptive_member_cap_full_trace(self):
        """Expert trace with 20+ boundaries → no cap, effective_members = config.members."""
        trace = "<think>" + "Sentence.\n" * 20 + "</think>\\boxed{x}"
        sample = _make_sample(answer=trace)
        backend = _ControlledBackend([f"g{i}" for i in range(30)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=6, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        # n_eff_max=20 >= max_members=6 → no cap: 6 scaffolded + 1 naked = 7
        assert len(results) == 7

    def test_allocation_spans_n_eff_max(self):
        """C_k allocation spans 1..n_eff_max across scaffolded members; last member hits C_max."""
        # 3 boundaries → n_eff_max=3, max_members=4 → scaffolded_max=min(4,4)=4, c_max=3
        trace = "<think>A.\nB.\nC.\n</think>\\boxed{3}"
        sample = _make_sample(answer=trace)
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=4, naked_always=False),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        # step=3/4=0.75 → C_k = [1, 2, 2, 3]; last member reaches n_eff_max=3
        assert len(results) == 4
        chunks = [r.metadata["_esc_allocated_chunks"] for r in results]
        assert chunks[-1] == 3  # last member hits c_max=n_eff_max
        assert chunks[0] == 1  # first member starts at 1

    def test_reward_shaping_uses_n_eff_max(self):
        """Shaped reward denominator C_max = n_eff_max (not config.max_members)."""
        # 2 boundaries → n_eff_max=2, max_members=4 → scaffolded_max=3, c_max=2
        trace = "<think>Step one.\nStep two.\n</think>\\boxed{2}"
        sample = _make_sample(answer=trace)
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        lambda_s_max = 0.10
        reward_fn = _make_reward_fn(1.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=4, naked_always=False, reward_threshold=1.1),
            reward_shaping=RewardShapingConfig(lambda_s_max=lambda_s_max),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=5,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        # c_max=n_eff_max=2, step=2/3=0.667 → c_k=[1,1,2]; progress=0.5, lambda_s=0.05
        # member 2 (last): c_k=2, shaped = 1.0*(1 - 0.05*2/2) = 0.95
        assert len(results) == 3
        last = results[-1]
        assert last.metadata["_esc_allocated_chunks"] == 2  # c_k=n_eff_max
        expected = 1.0 * (1.0 - lambda_s_max * 0.5 * 2.0 / 2)  # C_k/c_max=1.0
        assert abs(last.metadata["_esc_shaped_reward"] - expected) < 1e-6

    def test_horizontal_factor_cmax_is_n_eff_max(self):
        """Tri-factor alpha_h exponent uses n_eff_max as C_max denominator."""
        injected = [1, 2]
        generated = [3, 4]
        result = _make_result_with_segments(injected_toks=injected, generated_toks=generated)
        tf = TriFactorConfig(alpha_h=0.7, alpha_v=1.0, alpha_p=1.0, b_ext=0)
        # c_max=n_eff_max=1: alpha_h^(1/1) = 0.7 (full dampening)
        weights_adaptive = _build_tri_factor_weights(result, c_k=1, c_max=1, n_eff=1, tf_cfg=tf)
        gen_weights = weights_adaptive[len(injected) :]  # skip injected
        assert all(abs(w - 0.7) < 1e-6 for w in gen_weights)
        # Old c_max=12: alpha_h^(1/12) ≈ 0.967 (near-zero dampening — the bug)
        weights_old = _build_tri_factor_weights(result, c_k=1, c_max=12, n_eff=1, tf_cfg=tf)
        gen_weights_old = weights_old[len(injected) :]
        assert all(w > 0.95 for w in gen_weights_old)

    def test_answer_extraction_no_trailing_text(self):
        """`_extract_last_boxed` handles trailing text after \\boxed{} in completion."""
        from mlx_grpo.types.builtins.math_helpers import _extract_last_boxed

        # Simple answer with trailing reasoning (the observed live-training bug)
        completion = "some thinking\n\\boxed{6}\n\nWait, hold on, let me reconsider."
        assert _extract_last_boxed(completion) == "6"

        # Nested braces (fraction answer) — must not be truncated
        completion2 = "thinking\n\\boxed{\\frac{1}{2}}\n\nActually wait..."
        assert _extract_last_boxed(completion2) == "\\frac{1}{2}"

        # Only the LAST boxed matters (scaffold prefix may contain earlier answers)
        completion3 = "scaffold \\boxed{3} model thinking \\boxed{6} trailing text"
        assert _extract_last_boxed(completion3) == "6"

    def test_answer_extraction_strips_trailing(self):
        """`_strip_answer_close` removes close-brace from raw answer capture text."""
        # MCQ: model generates "C}" inside \boxed{} context → extract "C"
        assert _strip_answer_close("C}") == "C"
        # Math: model continues after closing brace → extract answer only
        assert _strip_answer_close("8}\nWait, hold on, let me reconsider.") == "8"
        # No brace: return as-is
        assert _strip_answer_close("42") == "42"
        # Multi-char answer before brace
        assert _strip_answer_close("\\frac{1}{2}") == "\\frac{1"

    def test_n_eff_max_logged_in_metadata(self):
        """All rollouts carry _esc_n_eff_max and _esc_effective_members metadata."""
        trace = "<think>step one.\nstep two.\n</think>\\boxed{42}"
        sample = _make_sample(answer=trace)
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=6, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        for r in results:
            assert "_esc_n_eff_max" in r.metadata
            assert "_esc_effective_members" in r.metadata
            assert r.metadata["_esc_n_eff_max"] >= 1
            assert r.metadata["_esc_effective_members"] == len(results)


# ── Answer-prefix scaffold tests ────────────────────────────────────────────────


class TestAnswerPrefixText:
    """Tests for _answer_prefix_text helper."""

    def test_zero_tokens_returns_empty(self):
        """n_tokens=0 → always empty string."""
        assert _answer_prefix_text("\\boxed{7, 13, 22}", 0, _ControlledBackend()) == ""

    def test_empty_target_returns_empty(self):
        """Empty/whitespace target → empty string."""
        assert _answer_prefix_text("", 10, _ControlledBackend()) == ""
        assert _answer_prefix_text("   ", 10, _ControlledBackend()) == ""

    def test_extracts_from_boxed(self):
        """Extracts content from \\boxed{...} first."""
        backend = _ControlledBackend()
        result = _answer_prefix_text("\\boxed{7, 13, 22, 34}", 3, backend)
        # Content "7, 13, 22, 34" → first 3 tokens
        assert len(result) > 0

    def test_fallback_strips_thinking_tags(self):
        """Falls back to stripping <think> when no \\boxed present."""
        backend = _ControlledBackend()
        result = _answer_prefix_text("<think>ignore this</think>answer text", 2, backend)
        assert "think" not in result.lower() or len(result) == 0

    def test_caps_at_n_tokens(self):
        """Result is capped at n_tokens from the backend."""
        backend = _ControlledBackend()
        # "7, 13, 22, 34, 38, 45" — request just 1 token
        result = _answer_prefix_text("\\boxed{7, 13, 22, 34, 38, 45}", 1, backend)
        # 1 token max — result should be shorter than full content
        full = _answer_prefix_text("\\boxed{7, 13, 22, 34, 38, 45}", 100, backend)
        assert len(result) <= len(full)


class TestAnswerPrefixScaffold:
    """Integration tests for answer_prefix_tokens in run_escalation."""

    def test_default_zero_no_change(self):
        """answer_prefix_tokens=0 (default) → no answer_prefix_text in rollouts."""
        sample = _make_sample(answer="<think>think.\n</think>\\boxed{42}")
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        esc_cfg = EscalationConfig(
            enabled=True,
            scaffold=EscalationScaffoldConfig(answer_prefix_tokens=0),
            horizontal=HorizontalConfig(max_members=2, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 8, "answer_max_tokens": 4},
            reward_fn=_make_reward_fn(0.0),
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=32,
            esc_cfg=esc_cfg,
        )
        # When 0, all members should complete without error
        assert len(results) >= 1

    def test_nonzero_injects_decaying_prefix(self):
        """answer_prefix_tokens>0 → member 0 gets more prefix than member 1."""
        sample = _make_sample(answer="<think>think.\n</think>\\boxed{7, 13, 22}")
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        esc_cfg = EscalationConfig(
            enabled=True,
            scaffold=EscalationScaffoldConfig(answer_prefix_tokens=4, boundary_mode="line"),
            horizontal=HorizontalConfig(max_members=2, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 8, "answer_max_tokens": 8},
            reward_fn=_make_reward_fn(0.0),
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=32,
            esc_cfg=esc_cfg,
        )
        assert len(results) >= 1
        # Naked member never gets answer prefix
        naked = next((r for r in results if r.metadata.get("_view_is_naked")), None)
        if naked:
            injected_tags = [s.tag for s in naked.segments if s.tag == "answer_hint"]
            assert len(injected_tags) == 0

    def test_naked_never_gets_prefix(self):
        """Naked member always gets answer_prefix_text='' regardless of config."""
        sample = _make_sample(answer="\\boxed{7, 13, 22}")
        backend = _ControlledBackend([f"g{i}" for i in range(20)])
        trainer = _make_trainer(backend)
        esc_cfg = EscalationConfig(
            enabled=True,
            scaffold=EscalationScaffoldConfig(answer_prefix_tokens=10),
            horizontal=HorizontalConfig(max_members=1, naked_always=True),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 8, "answer_max_tokens": 8},
            reward_fn=_make_reward_fn(0.0),
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=32,
            esc_cfg=esc_cfg,
        )
        naked = next((r for r in results if r.metadata.get("_view_is_naked")), None)
        assert naked is not None
        # Naked member: no answer_hint segment injected
        hint_segs = [s for s in naked.segments if s.tag == "answer_hint"]
        assert len(hint_segs) == 0

    def test_config_field_default_off(self):
        """EscalationScaffoldConfig.answer_prefix_tokens defaults to 0 (non-breaking)."""
        cfg = EscalationScaffoldConfig()
        assert cfg.answer_prefix_tokens == 0

    def test_config_parsed_from_yaml_dict(self):
        """answer_prefix_tokens is parsed from YAML dict via build_escalation_config."""
        from mlx_grpo.config.escalation import build_escalation_config

        data = {"enabled": True, "scaffold": {"answer_prefix_tokens": 5}}
        esc_cfg = build_escalation_config(data)
        assert esc_cfg.scaffold.answer_prefix_tokens == 5


class TestAnswerStop:
    """answer_stop from type_config prevents hard-zero in lotto-style tasks."""

    def test_no_stop_by_default(self):
        """Without answer_stop, answer generation has no stop sequence — existing math behavior."""
        backend = _ControlledBackend(["think_content", "42"], stop_at="}")
        ctx = _make_ctx(backend, type_cfg={"thinking_budget": 16, "answer_max_tokens": 8})
        _vertical_rollout(ctx)
        result = ctx.build_result()
        # answer segment present with no stop-at-} applied (stop_at="}" not in params.stop)
        answer_segs = [s for s in result.segments if s.tag == "answer"]
        assert len(answer_segs) == 1
        # text is "42" (not truncated by "}" stop since no stop passed)
        assert "42" in answer_segs[0].text

    def test_answer_stop_respected_in_generate(self):
        """answer_stop=['}'] is forwarded to ctx.generate — model stops at '}'."""
        # Backend returns answer text containing '}'; stop_at="}" trims it.
        backend = _ControlledBackend(
            ["thinking", "5, 14, 22, 31, 38, 42} extra reasoning"],
            stop_at="}",
        )
        ctx = _make_ctx(
            backend,
            type_cfg={"thinking_budget": 16, "answer_max_tokens": 32, "answer_stop": ["}"]},
        )
        _vertical_rollout(ctx)
        result = ctx.build_result()
        answer_segs = [s for s in result.segments if s.tag == "answer"]
        assert len(answer_segs) == 1
        # Backend consumed '}' as stop → answer text has no trailing "} extra reasoning"
        assert "}" not in answer_segs[0].text
        assert "extra" not in answer_segs[0].text

    def test_answer_stop_empty_list_means_no_stop(self):
        """answer_stop=[] (empty) treated same as no stop — no stop sequence passed."""
        backend = _ControlledBackend(["thinking", "ok"], stop_at="}")
        ctx = _make_ctx(
            backend,
            type_cfg={"thinking_budget": 16, "answer_max_tokens": 8, "answer_stop": []},
        )
        _vertical_rollout(ctx)
        result = ctx.build_result()
        answer_segs = [s for s in result.segments if s.tag == "answer"]
        assert len(answer_segs) == 1


# ── Jaccard anchor tests ────────────────────────────────────────────────────────


class TestJaccardAnchorFromNaked:
    def test_returns_empty_when_no_overlap(self):
        assert _jaccard_anchor_from_naked("completely different words", "xyz abc mno") == ""

    def test_returns_empty_when_either_empty(self):
        assert _jaccard_anchor_from_naked("", "some text") == ""
        assert _jaccard_anchor_from_naked("some text", "") == ""

    def test_extracts_expert_sentences_with_shared_terms(self):
        naked = "The question concerns rotation hypothesis construct validity."
        expert = (
            "The rotation hypothesis explains construct validity.\n"
            "It also accounts for inter-rater reliability.\n"
            "Random noise is unrelated to this model."
        )
        result = _jaccard_anchor_from_naked(naked, expert)
        assert result != ""
        assert "rotation" in result.lower() or "construct" in result.lower()

    def test_respects_max_words(self):
        naked = "probability distribution estimation statistical inference model"
        expert = (
            "Statistical inference requires probability distribution estimation. "
            "Bayesian model estimation uses prior probability distribution updates. "
            "Frequentist estimation relies on statistical inference procedures."
        )
        result = _jaccard_anchor_from_naked(naked, expert, max_words=10)
        assert len(result.split()) <= 15  # slight slack for sentence splits

    def test_skips_zero_score_sentences(self):
        naked = "quantum entanglement superposition measurement"
        expert = "This sentence has nothing relevant. Quantum entanglement matters here."
        result = _jaccard_anchor_from_naked(naked, expert)
        assert "Quantum" in result or "quantum" in result

    def test_filters_stopwords(self):
        naked = "this that with from they have been"
        expert = "this that with some more would could"
        assert _jaccard_anchor_from_naked(naked, expert) == ""

    def test_returns_string_type(self):
        result = _jaccard_anchor_from_naked("hypothesis testing significance", "hypothesis test")
        assert isinstance(result, str)


# ── naked_first config and run_escalation tests ────────────────────────────────


class TestNakedFirst:
    def _esc_cfg_naked_first(self, reward_threshold: float = 1.0) -> EscalationConfig:
        return EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(
                max_members=2,
                naked_always=True,
                naked_first=True,
                reward_threshold=reward_threshold,
            ),
        )

    def test_naked_first_config_field_defaults_false(self):
        h = HorizontalConfig()
        assert h.naked_first is False

    def test_naked_first_config_parses_from_yaml(self):
        cfg = build_escalation_config({"horizontal": {"naked_first": True}})
        assert cfg.horizontal.naked_first is True

    def test_naked_passes_returns_only_naked(self):
        """When naked reward >= threshold, only 1 rollout returned (the naked)."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(1.0)  # naked always passes
        esc_cfg = self._esc_cfg_naked_first(reward_threshold=1.0)
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert len(results) == 1
        assert results[0].metadata["_view_is_naked"] is True

    def test_naked_fails_runs_scaffolded(self):
        """When naked fails, scaffolded members run."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)  # naked always fails
        esc_cfg = self._esc_cfg_naked_first(reward_threshold=1.0)
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert len(results) > 1
        assert results[0].metadata["_view_is_naked"] is True  # naked is first
        scaffolded = [r for r in results if not r.metadata.get("_view_is_naked")]
        assert len(scaffolded) >= 1

    def test_naked_first_naked_is_index_zero(self):
        """naked_first=True: naked result is first in list."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = self._esc_cfg_naked_first()
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert results[0].metadata.get("_view_name") == "naked"

    def test_skip_if_naked_correct_false_runs_full_ladder(self):
        """skip_if_naked_correct=False: scaffolded members run even when naked passes."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(1.0)  # naked always passes
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(
                max_members=2,
                naked_always=True,
                naked_first=True,
                skip_if_naked_correct=False,
                reward_threshold=1.0,
            ),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        # naked (1) + scaffolded members still run despite naked passing
        assert len(results) > 1
        assert results[0].metadata["_view_is_naked"] is True

    def test_naked_last_unaffected_when_naked_first_false(self):
        """Original naked_always=True, naked_first=False: naked still goes last."""
        backend = _ControlledBackend([f"g{i}" for i in range(50)])
        trainer = _make_trainer(backend)
        sample = _make_sample()
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(max_members=2, naked_always=True, naked_first=False),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert results[-1].metadata.get("_view_name") == "naked"


# ── TestForcedAnswer ─────────────────────────────────────────────────────────


class TestForcedAnswer:
    """Tests for forced_answer_on_last_member: inject GT on last member when all fail."""

    def _esc_cfg_forced(
        self, inject_chars: int = -1, end_prob: float = 1.0, max_members: int = 2
    ) -> EscalationConfig:
        return EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(
                max_members=max_members,
                naked_always=False,
                forced_answer_on_last_member=True,
                forced_answer_budget_mult=2.0,
                forced_answer_end_prob=end_prob,
                forced_answer_inject_chars=inject_chars,
                reward_threshold=1.0,
            ),
        )

    def test_config_defaults(self):
        h = HorizontalConfig()
        assert h.forced_answer_on_last_member is False
        assert h.forced_answer_budget_mult == pytest.approx(3.0)
        assert h.forced_answer_end_prob == pytest.approx(0.2)
        assert h.forced_answer_inject_chars == -1

    def test_config_parsed_from_yaml(self):
        cfg = build_escalation_config(
            {
                "horizontal": {
                    "forced_answer_on_last_member": True,
                    "forced_answer_end_prob": 0.3,
                    "forced_answer_inject_chars": 5,
                }
            }
        )
        assert cfg.horizontal.forced_answer_on_last_member is True
        assert cfg.horizontal.forced_answer_end_prob == pytest.approx(0.3)
        assert cfg.horizontal.forced_answer_inject_chars == 5

    def test_extract_forced_answer_from_boxed(self):
        from mlx_grpo.training.escalation_plan import _extract_forced_answer

        assert _extract_forced_answer("\\boxed{42}") == "42"
        assert _extract_forced_answer("some text \\boxed{xyz} more") == "xyz"

    def test_extract_forced_answer_strips_thinking(self):
        from mlx_grpo.training.escalation_plan import _extract_forced_answer

        assert _extract_forced_answer("<think>ignore</think>42") == "42"

    def test_extract_forced_answer_raw_fallback(self):
        from mlx_grpo.training.escalation_plan import _extract_forced_answer

        assert _extract_forced_answer("just text") == "just text"

    def test_extract_forced_answer_empty(self):
        from mlx_grpo.training.escalation_plan import _extract_forced_answer

        assert _extract_forced_answer("") == ""

    def test_forced_answer_forces_reward_to_one(self):
        """Last member (all fail, end_prob=1.0): reward overridden to 1.0."""
        sample = _make_sample(answer="<think>step.\n</think>\\boxed{42}")
        backend = _ControlledBackend([f"g{i}" for i in range(30)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)  # all members fail
        esc_cfg = self._esc_cfg_forced(end_prob=1.0, max_members=2)
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        last = results[-1]
        assert last.metadata.get("_esc_forced_answer") is True
        assert last.metadata["_esc_reward"] == pytest.approx(1.0)

    def test_forced_answer_not_fired_when_early_stop(self):
        """Prior member passes (early_stop=True): forced_answer never fires."""
        sample = _make_sample(answer="<think>step.\n</think>\\boxed{42}")
        backend = _ControlledBackend([f"g{i}" for i in range(30)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(1.0)  # first member passes → early_stop, loop breaks
        esc_cfg = self._esc_cfg_forced(end_prob=1.0, max_members=3)
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert not any(r.metadata.get("_esc_forced_answer") for r in results)

    def test_forced_answer_inject_chars_minus_one_means_full(self):
        """inject_chars=-1 → full extracted answer used."""
        from mlx_grpo.training.escalation_plan import _extract_forced_answer

        gt = "abcdef"
        extracted = _extract_forced_answer(f"\\boxed{{{gt}}}")
        assert extracted == gt  # -1 → no slice, full answer

    def test_forced_answer_inject_chars_truncates(self):
        """inject_chars=3 → only first 3 characters of GT injected."""
        from mlx_grpo.training.escalation_plan import _extract_forced_answer

        gt = "abcdef"
        n_chars = 3
        extracted = _extract_forced_answer(f"\\boxed{{{gt}}}")
        truncated = extracted[:n_chars]
        assert truncated == "abc"
        assert len(truncated) == n_chars

    def test_forced_answer_naked_followup_config_default(self):
        """forced_answer_naked_followup defaults to False (non-breaking)."""
        assert HorizontalConfig().forced_answer_naked_followup is False

    def test_forced_answer_naked_followup_parsed_from_yaml(self):
        cfg = build_escalation_config({"horizontal": {"forced_answer_naked_followup": True}})
        assert cfg.horizontal.forced_answer_naked_followup is True

    def test_forced_answer_naked_followup_appended(self):
        """When forced fires and naked_followup=True, a naked_followup rollout is appended."""
        sample = _make_sample(answer="<think>step.\n</think>\\boxed{42}")
        backend = _ControlledBackend([f"g{i}" for i in range(40)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(
                max_members=2,
                naked_always=False,
                forced_answer_on_last_member=True,
                forced_answer_end_prob=1.0,
                forced_answer_naked_followup=True,
                reward_threshold=1.0,
            ),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        followup = [r for r in results if r.metadata.get("_view_name") == "naked_followup"]
        assert len(followup) == 1
        assert followup[0].metadata.get("_esc_forced_followup") is True
        assert followup[0].metadata.get("_view_is_naked") is True

    def test_forced_answer_naked_followup_not_appended_when_disabled(self):
        """When naked_followup=False (default), no naked_followup rollout."""
        sample = _make_sample(answer="<think>step.\n</think>\\boxed{42}")
        backend = _ControlledBackend([f"g{i}" for i in range(30)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(0.0)
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(
                max_members=2,
                naked_always=False,
                forced_answer_on_last_member=True,
                forced_answer_end_prob=1.0,
                forced_answer_naked_followup=False,
                reward_threshold=1.0,
            ),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert not any(r.metadata.get("_view_name") == "naked_followup" for r in results)

    def test_forced_answer_naked_followup_not_appended_when_forced_not_fired(self):
        """naked_followup only appends when forced_answer actually injected."""
        sample = _make_sample(answer="<think>step.\n</think>\\boxed{42}")
        backend = _ControlledBackend([f"g{i}" for i in range(30)])
        trainer = _make_trainer(backend)
        reward_fn = _make_reward_fn(1.0)  # early_stop fires before last member
        esc_cfg = EscalationConfig(
            enabled=True,
            horizontal=HorizontalConfig(
                max_members=3,
                naked_always=False,
                forced_answer_on_last_member=True,
                forced_answer_end_prob=1.0,
                forced_answer_naked_followup=True,
                reward_threshold=1.0,
            ),
        )
        results = run_escalation(
            trainer=trainer,
            sample=sample,
            task_type=MagicMock(),
            merged_type_config={"thinking_budget": 16, "answer_max_tokens": 4},
            reward_fn=reward_fn,
            prompt_tokens=backend.encode("p"),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=64,
            esc_cfg=esc_cfg,
        )
        assert not any(r.metadata.get("_view_name") == "naked_followup" for r in results)

    def test_forced_probability_decay_schedule(self):
        """forced_p decays linearly from 1.0 → end_prob over training."""
        end_prob = 0.2
        # iter=0 → progress=0 → forced_p=1.0
        p0 = 1.0 - 0.0 * (1.0 - end_prob)
        assert p0 == pytest.approx(1.0)
        # iter=5/10 → progress=0.5 → forced_p=0.6
        p5 = 1.0 - 0.5 * (1.0 - end_prob)
        assert p5 == pytest.approx(0.6)
        # iter=10/10 → progress=1.0 → forced_p=end_prob
        p10 = 1.0 - 1.0 * (1.0 - end_prob)
        assert p10 == pytest.approx(end_prob)
        # monotonically non-increasing
        assert p0 >= p5
        assert p10 == pytest.approx(end_prob)


def _make_rollout_v2(raw, c_k, is_naked=False):
    r = MagicMock()
    r.metadata = {
        "_esc_reward": raw,
        "_esc_allocated_chunks": c_k,
        "_view_is_naked": is_naked,
    }
    return r


class TestGroupShapingV2:
    def _cfg(self, **kw):
        return RewardShapingConfig(**kw)

    def test_v2_disabled_by_default(self):
        assert RewardShapingConfig().reward_shaping_v2 is False

    def test_v1_reduces_correct_below_raw(self):
        cfg = self._cfg(reward_shaping_v2=False, lambda_s_max=0.02)
        group = [_make_rollout_v2(1.0, c_k=10)]
        _apply_group_shaping(group, cfg, c_max=10, iteration=100, total_iters=100)
        assert group[0].metadata["_esc_shaped_reward"] < 1.0

    def test_v2_correct_never_below_raw(self):
        cfg = self._cfg(reward_shaping_v2=True, lambda_s_max=0.05)
        for c_k in [1, 5, 10]:
            group = [_make_rollout_v2(1.0, c_k=c_k)]
            _apply_group_shaping(group, cfg, c_max=10, iteration=50, total_iters=100)
            shaped = group[0].metadata["_esc_shaped_reward"]
            assert shaped >= 1.0, f"c_k={c_k}: shaped={shaped} < raw=1.0"

    def test_v2_less_scaffold_gets_higher_shaped(self):
        cfg = self._cfg(reward_shaping_v2=True, lambda_s_max=0.05)
        r_low = _make_rollout_v2(0.8, c_k=1)
        r_high = _make_rollout_v2(0.8, c_k=9)
        # put both in one group so any_correct triggers for both
        _apply_group_shaping([r_low, r_high], cfg, c_max=10, iteration=50, total_iters=100)
        assert r_low.metadata["_esc_shaped_reward"] > r_high.metadata["_esc_shaped_reward"]

    def test_v2_max_scaffold_shaped_equals_raw(self):
        cfg = self._cfg(reward_shaping_v2=True, lambda_s_max=0.05)
        group = [_make_rollout_v2(0.8, c_k=10)]
        _apply_group_shaping(group, cfg, c_max=10, iteration=50, total_iters=100)
        assert abs(group[0].metadata["_esc_shaped_reward"] - 0.8) < 1e-9

    def test_v2_raw_1_shaped_equals_1(self):
        cfg = self._cfg(reward_shaping_v2=True, lambda_s_max=0.05)
        group = [_make_rollout_v2(1.0, c_k=3)]
        _apply_group_shaping(group, cfg, c_max=10, iteration=50, total_iters=100)
        assert abs(group[0].metadata["_esc_shaped_reward"] - 1.0) < 1e-9

    def test_v2_lambda_zero_shaped_equals_raw(self):
        cfg = self._cfg(reward_shaping_v2=True, lambda_s_max=0.0)
        group = [_make_rollout_v2(0.7, c_k=5)]
        _apply_group_shaping(group, cfg, c_max=10, iteration=50, total_iters=100)
        assert abs(group[0].metadata["_esc_shaped_reward"] - 0.7) < 1e-9

    def test_v2_naked_member_unchanged(self):
        cfg = self._cfg(reward_shaping_v2=True, lambda_s_max=0.05)
        r_naked = _make_rollout_v2(1.0, c_k=0, is_naked=True)
        r_scaffolded = _make_rollout_v2(1.0, c_k=5)
        _apply_group_shaping([r_naked, r_scaffolded], cfg, c_max=10, iteration=50, total_iters=100)
        # naked: shaped = raw (unmodified)
        assert r_naked.metadata["_esc_shaped_reward"] == 1.0

    def test_v2_wrong_in_success_group_stays_zero(self):
        cfg = self._cfg(reward_shaping_v2=True, lambda_s_max=0.05)
        r_correct = _make_rollout_v2(1.0, c_k=5)
        r_wrong = _make_rollout_v2(0.0, c_k=3)
        _apply_group_shaping([r_correct, r_wrong], cfg, c_max=10, iteration=50, total_iters=100)
        assert abs(r_wrong.metadata["_esc_shaped_reward"]) < 1e-9

        end_prob = 0.2
        # iter=0 → progress=0 → forced_p=1.0
        p0 = 1.0 - 0.0 * (1.0 - end_prob)
        assert p0 == pytest.approx(1.0)
        # iter=5/10 → progress=0.5 → forced_p=0.6
        p5 = 1.0 - 0.5 * (1.0 - end_prob)
        assert p5 == pytest.approx(0.6)
        # iter=10/10 → progress=1.0 → forced_p=end_prob
        p10 = 1.0 - 1.0 * (1.0 - end_prob)
        assert p10 == pytest.approx(end_prob)
        # monotonically non-increasing
        assert p0 >= p5
        assert p10 == pytest.approx(end_prob)


# ── TestAnyCorrectThreshold ───────────────────────────────────────────────────


def _make_rollout_simple(raw: float, c_k: int, is_naked: bool = False) -> MagicMock:
    r = MagicMock()
    r.metadata = {
        "_esc_reward": raw,
        "_esc_allocated_chunks": c_k,
        "_view_is_naked": is_naked,
    }
    return r


class TestAnyCorrectThreshold:
    def test_default_threshold_is_0_5(self):
        """RewardShapingConfig.any_correct_threshold defaults to 0.5."""
        rs = RewardShapingConfig()
        assert rs.any_correct_threshold == pytest.approx(0.5)

    def test_partial_exam_below_threshold_not_correct(self):
        """rewards=[0.0, 0.0, 0.22, 0.0], threshold=0.5 → any_correct=False → failure path.

        With failure_floor=0.0 (default), all non-naked shaped=0.0.
        """
        rs = RewardShapingConfig(any_correct_threshold=0.5, failure_floor=0.0)
        rollouts = [
            _make_rollout_simple(0.0, 1),
            _make_rollout_simple(0.0, 2),
            _make_rollout_simple(0.22, 3),
            _make_rollout_simple(0.0, 0, is_naked=True),
        ]
        _apply_group_shaping(rollouts, rs, c_max=4, iteration=0, total_iters=10)
        for r in rollouts:
            if not r.metadata["_view_is_naked"]:
                assert r.metadata["_esc_shaped_reward"] == pytest.approx(0.0)

    def test_reward_at_threshold_triggers_success(self):
        """rewards=[0.0, 0.5], threshold=0.5 → any_correct=True (>= not >)."""
        rs = RewardShapingConfig(any_correct_threshold=0.5, lambda_s_max=0.0)
        r_pass = _make_rollout_simple(0.5, 2)
        r_fail = _make_rollout_simple(0.0, 1)
        naked = _make_rollout_simple(0.0, 0, is_naked=True)
        _apply_group_shaping([r_pass, r_fail, naked], rs, c_max=4, iteration=0, total_iters=10)
        # success path: raw * (1 - 0 * c_k/cm) = raw
        assert r_pass.metadata["_esc_shaped_reward"] == pytest.approx(0.5)
        # wrong member on success path: 0.0 * (1 - ...) = 0.0
        assert r_fail.metadata["_esc_shaped_reward"] == pytest.approx(0.0)

    def test_threshold_zero_matches_old_behaviour(self):
        """threshold=0.0001, rewards=[0.0, 0.001] → any_correct=True (old behaviour)."""
        rs = RewardShapingConfig(any_correct_threshold=0.0001, lambda_s_max=0.0)
        r = _make_rollout_simple(0.001, 2)
        naked = _make_rollout_simple(0.0, 0, is_naked=True)
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=0, total_iters=10)
        # 0.001 >= 0.0001 → success path
        assert r.metadata["_esc_shaped_reward"] == pytest.approx(0.001)

    def test_all_zero_always_failure(self):
        """rewards=[0.0, 0.0, 0.0] → failure path regardless of threshold."""
        rs = RewardShapingConfig(any_correct_threshold=0.5, failure_floor=0.0)
        rollouts = [_make_rollout_simple(0.0, i + 1) for i in range(3)]
        _apply_group_shaping(rollouts, rs, c_max=4, iteration=0, total_iters=10)
        for r in rollouts:
            assert r.metadata["_esc_shaped_reward"] == pytest.approx(0.0)


# ── TestFailureFloor ──────────────────────────────────────────────────────────


class TestFailureFloor:
    def test_default_floor_zero(self):
        """RewardShapingConfig.failure_floor defaults to 0.0."""
        rs = RewardShapingConfig()
        assert rs.failure_floor == pytest.approx(0.0)

    def test_default_floor_wrong_answers_all_zero(self):
        """all rewards=0, lambda_f_max=0.02, iteration=10 → all shaped=0.0 (floor=0)."""
        rs = RewardShapingConfig(lambda_f_max=0.02, failure_floor=0.0)
        rollouts = [_make_rollout_simple(0.0, c) for c in [1, 2, 3]]
        naked = _make_rollout_simple(0.0, 0, is_naked=True)
        _apply_group_shaping(rollouts + [naked], rs, c_max=4, iteration=10, total_iters=10)
        for r in rollouts:
            assert r.metadata["_esc_shaped_reward"] == pytest.approx(0.0)

    def test_explicit_floor_enables_nudge(self):
        """failure_floor=0.05, lambda_f_max=0.10, c_k=10, c_max=10, iteration=0.

        lambda_f = 0.10 * (1-0) = 0.10; 0.10*10/10 = 0.10; min(0.10, 0.05) = 0.05.
        """
        rs = RewardShapingConfig(lambda_f_max=0.10, failure_floor=0.05)
        r = _make_rollout_simple(0.0, 10)
        naked = _make_rollout_simple(0.0, 0, is_naked=True)
        _apply_group_shaping([r, naked], rs, c_max=10, iteration=0, total_iters=10)
        assert r.metadata["_esc_shaped_reward"] == pytest.approx(0.05)

    def test_floor_caps_at_floor_value(self):
        """failure_floor=0.005, large lambda_f → shaped <= 0.005."""
        rs = RewardShapingConfig(lambda_f_max=1.0, failure_floor=0.005)
        r = _make_rollout_simple(0.0, 4)
        naked = _make_rollout_simple(0.0, 0, is_naked=True)
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=0, total_iters=10)
        assert r.metadata["_esc_shaped_reward"] <= 0.005 + 1e-9

    def test_naked_in_failure_group_stays_zero(self):
        """naked member raw=0.0 → shaped=raw=0.0 (naked branch, not failure branch)."""
        rs = RewardShapingConfig(lambda_f_max=0.02, failure_floor=0.02)
        r = _make_rollout_simple(0.0, 2)
        naked = _make_rollout_simple(0.0, 0, is_naked=True)
        _apply_group_shaping([r, naked], rs, c_max=4, iteration=0, total_iters=10)
        assert naked.metadata["_esc_shaped_reward"] == pytest.approx(0.0)
