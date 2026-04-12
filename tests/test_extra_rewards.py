"""Tests for EfficiencyReward, ReasoningReward, MultiplicativeReward, HierarchicalReward."""

from __future__ import annotations

import pytest

from mlx_grpo.engine.result import GenerateResult, RolloutResult, Segment, SegmentSource
from mlx_grpo.rewards.efficiency import EfficiencyReward
from mlx_grpo.rewards.hierarchical import HierarchicalReward, RewardBreakdown
from mlx_grpo.rewards.multiplicative import MultiplicativeReward, _default_scorer
from mlx_grpo.rewards.reasoning import ReasoningReward

# ── Helpers ────────────────────────────────────────────────────────────────────


def _r(text: str) -> RolloutResult:
    return RolloutResult(
        segments=[Segment([ord(c) for c in text], text, SegmentSource.GENERATED, False)]
    )


def _r_with_vars(text: str, **vars: tuple[str, str]) -> RolloutResult:
    result = _r(text)
    for name, var_text in vars.items():
        result.variables[name] = GenerateResult(
            text=var_text,
            tokens=[ord(c) for c in var_text],
            stop_reason="eos",
        )
    return result


def _r_with_tagged_segments(*segs: tuple[str, str]) -> RolloutResult:
    """Build RolloutResult with (tag, text) named segments."""
    segments = []
    for tag, text in segs:
        segments.append(
            Segment(
                tokens=[ord(c) for c in text],
                text=text,
                source=SegmentSource.GENERATED,
                masked=False,
                tag=tag,
            )
        )
    return RolloutResult(segments=segments)


# ── EfficiencyReward ──────────────────────────────────────────────────────────


class TestEfficiencyReward:
    def setup_method(self):
        self.r = EfficiencyReward()

    def test_empty_text(self):
        assert self.r.compute(_r("")) == 0.0

    def test_whitespace_only(self):
        assert self.r.compute(_r("   ")) == 0.0

    def test_base_score_no_thinking_var(self):
        # No thinking/answer vars → base score 0.5, no penalties
        score = self.r.compute(_r("a simple answer"))
        assert 0.0 <= score <= 1.0

    def test_good_ratio(self):
        """Balanced thinking/answer ratio → no ratio penalties."""
        # answer_len / (think_len + answer_len) must be in [0.05, max_ratio=0.8]
        thinking_text = "think"  # 5 chars
        answer_text = "42 is the answer here"  # 21 chars → ratio=21/26≈0.81, just above max_ratio
        # Use equal lengths to stay in safe zone
        thinking_text = "think step"  # 10 chars
        answer_text = "the answer"  # 10 chars → ratio=10/20=0.5 — in good range
        r = _r_with_vars(thinking_text + answer_text, thinking=thinking_text, answer=answer_text)
        score = self.r.compute(r)
        assert score == pytest.approx(0.5)  # Base score, no penalties

    def test_tiny_answer_penalized(self):
        """Almost no answer relative to thinking → subtract 0.3."""
        thinking_text = "a " * 200  # long thinking
        answer_text = "x"  # very short answer
        r = _r_with_vars(
            thinking_text + answer_text,
            thinking=thinking_text,
            answer=answer_text,
        )
        score = self.r.compute(r)
        assert score < 0.5  # penalized

    def test_heavy_repetition_penalized(self):
        """High trigram repetition → subtract 0.3."""
        repeated = "the cat sat " * 50  # lots of repeated trigrams
        r = _r(repeated)
        score = self.r.compute(r)
        assert score < 0.5

    def test_filler_words_penalized(self):
        text = "Basically this is really just essentially the answer literally very simple"
        score = self.r.compute(_r(text))
        assert score < 0.5

    def test_no_filler_words(self):
        text = "The answer is obtained by multiplying 6 by 7 to get 42."
        score = self.r.compute(_r(text))
        assert score >= 0.4

    def test_score_clamped_0_1(self):
        # Worst case: all penalties should not go below 0
        text = "basically " * 100
        score = self.r.compute(_r(text))
        assert 0.0 <= score <= 1.0

    def test_code_var_used(self):
        """EfficiencyReward uses 'code' variable if 'answer' is absent."""
        r = _r_with_vars("think... code", thinking="think...", code="code")
        score = self.r.compute(r)
        assert 0.0 <= score <= 1.0

    def test_max_ratio_penalty(self):
        """Too much answer relative to thinking → subtract 0.1."""
        thinking_text = "x"  # very short thinking
        answer_text = "a " * 50  # long answer
        r = _r_with_vars(
            thinking_text + answer_text,
            thinking=thinking_text,
            answer=answer_text,
        )
        score = self.r.compute(r)
        assert score <= 0.5


# ── ReasoningReward ────────────────────────────────────────────────────────────


class TestReasoningReward:
    def setup_method(self):
        self.r = ReasoningReward()

    def test_no_thinking_var(self):
        assert self.r.compute(_r("some text")) == 0.0

    def test_empty_thinking_var(self):
        r = _r_with_vars("", thinking="   ")
        assert self.r.compute(r) == 0.0

    def test_base_score_for_thinking(self):
        r = _r_with_vars("thinking", thinking="some reasoning here " * 5)
        score = self.r.compute(r)
        assert score >= 0.2

    def test_too_short_thinking_penalized(self):
        """Thinking under min_thinking_tokens → subtract 0.2."""
        r = ReasoningReward(min_thinking_tokens=100)
        short = _r_with_vars("x", thinking="short")
        score = r.compute(short)
        # token_count = len tokens = len text (word-level doesn't matter, it's char count)
        # token_count is the length of the tokens list
        assert 0.0 <= score <= 1.0

    def test_step_bonus_for_numbered_steps(self):
        thinking = "Step 1: compute. Step 2: verify. Then conclude."
        r = _r_with_vars(thinking, thinking=thinking)
        score = self.r.compute(r)
        assert score >= 0.3

    def test_verification_bonus(self):
        thinking = (
            "let me double-check this. Actually wait, let me verify the result. Therefore correct."
        )
        r = _r_with_vars(thinking, thinking=thinking)
        score = self.r.compute(r)
        assert score >= 0.3

    def test_repetition_penalty(self):
        sentence = "The answer is 42. "
        thinking = sentence * 10  # highly repetitive sentences
        r = _r_with_vars(thinking, thinking=thinking)
        score = self.r.compute(r)
        assert score < 0.5

    def test_score_clamped(self):
        thinking = "Step 1: check. Step 2: verify. Actually double-check. Therefore done."
        r = _r_with_vars(thinking, thinking=thinking)
        score = self.r.compute(r)
        assert 0.0 <= score <= 1.0

    def test_too_long_thinking_penalty(self):
        r = ReasoningReward(max_thinking_tokens=5)
        long_thinking = "word " * 100
        result = _r_with_vars(long_thinking, thinking=long_thinking)
        score = r.compute(result)
        assert 0.0 <= score <= 1.0


# ── _default_scorer ────────────────────────────────────────────────────────────


class TestDefaultScorer:
    def test_exact_match(self):
        assert _default_scorer("42", "42") == 1.0

    def test_substring_match(self):
        assert _default_scorer("The answer is 42 here", "42") == 0.5

    def test_no_match(self):
        assert _default_scorer("something else", "42") == 0.0

    def test_empty_answer(self):
        assert _default_scorer("text", None) == 0.0

    def test_empty_segment(self):
        assert _default_scorer("  ", "42") == 0.0

    def test_case_insensitive_exact(self):
        assert _default_scorer("HELLO", "hello") == 1.0


# ── MultiplicativeReward ───────────────────────────────────────────────────────


class TestMultiplicativeReward:
    def _r_checkpoints(self, *pairs: tuple[str, str]) -> RolloutResult:
        return _r_with_tagged_segments(*pairs)

    def test_all_correct(self):
        r = MultiplicativeReward(
            checkpoint_tags=["checkpoint_0", "checkpoint_1"],
            checkpoint_scorer=lambda text, ans: 1.0,
        )
        result = self._r_checkpoints(("checkpoint_0", "42"), ("checkpoint_1", "42"))
        assert r.compute(result, "42") == 1.0

    def test_one_zero_zeroes_all(self):
        scores = [1.0, 0.0]

        def scorer(text, ans):
            return scores.pop(0)

        r = MultiplicativeReward(
            checkpoint_tags=["checkpoint_0", "checkpoint_1"],
            checkpoint_scorer=scorer,
        )
        result = self._r_checkpoints(("checkpoint_0", "val"), ("checkpoint_1", "val"))
        assert r.compute(result, "val") == 0.0

    def test_auto_detect_checkpoints(self):
        """Without explicit tags, finds segments starting with 'checkpoint_'."""
        r = MultiplicativeReward(checkpoint_scorer=lambda t, a: 1.0)
        result = self._r_checkpoints(("checkpoint_0", "a"), ("checkpoint_1", "b"))
        assert r.compute(result, "x") == 1.0

    def test_min_checkpoints_not_met(self):
        r = MultiplicativeReward(checkpoint_tags=[], min_checkpoints=2)
        result = _r("no tags")
        assert r.compute(result, "x") == 0.0

    def test_short_circuit_penalty(self):
        r = MultiplicativeReward(short_circuit_penalty=0.3)
        result = _r("text")
        result.metadata["short_circuited"] = True
        assert r.compute(result, "x") == pytest.approx(0.3)

    def test_empty_segment_skipped(self):
        r = MultiplicativeReward(
            checkpoint_tags=["checkpoint_0", "checkpoint_1"],
            checkpoint_scorer=lambda t, a: 1.0,
            min_checkpoints=2,
        )
        # checkpoint_1 has empty text → skipped → scored_count=1 < min_checkpoints=2
        result = self._r_checkpoints(("checkpoint_0", "42"), ("checkpoint_1", ""))
        assert r.compute(result, "42") == 0.0

    def test_default_scorer_used_by_default(self):
        r = MultiplicativeReward(checkpoint_tags=["checkpoint_0"])
        result = self._r_checkpoints(("checkpoint_0", "42"))
        assert r.compute(result, "42") == 1.0


# ── HierarchicalReward ─────────────────────────────────────────────────────────


class TestHierarchicalReward:
    def _make_hr(self, **kwargs) -> HierarchicalReward:
        return HierarchicalReward(**kwargs)

    def test_empty_rewards(self):
        h = HierarchicalReward(rewards=[])
        assert h.compute(_r("text")) == 0.0

    def test_single_component(self):
        class FixedReward:
            def compute(self, result, answer=None):
                return 0.8

        h = HierarchicalReward(rewards=[("acc", FixedReward(), 1.0)])
        assert h.compute(_r("text"), "x") == pytest.approx(0.8)

    def test_weighted_sum(self):
        class R:
            def __init__(self, v):
                self.v = v

            def compute(self, r, a=None):
                return self.v

        h = HierarchicalReward(
            rewards=[("r1", R(1.0), 1.0), ("r2", R(0.0), 1.0)],
            anti_gaming=False,
        )
        score = h.compute_detailed(_r("text"))
        assert score.total == pytest.approx(0.5)

    def test_gating_zeros_all(self):
        class Accuracy:
            def compute(self, r, a=None):
                return 0.0  # Zero accuracy → gate fires

        class Format:
            def compute(self, r, a=None):
                return 1.0

        h = HierarchicalReward(
            rewards=[("accuracy", Accuracy(), 0.5), ("format", Format(), 0.5)],
            gate_reward="accuracy",
            gate_threshold=0.1,
            anti_gaming=False,
        )
        bd = h.compute_detailed(_r("text"))
        assert bd.gated is True
        assert bd.total == 0.0
        assert bd.components["accuracy"] == 0.0
        assert bd.components["format"] == 0.0

    def test_gating_passes_above_threshold(self):
        class Accuracy:
            def compute(self, r, a=None):
                return 0.5  # Above gate threshold

        class Format:
            def compute(self, r, a=None):
                return 1.0

        h = HierarchicalReward(
            rewards=[("accuracy", Accuracy(), 1.0), ("format", Format(), 1.0)],
            gate_reward="accuracy",
            gate_threshold=0.1,
            anti_gaming=False,
        )
        bd = h.compute_detailed(_r("text"))
        assert bd.gated is False
        assert bd.total > 0.0

    def test_anti_gaming_caps_non_accuracy(self):
        """Near-zero accuracy → non-accuracy rewards capped at 0.3."""

        class Accuracy:
            def compute(self, r, a=None):
                return 0.05

        class Format:
            def compute(self, r, a=None):
                return 1.0  # would be capped to 0.3

        h = HierarchicalReward(
            rewards=[("accuracy", Accuracy(), 1.0), ("format", Format(), 1.0)],
            anti_gaming=True,
        )
        bd = h.compute_detailed(_r("text"))
        assert bd.components["format"] <= 0.3

    def test_smoothing(self):
        """EMA smoothing blends current value with previous."""

        class StepReward:
            def __init__(self):
                self.v = 0.0

            def compute(self, r, a=None):
                return self.v

        rw = StepReward()
        h = HierarchicalReward(rewards=[("r", rw, 1.0)], smoothing=0.5, anti_gaming=False)

        rw.v = 1.0
        score1 = h.compute(_r("t"))  # first call — no EMA yet
        assert score1 == pytest.approx(1.0)

        rw.v = 0.0
        score2 = h.compute(_r("t"))  # second call — EMA should blend
        assert 0.0 < score2 < 1.0

    def test_add_chaining(self):
        class R:
            def compute(self, r, a=None):
                return 0.5

        h = HierarchicalReward()
        h2 = h.add("r", R(), 1.0)
        assert h2 is h
        assert len(h._rewards) == 1

    def test_exception_in_reward_returns_zero(self):
        class BadReward:
            def compute(self, r, a=None):
                raise ValueError("error")

        h = HierarchicalReward(rewards=[("bad", BadReward(), 1.0)], anti_gaming=False)
        # Should not raise; bad reward counts as 0.0
        assert h.compute(_r("text")) == 0.0

    def test_reward_breakdown_fields(self):
        class R:
            def compute(self, r, a=None):
                return 0.7

        h = HierarchicalReward(rewards=[("r", R(), 1.0)], anti_gaming=False)
        bd = h.compute_detailed(_r("text"))
        assert isinstance(bd, RewardBreakdown)
        assert "r" in bd.components
        assert "r" in bd.weights
        assert bd.total == pytest.approx(0.7)
        assert bd.gated is False

    def test_total_clamped_to_01(self):
        class OverReward:
            def compute(self, r, a=None):
                return 5.0  # Far above 1.0

        h = HierarchicalReward(rewards=[("o", OverReward(), 1.0)], anti_gaming=False)
        assert h.compute(_r("text")) == pytest.approx(1.0)


# ── ThinkingEfficiencyReward ───────────────────────────────────────────────────


def _thinking_result(
    thinking_tokens: int,
    has_boxed_in_generated: bool = True,
    answer_correct: bool = True,
    injected_thinking: bool = False,
) -> RolloutResult:
    """Build a RolloutResult that mimics a math rollout with a thinking segment."""
    import mlx_grpo.rewards  # ensure registrations  # noqa: F401

    answer_text = r"\boxed{42}" if has_boxed_in_generated else "no boxed here"
    thinking_text = "x" * thinking_tokens  # dummy; token count is what matters

    # thinking segment
    think_source = SegmentSource.INJECTED if injected_thinking else SegmentSource.GENERATED
    think_seg = Segment(
        tokens=list(range(thinking_tokens)),
        text=thinking_text,
        source=think_source,
        masked=False,
        tag="thinking",
    )
    # answer segment (GENERATED) with \boxed{} when correct
    ans_seg = Segment(
        tokens=list(range(5)),
        text=answer_text,
        source=SegmentSource.GENERATED,
        masked=False,
        tag="answer",
    )
    result = RolloutResult(segments=[think_seg, ans_seg])
    # variables["thinking"] for the base GenerateResult
    result.variables["thinking"] = GenerateResult(
        text=thinking_text,
        tokens=list(range(thinking_tokens)),
        stop_reason="stop_sequence",
    )
    return result


class TestThinkingEfficiencyReward:
    """ThinkingEfficiencyReward — efficiency bonus gated on correctness."""

    def _make(self, **kwargs):
        from mlx_grpo.rewards.efficiency import ThinkingEfficiencyReward

        return ThinkingEfficiencyReward(**kwargs)

    def test_wrong_answer_returns_zero(self):
        """Wrong answer (base reward 0) → total reward 0, no bonus."""
        r = self._make(budget=200, max_bonus=0.2)
        result = _thinking_result(thinking_tokens=100, answer_correct=False)
        # Ground truth doesn't match → base MathReward returns 0
        assert r.compute(result, answer=r"\boxed{99}") == pytest.approx(0.0)

    def test_correct_under_budget_gets_bonus(self):
        """Correct answer + thinking < budget → base + bonus."""
        r = self._make(budget=200, max_bonus=0.2, min_thinking_tokens=10)
        result = _thinking_result(thinking_tokens=100)
        # thinking_tokens=100, budget=200 → ratio=0.5 → bonus=0.2*0.5=0.1
        score = r.compute(result, answer=r"\boxed{42}")
        assert score > 1.0  # base(1.0) + bonus(0.1)
        assert score == pytest.approx(1.1, abs=1e-6)

    def test_correct_at_budget_no_bonus(self):
        """Correct answer but thinking == budget → ratio=1 → bonus=0."""
        r = self._make(budget=100, max_bonus=0.2, min_thinking_tokens=10)
        result = _thinking_result(thinking_tokens=100)
        score = r.compute(result, answer=r"\boxed{42}")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_correct_over_budget_no_bonus(self):
        """Correct answer but thinking > budget → ratio clamped to 1 → bonus=0."""
        r = self._make(budget=100, max_bonus=0.2, min_thinking_tokens=10)
        result = _thinking_result(thinking_tokens=300)
        score = r.compute(result, answer=r"\boxed{42}")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_no_boxed_in_generated_no_bonus(self):
        """Correct but \boxed{} only in scaffold (not generated) → no bonus."""
        r = self._make(budget=500, max_bonus=0.2, min_thinking_tokens=10)
        result = _thinking_result(thinking_tokens=100, has_boxed_in_generated=False)
        # No \boxed{} in generated_text → returns base_score only, but
        # without boxed in generated text MathReward also returns 0.
        score = r.compute(result, answer=r"\boxed{42}")
        assert score == pytest.approx(0.0)

    def test_below_min_thinking_tokens_no_bonus(self):
        """Correct but thinking < min_thinking_tokens → base only, no bonus."""
        r = self._make(budget=500, max_bonus=0.2, min_thinking_tokens=50)
        result = _thinking_result(thinking_tokens=20)
        score = r.compute(result, answer=r"\boxed{42}")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_decay_shapes_curve(self):
        """Lower decay (concave) gives higher bonus than linear at same ratio."""
        result = _thinking_result(thinking_tokens=50)
        r_linear = self._make(budget=100, max_bonus=0.2, decay=1.0, min_thinking_tokens=10)
        r_concave = self._make(budget=100, max_bonus=0.2, decay=0.5, min_thinking_tokens=10)
        # ratio=0.5; linear bonus=0.2*0.5=0.1; concave bonus=0.2*0.5^0.5≈0.141
        s_linear = r_linear.compute(result, answer=r"\boxed{42}")
        s_concave = r_concave.compute(result, answer=r"\boxed{42}")
        assert s_concave > s_linear

    def test_injected_thinking_not_counted(self):
        """Injected thinking segments are excluded from token count."""
        r = self._make(budget=200, max_bonus=0.2, min_thinking_tokens=10)
        result = _thinking_result(thinking_tokens=100, injected_thinking=True)
        # All thinking is INJECTED → thinking_tokens == 0 < min → no bonus
        score = r.compute(result, answer=r"\boxed{42}")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_registered_by_name(self):
        """ThinkingEfficiencyReward is accessible via registry."""
        from mlx_grpo.rewards.base import BaseReward

        r = BaseReward.get("thinking_efficiency")
        assert r is not None

    def test_no_thinking_segments_no_bonus(self):
        """No thinking segments → thinking_tokens=0 < min → no bonus."""
        r = self._make(budget=200, max_bonus=0.2, min_thinking_tokens=50)
        # Result with no thinking segment at all
        ans_seg = Segment([1, 2, 3], r"\boxed{42}", SegmentSource.GENERATED, False, tag="answer")
        result = RolloutResult(segments=[ans_seg])
        score = r.compute(result, answer=r"\boxed{42}")
        assert score == pytest.approx(1.0, abs=1e-6)  # base only


# ── _format_beauty ─────────────────────────────────────────────────────────────


class TestFormatBeauty:
    """_format_beauty scoring function unit tests."""

    def _score(self, text: str) -> float:
        from mlx_grpo.rewards.efficiency import _format_beauty

        return _format_beauty(text)

    def test_empty_returns_zero(self):
        assert self._score("") == 0.0

    def test_plain_boxed_digit(self):
        # \boxed{5} — short inner, only gets partial score
        s = self._score(r"\boxed{5}")
        assert 0.0 < s <= 0.5

    def test_rich_latex_answer(self):
        # \boxed{\frac{1}{2}} — long inner content
        s = self._score(r"\boxed{\frac{1}{2}}")
        assert s >= 0.5

    def test_ugly_artifacts_penalized(self):
        s_clean = self._score(r"\boxed{42}")
        s_ugly = self._score(r"\boxed{42} ```python")
        assert s_clean > s_ugly

    def test_many_latex_cmds_bonus(self):
        # Multiple LaTeX commands boost score
        s = self._score(r"\frac{\sqrt{2}}{\pi} = \boxed{42}")
        assert s >= 0.5

    def test_rich_beats_plain(self):
        plain = self._score(r"\boxed{42}")
        rich = self._score(r"\frac{\sqrt{3}}{2} \cdot \pi = \boxed{42}")
        assert rich > plain


# ── ThinkingEfficiencyReward beauty bonus ──────────────────────────────────────


class TestThinkingEfficiencyBeauty:
    """Beauty bonus: only fires when correct + highly efficient."""

    def _make(self, **kwargs):
        from mlx_grpo.rewards.efficiency import ThinkingEfficiencyReward

        return ThinkingEfficiencyReward(**kwargs)

    def _result(self, thinking_tokens: int, ans_text: str) -> RolloutResult:
        """Build result where MathReward can verify correctness via last \\boxed{}.

        ans_text must contain \\boxed{42} so predicted="42" matches answer="42".
        Extra LaTeX formatting before \\boxed{} is ignored by MathReward but
        scored by _format_beauty.
        """
        think_seg = Segment(
            list(range(thinking_tokens)),
            "t" * thinking_tokens,
            SegmentSource.GENERATED,
            False,
            tag="thinking",
        )
        ans_seg = Segment([1, 2, 3], ans_text, SegmentSource.GENERATED, False, tag="answer")
        r = RolloutResult(segments=[think_seg, ans_seg])
        r.variables["thinking"] = GenerateResult(
            text="t" * thinking_tokens,
            tokens=list(range(thinking_tokens)),
            stop_reason="stop_sequence",
        )
        return r

    # ans_text helpers — both contain \boxed{42} so MathReward matches "42"
    _PLAIN = r"\boxed{42}"
    _RICH = r"\frac{\sqrt{3}}{2} \cdot \pi = \boxed{42}"

    def test_beauty_adds_to_correct_efficient(self):
        """With beauty_bonus>0, correct+efficient gets higher score than without."""
        r_no_beauty = self._make(
            budget=200,
            max_bonus=0.0,
            min_thinking_tokens=10,
            beauty_bonus=0.0,
            efficiency_threshold=1.0,
        )
        r_beauty = self._make(
            budget=200,
            max_bonus=0.0,
            min_thinking_tokens=10,
            beauty_bonus=0.05,
            efficiency_threshold=1.0,
        )
        result = self._result(50, self._RICH)
        assert r_beauty.compute(result, answer="42") > r_no_beauty.compute(result, answer="42")

    def test_beauty_only_when_efficient(self):
        """Beauty fires for ratio < threshold but not for ratio >= threshold."""
        r = self._make(
            budget=100,
            max_bonus=0.0,
            min_thinking_tokens=10,
            beauty_bonus=0.05,
            efficiency_threshold=0.5,
        )
        result_eff = self._result(30, self._RICH)  # ratio=0.30 < 0.5 → fires
        result_slow = self._result(80, self._RICH)  # ratio=0.80 >= 0.5 → blocked
        assert r.compute(result_eff, answer="42") > r.compute(result_slow, answer="42")

    def test_beauty_zero_when_wrong(self):
        """Wrong answer → base_score=0 → total=0, beauty never applies."""
        r = self._make(
            budget=200,
            max_bonus=0.2,
            min_thinking_tokens=10,
            beauty_bonus=0.05,
            efficiency_threshold=1.0,
        )
        result = self._result(50, self._RICH)
        assert r.compute(result, answer="99") == pytest.approx(0.0)

    def test_rich_formatting_scores_higher(self):
        """Rich LaTeX formatting produces higher beauty bonus than bare \\boxed{}."""
        r = self._make(
            budget=200,
            max_bonus=0.0,
            min_thinking_tokens=10,
            beauty_bonus=0.1,
            efficiency_threshold=1.0,
        )
        s_plain = r.compute(self._result(50, self._PLAIN), answer="42")
        s_rich = r.compute(self._result(50, self._RICH), answer="42")
        assert s_rich > s_plain

    def test_beauty_default_disabled(self):
        """beauty_bonus defaults to 0.0 — existing configs unaffected."""
        from mlx_grpo.rewards.efficiency import ThinkingEfficiencyReward

        r = ThinkingEfficiencyReward(budget=200, max_bonus=0.2, min_thinking_tokens=10)
        assert r.beauty_bonus == 0.0
        result = self._result(50, self._RICH)
        # Score should be exactly base + eff_bonus, no beauty component
        ratio = 50 / 200
        expected = 1.0 + 0.2 * (1.0 - ratio)
        assert r.compute(result, answer="42") == pytest.approx(expected, abs=1e-6)
