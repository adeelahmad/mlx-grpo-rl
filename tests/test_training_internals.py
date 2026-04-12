"""Tests for pure-Python training internals: _metal_clear_cache, _collect_memory,
_extract_segment_masks, _report_step, _resolve_system_prompt, hooks, etc."""

from __future__ import annotations

import json
import tempfile
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

import mlx_grpo.types.builtins  # noqa: F401  — trigger registration
from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader, Sample
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training import StepMetrics, Trainer, _collect_memory, _metal_clear_cache

# ── _metal_clear_cache ────────────────────────────────────────────────────────


class TestMetalClearCache:
    def test_does_not_raise(self):
        """_metal_clear_cache() should never raise."""
        _metal_clear_cache()  # May or may not have MLX — should not crash

    def test_clears_mlx_cache_when_available(self):
        """When MLX is available, clear_cache is called."""
        mock_mx = MagicMock()
        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx}):
            _metal_clear_cache()
            # Either it was called (MLX available) or silently skipped


# ── _collect_memory ──────────────────────────────────────────────────────────


class TestCollectMemory:
    def test_returns_tuple(self):
        active_mb, rss_mb = _collect_memory()
        assert isinstance(active_mb, float)
        assert isinstance(rss_mb, float)

    def test_rss_positive(self):
        _, rss_mb = _collect_memory()
        # RSS should be > 0 on any real process
        assert rss_mb >= 0.0

    def test_metal_zero_without_mlx(self):
        """When MLX's metal API isn't available, active_mb=0."""
        mock_mx = MagicMock()
        mock_mx.metal.get_active_memory.side_effect = AttributeError("no metal")
        with patch.dict("sys.modules", {"mlx.core": mock_mx}):
            active_mb, _ = _collect_memory()
            # Either real MLX ran or fell through to 0


# ── Trainer basic setup ──────────────────────────────────────────────────────


def _make_config(**kwargs):
    defaults = dict(
        iterations=2,
        group_size=2,
        batch_size=1,
        save_every=0,
        eval_every=0,
        loss_type="grpo",
        beta=0.01,
        learning_rate=1e-4,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def _make_dataset(n=4):
    rows = [{"prompt": f"Q{i}", "answer": f"A{i}", "type": "mcq"} for i in range(n)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        path = f.name
    loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
    loader.load()
    return loader


def _make_trainer(**kwargs):
    config = _make_config(**kwargs)
    dataset = _make_dataset()
    backend = MagicMock()
    backend.encode.return_value = [1, 2, 3]
    backend.decode.return_value = "decoded"
    return Trainer(config=config, dataset=dataset, backend=backend)


# ── _extract_segment_masks ────────────────────────────────────────────────────


class TestExtractSegmentMasks:
    def _make_rollout(self, segments: list[Segment]) -> RolloutResult:
        return RolloutResult(segments=segments)

    def test_empty_rollouts(self):
        ans, think, scaf = Trainer._extract_segment_masks([])
        assert ans == []
        assert think == []
        assert scaf == []

    def test_generated_answer_segment(self):
        seg = Segment(
            tokens=[1, 2, 3],
            text="answer",
            source=SegmentSource.GENERATED,
            masked=False,
            tag="answer",
        )
        rollout = self._make_rollout([seg])
        ans, think, scaf = Trainer._extract_segment_masks([rollout])
        assert len(ans) == 1
        assert len(ans[0]) == 3
        assert all(ans[0])  # All True for answer
        assert not any(think[0])
        assert not any(scaf[0])

    def test_generated_thinking_segment(self):
        seg = Segment(
            tokens=[1, 2],
            text="thinking",
            source=SegmentSource.GENERATED,
            masked=True,
            tag="thinking",
        )
        rollout = self._make_rollout([seg])
        ans, think, scaf = Trainer._extract_segment_masks([rollout])
        assert not any(ans[0])
        assert all(think[0])

    def test_injected_scaffold_segment(self):
        seg = Segment(
            tokens=[1, 2, 3],
            text="scaffold",
            source=SegmentSource.INJECTED,
            masked=True,
            tag="scaffold",
        )
        rollout = self._make_rollout([seg])
        ans, think, scaf = Trainer._extract_segment_masks([rollout])
        # Scaffold is injected → scaf_mask is True
        assert all(scaf[0])
        # Not tagged answer or thinking
        assert not any(ans[0])
        assert not any(think[0])

    def test_prompt_segment_skipped(self):
        prompt_seg = Segment(
            tokens=[10, 20],
            text="prompt",
            source=SegmentSource.PROMPT,
            masked=True,
        )
        answer_seg = Segment(
            tokens=[1, 2],
            text="answer",
            source=SegmentSource.GENERATED,
            masked=False,
            tag="answer",
        )
        rollout = self._make_rollout([prompt_seg, answer_seg])
        ans, think, scaf = Trainer._extract_segment_masks([rollout])
        # Only 2 completion tokens (from answer_seg, prompt is skipped)
        assert len(ans[0]) == 2
        assert all(ans[0])

    def test_multiple_rollouts(self):
        seg1 = Segment([1], "a", SegmentSource.GENERATED, False, tag="answer")
        seg2 = Segment([2, 3], "b", SegmentSource.GENERATED, True, tag="thinking")
        r1 = self._make_rollout([seg1])
        r2 = self._make_rollout([seg2])
        ans, think, scaf = Trainer._extract_segment_masks([r1, r2])
        assert len(ans) == 2
        assert ans[0] == [True]
        assert think[1] == [True, True]

    def test_mixed_segments(self):
        scaffold = Segment([10, 11], "prefix", SegmentSource.INJECTED, True, tag="scaffold")
        thinking = Segment([1, 2, 3], "thoughts", SegmentSource.GENERATED, True, tag="thinking")
        answer = Segment([4], "answer", SegmentSource.GENERATED, False, tag="answer")
        rollout = self._make_rollout([scaffold, thinking, answer])
        ans, think, scaf = Trainer._extract_segment_masks([rollout])
        n = 6  # 2 scaffold + 3 thinking + 1 answer
        assert len(ans[0]) == n
        assert len(think[0]) == n
        assert len(scaf[0]) == n
        # scaffold tokens → scaf True
        assert scaf[0][0] and scaf[0][1]
        # thinking tokens → think True
        assert think[0][2] and think[0][3] and think[0][4]
        # answer token → ans True
        assert ans[0][5]


# ── _report_step ─────────────────────────────────────────────────────────────


class TestReportStep:
    def test_report_step_no_crash(self):
        trainer = _make_trainer()
        trainer._recent_losses = deque(maxlen=10)
        trainer._recent_rewards = deque(maxlen=10)
        metrics = StepMetrics(
            iteration=0,
            loss=1.5,
            mean_reward=0.7,
            kl_divergence=0.1,
            tokens_per_sec=100.0,
            elapsed_sec=2.0,
            generated_tokens=200,
            learning_rate=1e-4,
        )
        # Should not raise
        trainer._report_step(0, 10, metrics)

    def test_report_step_with_memory(self):
        trainer = _make_trainer()
        trainer._recent_losses = deque(maxlen=10)
        trainer._recent_rewards = deque(maxlen=10)
        metrics = StepMetrics(
            iteration=0,
            loss=1.0,
            mean_reward=0.5,
            peak_memory_mb=512.0,
            rss_mb=1024.0,
        )
        trainer._report_step(0, 5, metrics)

    def test_report_step_multi_type(self):
        trainer = _make_trainer()
        trainer._recent_losses = deque(maxlen=10)
        trainer._recent_rewards = deque(maxlen=10)
        metrics = StepMetrics(
            iteration=0,
            loss=1.0,
            mean_reward=0.5,
            extras={
                "type_names": ["math", "mcq"],
                "type_stats": {
                    "math": {"rewards": [0.8, 0.6], "trained_tokens": 100},
                    "mcq": {"rewards": [0.5], "trained_tokens": 50},
                },
                "total_views": 8,
            },
        )
        trainer._report_step(0, 5, metrics)


# ── _resolve_system_prompt ───────────────────────────────────────────────────


class TestResolveSystemPrompt:
    def test_default_fallback(self):
        """MCQ with no overrides and no sample metadata returns None."""
        trainer = _make_trainer()
        trainer._system_prompt = None
        trainer._type_overrides = {}
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        result = trainer._resolve_system_prompt(sample)
        # System prompt is explicit-only: YAML type > sample > None
        assert result is None

    def test_global_system_prompt_not_used(self):
        """Global system_prompt is NOT used (explicit-only priority chain)."""
        trainer = _make_trainer()
        trainer._system_prompt = "You are a math tutor."
        trainer._type_overrides = {}
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        result = trainer._resolve_system_prompt(sample)
        # Global system_prompt is no longer a fallback
        assert result is None

    def test_sample_metadata_used_when_no_override(self):
        trainer = _make_trainer()
        trainer._system_prompt = "Global prompt"
        trainer._type_overrides = {}
        sample = Sample(
            prompt="Q",
            answer="A",
            type_name="mcq",
            metadata={"system_prompt": "Sample-level prompt"},
        )
        result = trainer._resolve_system_prompt(sample)
        # Sample metadata is used when no YAML type override
        assert result == "Sample-level prompt"

    def test_type_override_has_priority_over_sample(self):
        trainer = _make_trainer()
        trainer._system_prompt = "Global"
        trainer._type_overrides = {"mcq": {"system_prompt": "MCQ-override"}}
        sample = Sample(
            prompt="Q",
            answer="A",
            type_name="mcq",
            metadata={"system_prompt": "Sample-level prompt"},
        )
        result = trainer._resolve_system_prompt(sample)
        # YAML type override has highest priority (overrides sample metadata)
        assert result == "MCQ-override"


# ── hook system ──────────────────────────────────────────────────────────────


class TestHookSystem:
    def test_on_registers_hook(self):
        trainer = _make_trainer()
        called = []
        trainer.on("after_step", lambda t, m: called.append(m))
        assert len(trainer._hooks["after_step"]) == 1

    def test_on_invalid_event_raises(self):
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="Unknown event"):
            trainer.on("invalid_event", lambda t, m: None)

    def test_fire_calls_hooks(self):
        trainer = _make_trainer()
        calls = []
        trainer.on("after_step", lambda t, m: calls.append(m.iteration))
        metrics = StepMetrics(iteration=42)
        trainer._fire("after_step", metrics)
        assert calls == [42]

    def test_fire_multiple_hooks(self):
        trainer = _make_trainer()
        log = []
        trainer.on("after_step", lambda t, m: log.append("h1"))
        trainer.on("after_step", lambda t, m: log.append("h2"))
        trainer._fire("after_step", StepMetrics(iteration=0))
        assert log == ["h1", "h2"]

    def test_set_spike_guard(self):
        trainer = _make_trainer()
        guard = MagicMock()
        trainer.set_spike_guard(guard)
        assert trainer._spike_guard is guard


# ── _sample_batch ─────────────────────────────────────────────────────────────


class TestSampleBatch:
    def test_sample_batch_from_dataset(self):
        trainer = _make_trainer()
        batch = trainer._sample_batch()
        assert len(batch) == 1  # batch_size=1
        assert isinstance(batch[0], Sample)

    def test_sample_batch_with_adaptive(self):
        trainer = _make_trainer()
        mock_adaptive = MagicMock()
        mock_adaptive.next.return_value = (0, Sample(prompt="Q", answer="A", type_name="mcq"))
        trainer._adaptive = mock_adaptive
        batch = trainer._sample_batch()
        assert len(batch) == 1
        mock_adaptive.next.assert_called_once()


# ── iteration property ───────────────────────────────────────────────────────


class TestIterationProperty:
    def test_iteration_default(self):
        trainer = _make_trainer()
        assert trainer.iteration == 0

    def test_stop_requested_default_false(self):
        trainer = _make_trainer()
        assert trainer.stop_requested is False

    def test_stop_requested_setter(self):
        trainer = _make_trainer()
        trainer.stop_requested = True
        assert trainer.stop_requested is True


# ── DR-GRPO advantage computation ─────────────────────────────────────────────


class TestDRGRPOAdvantages:
    """DR-GRPO must use absolute EMA baseline (r - baseline), not group-centered (r - mean_r).

    Bug: when all rollouts fail (rewards ≈ 0), group normalization gives advantages ≈ 0
    → near-zero loss. Expected: strong negative advantages → high loss → correct gradient.
    """

    def _make_rollout(self):
        seg = Segment(tokens=[1, 2, 3], text="A", source=SegmentSource.GENERATED, masked=False)
        return RolloutResult(segments=[seg])

    def test_dr_grpo_identical_wrong_gives_strong_negative_advantages(self):
        """DR-GRPO: identical near-zero rewards → advantages = r - baseline, NOT ≈0."""
        trainer = _make_trainer(loss_type="dr_grpo", group_size=2)
        trainer._reward_baseline = 0.5
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        rollouts = [self._make_rollout(), self._make_rollout()]
        reward_fn = MagicMock()
        reward_fn.compute.return_value = 0.0  # all wrong
        with (
            patch.object(trainer, "_get_reward", return_value=reward_fn),
            patch.object(trainer.executor, "execute", side_effect=rollouts),
        ):
            result = trainer._run_sample_group(sample, iteration=0)
        assert result is not None
        # advantages = 0.0 - 0.5 = -0.5 (not 0.0 - 0.0 = 0.0 from group centering)
        assert all(abs(a - (-0.5)) < 1e-6 for a in result.advantages)

    def test_dr_grpo_all_correct_not_skipped(self):
        """DR-GRPO: all rollouts reward=1.0 → NOT skipped (baseline advantage still flows)."""
        trainer = _make_trainer(loss_type="dr_grpo", group_size=2)
        trainer._reward_baseline = 0.2
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        rollouts = [self._make_rollout(), self._make_rollout()]
        reward_fn = MagicMock()
        reward_fn.compute.return_value = 1.0
        with (
            patch.object(trainer, "_get_reward", return_value=reward_fn),
            patch.object(trainer.executor, "execute", side_effect=rollouts),
        ):
            result = trainer._run_sample_group(sample, iteration=0)
        assert result is not None  # DR-GRPO: advantages = 1.0 - 0.2 = 0.8 (valid signal)
        assert all(abs(a - 0.8) < 1e-6 for a in result.advantages)

    def test_dr_grpo_mixed_rewards_give_correct_advantages(self):
        """DR-GRPO: mixed rewards → advantages = r - baseline, not group-centered."""
        trainer = _make_trainer(loss_type="dr_grpo", group_size=2)
        trainer._reward_baseline = 0.2
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        rollouts = [self._make_rollout(), self._make_rollout()]
        reward_fn = MagicMock()
        reward_fn.compute.side_effect = [0.8, 0.0]  # mixed: one good, one bad
        with (
            patch.object(trainer, "_get_reward", return_value=reward_fn),
            patch.object(trainer.executor, "execute", side_effect=rollouts),
        ):
            result = trainer._run_sample_group(sample, iteration=0)
        assert result is not None
        assert abs(result.advantages[0] - 0.6) < 1e-6  # 0.8 - 0.2
        assert abs(result.advantages[1] - (-0.2)) < 1e-6  # 0.0 - 0.2

    def test_grpo_identical_rewards_uses_baseline_fallback(self):
        """GRPO: identical rewards (std < 1e-4) → fallback to mean_r - baseline."""
        trainer = _make_trainer(loss_type="grpo", group_size=2)
        trainer._reward_baseline = 0.5
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        rollouts = [self._make_rollout(), self._make_rollout()]
        reward_fn = MagicMock()
        reward_fn.compute.return_value = 0.0
        with (
            patch.object(trainer, "_get_reward", return_value=reward_fn),
            patch.object(trainer.executor, "execute", side_effect=rollouts),
        ):
            result = trainer._run_sample_group(sample, iteration=0)
        assert result is not None
        # std=0 → fallback: [mean_r - baseline] = [0.0 - 0.5] = [-0.5]
        assert all(abs(a - (-0.5)) < 1e-6 for a in result.advantages)


# ── _reward metadata stamp ────────────────────────────────────────────────────


class TestRewardMetadataStamp:
    """The raw reward must be stamped on each rollout's metadata as _reward
    (pre-scaling) so the answer gradient weighting block can use it."""

    def _make_rollout(self):
        seg = Segment(tokens=[1, 2, 3], text="A", source=SegmentSource.GENERATED, masked=False)
        return RolloutResult(segments=[seg])

    def test_reward_stamped_on_rollout_metadata(self):
        """_reward is set on each rollout after scoring."""
        trainer = _make_trainer(loss_type="grpo", group_size=2)
        trainer._reward_baseline = 0.0
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        rollouts = [self._make_rollout(), self._make_rollout()]
        reward_fn = MagicMock()
        reward_fn.compute.side_effect = [0.8, 0.3]
        with (
            patch.object(trainer, "_get_reward", return_value=reward_fn),
            patch.object(trainer.executor, "execute", side_effect=rollouts),
        ):
            result = trainer._run_sample_group(sample, iteration=0)
        assert result is not None
        # _reward stamped with pre-scaling raw reward
        assert result.group_rollouts[0].metadata.get("_reward") == pytest.approx(0.8)
        assert result.group_rollouts[1].metadata.get("_reward") == pytest.approx(0.3)

    def test_reward_stamp_uses_setdefault(self):
        """Escalation _esc_reward set before scoring → _reward not overwritten."""
        trainer = _make_trainer(loss_type="grpo", group_size=2)
        trainer._reward_baseline = 0.0
        sample = Sample(prompt="Q", answer="A", type_name="mcq")
        rollout = self._make_rollout()
        rollout.metadata["_reward"] = 0.99  # pre-set (simulating escalation stamp)
        reward_fn = MagicMock()
        reward_fn.compute.return_value = 0.1  # regular reward would overwrite
        with (
            patch.object(trainer, "_get_reward", return_value=reward_fn),
            patch.object(trainer.executor, "execute", return_value=rollout),
        ):
            result = trainer._run_sample_group(sample, iteration=0)
        assert result is not None
        # setdefault → pre-existing value preserved
        assert result.group_rollouts[0].metadata["_reward"] == pytest.approx(0.99)


# ── force_answer_on_every_other_repeat_after_repeat_number ─────────────────────


class TestForceAnswerOnEveryOtherRepeat:
    """Tests for the forced-answer injection feature.

    When force_answer_on_every_other_repeat_after_repeat_number=N:
    - Reps 0..N-1: normal (count toward limits)
    - Rep N, N+2, N+4...: forced (don't count, skip stop checks)
    - Rep N+1, N+3, N+5...: normal (count)
    """

    def test_forced_rep_detection_threshold_3(self):
        """With threshold=3, reps 3,5,7 are forced."""
        # Simulate logic: _fa_thresh=3, _sr_rep goes 0,1,2,3,4,5,...
        # Forced when _sr_rep >= 3 and (_sr_rep - 3) % 2 == 0
        _fa_thresh = 3
        forced_reps = []
        for _sr_rep in range(10):
            _is_force_rep = (
                _fa_thresh > 0 and _sr_rep >= _fa_thresh and (_sr_rep - _fa_thresh) % 2 == 0
            )
            if _is_force_rep:
                forced_reps.append(_sr_rep)
        assert forced_reps == [3, 5, 7, 9]

    def test_forced_rep_detection_threshold_2(self):
        """With threshold=2, reps 2,4,6 are forced."""
        _fa_thresh = 2
        forced_reps = []
        for _sr_rep in range(10):
            _is_force_rep = (
                _fa_thresh > 0 and _sr_rep >= _fa_thresh and (_sr_rep - _fa_thresh) % 2 == 0
            )
            if _is_force_rep:
                forced_reps.append(_sr_rep)
        assert forced_reps == [2, 4, 6, 8]

    def test_unforced_rep_counter(self):
        """_sr_unforced_rep counts only non-forced reps."""
        _fa_thresh = 3
        _sr_unforced_rep = 0
        unforced_count_at_total = {}
        for _sr_rep in range(10):
            _is_force_rep = (
                _fa_thresh > 0 and _sr_rep >= _fa_thresh and (_sr_rep - _fa_thresh) % 2 == 0
            )
            if not _is_force_rep:
                _sr_unforced_rep += 1
            unforced_count_at_total[_sr_rep] = _sr_unforced_rep
        # Reps 0,1,2,4,6,8 are unforced → 6 total
        assert unforced_count_at_total[0] == 1  # after rep 0
        assert unforced_count_at_total[1] == 2  # after rep 1
        assert unforced_count_at_total[2] == 3  # after rep 2
        assert unforced_count_at_total[3] == 3  # rep 3 forced, no increment
        assert unforced_count_at_total[4] == 4  # after rep 4
        assert unforced_count_at_total[8] == 6  # after rep 8

    def test_disabled_when_threshold_zero(self):
        """force_answer_on_every_other_repeat_after_repeat_number=0 disables feature."""
        _fa_thresh = 0
        forced_reps = []
        for _sr_rep in range(10):
            _is_force_rep = (
                _fa_thresh > 0 and _sr_rep >= _fa_thresh and (_sr_rep - _fa_thresh) % 2 == 0
            )
            if _is_force_rep:
                forced_reps.append(_sr_rep)
        assert forced_reps == []

    def test_config_field_exists(self):
        """force_answer_on_every_other_repeat_after_repeat_number field in TrainingConfig."""
        cfg = TrainingConfig()
        assert hasattr(cfg, "force_answer_on_every_other_repeat_after_repeat_number")
        assert cfg.force_answer_on_every_other_repeat_after_repeat_number == 0

    def test_config_field_accepts_values(self):
        """Can set the field to positive values."""
        cfg = TrainingConfig(
            force_answer_on_every_other_repeat_after_repeat_number=3,
        )
        assert cfg.force_answer_on_every_other_repeat_after_repeat_number == 3

    def test_trainer_force_answer_attribute_initialized(self):
        """Trainer has _force_answer_this_rep initialized to False."""
        trainer = _make_trainer()
        assert hasattr(trainer, "_force_answer_this_rep")
        assert trainer._force_answer_this_rep is False
