"""Tests for RolloutLogger — detailed JSONL rollout logging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training.rollout_logger import RolloutLogger, StepContext


def _make_rollout(text: str = "test output") -> RolloutResult:
    return RolloutResult(
        segments=[Segment([ord(c) for c in text], text, SegmentSource.GENERATED, False)]
    )


class TestStepContext:
    def test_default_values(self):
        ctx = StepContext()
        assert ctx.iteration == 0
        assert ctx.loss == 0.0
        assert ctx.kl_divergence == 0.0

    def test_set_values(self):
        ctx = StepContext(
            iteration=5,
            loss=1.2,
            kl_divergence=0.04,
            grad_norm=0.9,
            tokens_per_sec=150.0,
            elapsed_sec=3.0,
            mean_reward=0.7,
            reward_std=0.15,
        )
        assert ctx.iteration == 5
        assert ctx.loss == 1.2
        assert ctx.grad_norm == 0.9


class TestRolloutLoggerInit:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            logger.close()
            assert path.exists()

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "rollouts.jsonl"
            logger = RolloutLogger(path)
            logger.close()
            assert path.exists()

    def test_appends_to_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            r = _make_rollout()

            l1 = RolloutLogger(path)
            l1.log(0, [r], [0.5], [0.1])
            l1.close()

            l2 = RolloutLogger(path)
            l2.log(1, [r], [0.8], [0.3])
            l2.close()

            lines = path.read_text().strip().splitlines()
            assert len(lines) == 2


class TestRolloutLoggerLog:
    def test_single_rollout_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            r = _make_rollout("hello world")
            logger.log(iteration=3, rollouts=[r], rewards=[1.0], advantages=[0.5])
            logger.close()
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["iteration"] == 3
            assert entry["reward"] == 1.0
            assert entry["advantage"] == 0.5
            assert entry["group_index"] == 0

    def test_multiple_rollouts_group_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            rollouts = [_make_rollout(f"r{i}") for i in range(4)]
            logger.log(0, rollouts, [0.0, 0.5, 1.0, 0.75], [0.0, 0.1, 0.2, 0.15])
            logger.close()
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 4
            for i, line in enumerate(lines):
                entry = json.loads(line)
                assert entry["group_index"] == i

    def test_correct_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            r = _make_rollout()
            logger.log(0, [r, r], [1.0, 0.5], [0.1, -0.1])
            logger.close()
            lines = path.read_text().strip().splitlines()
            assert json.loads(lines[0])["correct"] is True
            assert json.loads(lines[1])["correct"] is False

    def test_prompt_and_target_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            r = _make_rollout()
            logger.log(
                0,
                [r],
                [0.5],
                [0.1],
                prompts=["What is 2+2?"],
                answers=["<think>...4...</think>\\boxed{4}"],
                targets=["4"],
            )
            logger.close()
            entry = json.loads(path.read_text().strip())
            assert entry["prompt"] == "What is 2+2?"
            assert entry["target_answer"] == "4"

    def test_step_context_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            ctx = StepContext(
                iteration=10,
                total_iterations=100,
                loss=0.5,
                kl_divergence=0.02,
                grad_norm=0.4,
                tokens_per_sec=200.0,
                elapsed_sec=1.5,
                mean_reward=0.8,
                reward_std=0.1,
            )
            r = _make_rollout()
            logger.log(10, [r], [1.0], [0.5], step_ctx=ctx)
            logger.close()
            entry = json.loads(path.read_text().strip())
            assert "step" in entry
            assert entry["step"]["loss"] == 0.5
            assert entry["step"]["kl"] == 0.02
            assert entry["step"]["progress"] == 0.1

    def test_no_step_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            r = _make_rollout()
            logger.log(0, [r], [0.5], [0.0])  # no step_ctx
            logger.close()
            entry = json.loads(path.read_text().strip())
            assert "step" not in entry

    def test_timestamp_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            r = _make_rollout()
            logger.log(0, [r], [0.5], [0.0])
            logger.close()
            entry = json.loads(path.read_text().strip())
            assert "timestamp" in entry
            assert entry["timestamp"] > 0

    def test_no_tokens_flag(self):
        """log_tokens=False should reduce entry size but not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path, log_tokens=False)
            r = _make_rollout()
            logger.log(0, [r], [0.5], [0.0])
            logger.close()
            entry = json.loads(path.read_text().strip())
            assert "iteration" in entry  # basic fields always present

    def test_flush_happens(self):
        """File is readable immediately after log() without close()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            r = _make_rollout()
            logger.log(0, [r], [0.5], [0.0])
            # Don't close — flush should have happened
            content = path.read_text()
            assert len(content) > 0
            logger.close()


class TestRolloutLoggerClose:
    def test_close_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            logger.close()
            logger.close()  # Should not raise


class TestStepContextCkaDrift:
    def test_cka_drift_default_none(self) -> None:
        ctx = StepContext()
        assert ctx.cka_drift is None

    def test_cka_drift_included_in_step_json(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            ctx = StepContext(iteration=3, cka_drift=0.876)
            r = _make_rollout()
            logger.log(3, [r], [1.0], [0.5], step_ctx=ctx)
            logger.close()
            entry = json.loads(path.read_text().strip())
            assert abs(entry["step"]["cka_drift"] - 0.876) < 1e-5

    def test_cka_drift_omitted_when_none(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)
            ctx = StepContext(iteration=3)  # cka_drift=None
            r = _make_rollout()
            logger.log(3, [r], [1.0], [0.5], step_ctx=ctx)
            logger.close()
            entry = json.loads(path.read_text().strip())
            assert "cka_drift" not in entry["step"]


def _mock_rollout():
    from unittest.mock import MagicMock

    r = MagicMock()
    r.to_log_entry.return_value = {
        "type": "math",
        "generated_answer": "42",
        "extracted_answer": "42",
        "segments": [],
        "variables": {},
        "token_counts": {"prompt": 10, "generated": 20},
    }
    return r


class TestRolloutLoggerCorrectField:
    def _make_logger(self, threshold=None):
        tmp = tempfile.mktemp(suffix=".jsonl")
        kwargs = {"log_tokens": False}
        if threshold is not None:
            kwargs["correct_threshold"] = threshold
        return RolloutLogger(tmp, **kwargs), tmp

    def _read(self, path):
        return [json.loads(line) for line in open(path) if line.strip()]

    def _log_one(self, logger, reward):
        logger.log(
            0,
            [_mock_rollout()],
            [reward],
            [0.1],
            prompts=["q"],
            answers=["a"],
            targets=["42"],
        )
        logger.close()

    def test_default_threshold_is_0_5(self):
        logger, _ = self._make_logger()
        assert logger._correct_threshold == 0.5

    def test_reward_above_threshold_is_correct(self):
        logger, path = self._make_logger()
        self._log_one(logger, 0.8)
        assert self._read(path)[0]["correct"] is True

    def test_reward_at_threshold_is_not_correct(self):
        # strictly greater-than: 0.5 > 0.5 is False
        logger, path = self._make_logger(threshold=0.5)
        self._log_one(logger, 0.5)
        assert self._read(path)[0]["correct"] is False

    def test_reward_below_threshold_is_wrong(self):
        logger, path = self._make_logger()
        self._log_one(logger, 0.2)
        assert self._read(path)[0]["correct"] is False

    def test_shaped_reward_0_97_is_correct(self):
        # escalation-shaped correct answer (was 1.0, shaped to 0.97)
        logger, path = self._make_logger()
        self._log_one(logger, 0.97)
        assert self._read(path)[0]["correct"] is True

    def test_reward_above_1_is_correct(self):
        # ThinkingEfficiencyReward can push above 1.0
        logger, path = self._make_logger()
        self._log_one(logger, 1.15)
        assert self._read(path)[0]["correct"] is True

    def test_zero_reward_is_wrong(self):
        logger, path = self._make_logger()
        self._log_one(logger, 0.0)
        assert self._read(path)[0]["correct"] is False

    def test_legacy_threshold_1_0_rejects_0_98(self):
        # prove old behaviour recoverable via opt-in
        logger, path = self._make_logger(threshold=1.0)
        self._log_one(logger, 0.98)
        assert self._read(path)[0]["correct"] is False


class TestGroupBasedCorrect:
    """Test 'correct' field marks max-advantage group member."""

    def test_max_advantage_marked_correct_in_group(self):
        """With group_size=4, entry with max advantage gets correct=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)

            rollouts = [_make_rollout() for _ in range(4)]
            rewards = [0.6, 1.0, 0.7, 0.8]
            advantages = [0.2, 0.8, 0.3, 0.5]  # Max at index 1

            logger.log(
                0,
                rollouts,
                rewards,
                advantages,
                group_size=4,  # Enable group-based behavior
            )
            logger.close()

            entries = [
                json.loads(line) for line in path.read_text().strip().splitlines()
            ]
            assert entries[1]["correct"] is True
            assert entries[0]["correct"] is False
            assert entries[2]["correct"] is False
            assert entries[3]["correct"] is False

    def test_multiple_groups_each_has_one_correct(self):
        """With 2 groups of 3, each group has exactly one max-advantage member."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path)

            rollouts = [_make_rollout() for _ in range(6)]
            rewards = [0.5] * 6
            advantages = [
                0.1,
                0.8,  # Max in group 1
                0.2,
                0.3,
                0.9,  # Max in group 2
                0.4,
            ]

            logger.log(
                0,
                rollouts,
                rewards,
                advantages,
                group_size=3,
            )
            logger.close()

            entries = [
                json.loads(line) for line in path.read_text().strip().splitlines()
            ]
            assert entries[1]["correct"] is True
            assert entries[4]["correct"] is True
            # All others False
            assert all(entries[i]["correct"] is False for i in [0, 2, 3, 5])

    def test_group_size_1_each_entry_correct_based_on_threshold(self):
        """With group_size=1, each member is its own group, falls back to threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path, correct_threshold=0.5)

            rollouts = [_make_rollout() for _ in range(3)]
            rewards = [0.3, 0.7, 0.4]
            advantages = [0.1, 0.9, 0.2]

            logger.log(
                0,
                rollouts,
                rewards,
                advantages,
                group_size=1,  # Each is own group
            )
            logger.close()

            entries = [
                json.loads(line) for line in path.read_text().strip().splitlines()
            ]
            # Falls back to threshold comparison per group (each size 1)
            # Index 0: max in group [0.1], but reward 0.3 < 0.5
            # Index 1: max in group [0.9], and reward 0.7 > 0.5
            # Index 2: max in group [0.2], but reward 0.4 < 0.5
            assert entries[0]["correct"] is False
            assert entries[1]["correct"] is True
            assert entries[2]["correct"] is False

    def test_no_group_size_uses_reward_threshold(self):
        """Without group_size, behavior reverts to reward > threshold (backward compat)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path, correct_threshold=0.5)

            rollouts = [_make_rollout() for _ in range(3)]
            rewards = [0.3, 0.7, 0.4]
            advantages = [0.1, 0.9, 0.2]

            logger.log(
                0,
                rollouts,
                rewards,
                advantages,
                # No group_size parameter
            )
            logger.close()

            entries = [
                json.loads(line) for line in path.read_text().strip().splitlines()
            ]
            # Legacy: correct = reward > 0.5
            assert entries[0]["correct"] is False
            assert entries[1]["correct"] is True
            assert entries[2]["correct"] is False
