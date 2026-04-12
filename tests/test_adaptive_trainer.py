"""Tests for adaptive escalation trainer paths in training/__init__.py."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import patch

from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.training import Trainer, _prefix_from_thinking

# ── _prefix_from_thinking ────────────────────────────────────────────────────


class _WordBackend:
    """Word-level mock backend for deterministic token encoding."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class TestPrefixFromThinking:
    def test_strips_think_tag(self):
        b = _WordBackend()
        out = _prefix_from_thinking("<think>word1 word2 word3", 2, b)
        assert "word1" in out
        assert "word3" not in out

    def test_zero_tokens_empty(self):
        assert _prefix_from_thinking("word1 word2", 0, _WordBackend()) == ""

    def test_no_think_tag(self):
        b = _WordBackend()
        out = _prefix_from_thinking("step1 step2 step3", 2, b)
        assert "step1" in out

    def test_longer_than_text(self):
        b = _WordBackend()
        out = _prefix_from_thinking("<think>a b c", 100, b)
        assert "a" in out and "b" in out and "c" in out

    def test_empty_text(self):
        assert _prefix_from_thinking("", 5, _WordBackend()) == ""

    def test_only_think_tag(self):
        assert _prefix_from_thinking("<think>", 5, _WordBackend()) == ""

    def test_whitespace_stripping(self):
        b = _WordBackend()
        out = _prefix_from_thinking("  <think>  word1 word2", 1, b)
        assert "word1" in out

    def test_stops_before_end_think_tag(self):
        """Prefix must not include </think> or anything after it."""
        b = _WordBackend()
        full = "<think>step1 step2</think>\\boxed{42}"
        out = _prefix_from_thinking(full, 10, b)
        assert "</think>" not in out
        assert "\\boxed" not in out
        assert "step1" in out

    def test_stops_before_end_think_with_partial_prefix(self):
        """Requesting 1 token from 'a b </think> \\boxed{1}' gives only 'a'."""
        b = _WordBackend()
        out = _prefix_from_thinking("<think>a b </think>\\boxed{1}", 1, b)
        assert out.strip() == "a"
        assert "</think>" not in out


# ── Adaptive math rollouts via Trainer ──────────────────────────────────────


def _make_math_dataset(n: int = 3) -> DatasetLoader:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "prompt": f"Solve {i}+1=?",
                        "answer": "<think>step by step</think>\\boxed{" + str(i + 1) + "}",
                        "type": "math",
                        "metadata": {"ground_truth": str(i + 1)},
                    }
                )
                + "\n"
            )
        f.flush()
        return DatasetLoader(DatasetConfig(path=f.name, shuffle=False)).load()


class TestAdaptiveMathTrainer:
    def test_adaptive_escalation_runs(self, mock_backend):
        """Trainer with view_strategy=adaptive_escalation runs without crash."""
        dataset = _make_math_dataset(3)
        config = TrainingConfig(
            iterations=1,
            group_size=2,
            save_every=0,
            eval_every=0,
        )
        with patch("time.sleep"):  # Don't actually sleep during tests
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 2,
                        "escalation_step_tokens": 1,
                        "postfix_tokens": 0,
                    }
                },
            )
            trainer.train()
            assert trainer.iteration == 0

    def test_adaptive_generates_variable_views(self, mock_backend):
        """Adaptive loop generates 2+ rollouts per step."""
        dataset = _make_math_dataset(3)
        config = TrainingConfig(
            iterations=1,
            group_size=2,
            save_every=0,
            eval_every=0,
        )
        captured = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 2,
                        "escalation_step_tokens": 1,
                        "postfix_tokens": 0,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()

        assert len(captured) == 1
        # With max_escalations=2 + naked = 3 total rollouts minimum
        m = captured[0]
        assert m.extras.get("total_views", 0) >= 2

    def test_adaptive_with_stop_sequence_landing(self, mock_backend_factory):
        """When model lands (stop_sequence), loop breaks early."""
        # Configure backend to return </think> in first response
        backend = mock_backend_factory(
            call_1="thinking</think>",
            call_2="42",
        )
        dataset = _make_math_dataset(1)
        config = TrainingConfig(
            iterations=1,
            group_size=2,
            save_every=0,
            eval_every=0,
        )
        captured = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 5,
                        "escalation_step_tokens": 1,
                        "postfix_tokens": 0,
                        "stop_sequences": ["</think>"],
                        "thinking_budget": 20,
                        "answer_max_tokens": 20,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()

        assert len(captured) == 1

    def test_non_adaptive_type_uses_fixed_loop(self, mock_backend):
        """Without view_strategy, uses the standard fixed group loop."""
        dataset = _make_math_dataset(3)
        config = TrainingConfig(
            iterations=1,
            group_size=3,
            save_every=0,
            eval_every=0,
        )
        captured = []
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.on("after_step", lambda t, m: captured.append(m))
        trainer.train()
        assert len(captured[0].rewards) == 3  # fixed group_size=3

    def test_adaptive_postfix_injection(self, mock_backend):
        """With escalation_step_tokens and postfix_tokens, last attempt injects postfix."""
        dataset = _make_math_dataset(1)
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        captured_views = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 3,
                        "escalation_step_tokens": 2,
                        "postfix_tokens": 5,
                    }
                },
            )
            trainer.on(
                "after_step", lambda t, m: captured_views.append(m.extras.get("total_views", 0))
            )
            trainer.train()

        assert captured_views[0] >= 2

    def test_overfull_prefix_skipped(self, mock_backend):
        """Escalation attempt skipped when prefix_len >= thinking token count.

        Dataset answer is '<think>step by step</think>\\boxed{N+1}'.
        Thinking content = 'step by step' = 12 chars (char-level mock tokenizer).
        With escalation_step_tokens=100, attempt_idx=1 has prefix_len=100 >= 12
        and must be skipped.  Only escalation_0 + naked should run → total_views==2.
        """
        dataset = _make_math_dataset(1)
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        captured = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 3,
                        "escalation_step_tokens": 100,  # larger than ~31-char target
                        "postfix_tokens": 0,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()

        assert len(captured) == 1
        # Only escalation_0 (prefix=0) + naked survive; attempt_idx=1,2 skipped
        assert captured[0].extras.get("total_views") == 2

    def test_escalation_recovery_generates_answer(self, mock_backend):
        """Escalation rollout applies recovery when thinking hits max_tokens.

        With recovery_ratio=1.0, every attempt that hits max_tokens should
        still produce a complete rollout (with injected </think> + answer).
        The MathType.reward() should be able to find \\boxed{} in completion_text.
        """
        dataset = _make_math_dataset(1)
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        captured = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 2,
                        "escalation_step_tokens": 1,
                        "postfix_tokens": 0,
                        "recovery_ratio": 1.0,  # always recover
                        "answer_max_tokens": 5,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()

        assert len(captured) == 1
        # With recovery_ratio=1.0, all escalation attempts generate answers.
        # total_views = max_escalations + naked = 3
        assert captured[0].extras.get("total_views") >= 2

    def test_escalation_recovery_abandoned_when_ratio_zero(self, mock_backend):
        """With recovery_ratio=0, escalation attempts that hit max_tokens abandon."""
        dataset = _make_math_dataset(1)
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        captured = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 2,
                        "escalation_step_tokens": 1,
                        "postfix_tokens": 0,
                        "recovery_ratio": 0.0,  # never recover → always abandon
                        "answer_max_tokens": 5,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()

        assert len(captured) == 1

    def test_think_token_prepended_when_tokenizer_present(self, mock_backend):
        """When Trainer has a tokenizer and enable_thinking=True, <think> is appended
        to prompt_tokens so prefix injection lands inside the thinking context."""

        dataset = _make_math_dataset(1)
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)

        prepended_think = []

        class MockTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]  # prompt without <think>

            def decode(self, tokens):
                # Simulates prompt tail without <think>
                return "assistant\n"

            def encode(self, text, add_special_tokens=False):
                if "<think>" in text:
                    prepended_think.append(True)
                    return [9999]  # <think> mock token
                return [ord(c) for c in text]

        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                tokenizer=MockTokenizer(),
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 1,
                        "escalation_step_tokens": 0,
                        "postfix_tokens": 0,
                        "recovery_ratio": 1.0,
                    }
                },
            )
            trainer.train()

        # <think> token must have been appended to prompt_tokens
        assert prepended_think, "<think> was not prepended to prompt_tokens"

    def test_hooks_fire_for_adaptive(self, mock_backend):
        """after_step fires once per iteration even with adaptive loop."""
        dataset = _make_math_dataset(3)
        config = TrainingConfig(
            iterations=2,
            group_size=2,
            save_every=0,
            eval_every=0,
        )
        fired = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 2,
                        "escalation_step_tokens": 0,
                        "postfix_tokens": 0,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: fired.append(m.iteration))
            trainer.train()

        assert fired == [0, 1]
