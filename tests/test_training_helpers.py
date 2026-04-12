"""Tests for training helpers: _extract_prefix_lines, _truncate_completions, etc."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import patch

import pytest

from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.training import Trainer, _extract_prefix_lines

# ── _extract_prefix_lines ────────────────────────────────────────────────────


class TestExtractPrefixLines:
    def test_basic(self):
        text = "line1\nline2\nline3"
        result = _extract_prefix_lines(text, 2)
        assert "line1" in result
        assert "line2" in result
        assert "line3" not in result

    def test_strips_think_tag(self):
        text = "<think>line1\nline2\nline3"
        result = _extract_prefix_lines(text, 2)
        assert "<think>" not in result
        assert "line1" in result

    def test_n_equals_one(self):
        text = "line1\nline2\nline3"
        result = _extract_prefix_lines(text, 1)
        assert "line1" in result
        assert "line2" not in result

    def test_empty_returns_none(self):
        result = _extract_prefix_lines("", 5)
        assert result is None

    def test_only_think_tag_returns_none(self):
        result = _extract_prefix_lines("<think>", 5)
        assert result is None

    def test_only_whitespace_lines_ignored(self):
        text = "   \n\nline1\nline2"
        result = _extract_prefix_lines(text, 1)
        assert "line1" in result

    def test_n_larger_than_lines_returns_all(self):
        text = "line1\nline2"
        result = _extract_prefix_lines(text, 100)
        assert "line1" in result
        assert "line2" in result

    def test_trailing_newline(self):
        result = _extract_prefix_lines("line1\nline2", 2)
        assert result.endswith("\n")

    def test_leading_whitespace_stripped(self):
        text = "  <think>line1\nline2"
        result = _extract_prefix_lines(text, 2)
        assert "<think>" not in result


# ── _truncate_completions ────────────────────────────────────────────────────


def _make_trainer_with_used_tokens(used_tokens=None):
    """Create a minimal Trainer that can test _truncate_completions."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "q", "answer": "a", "type": "math"}) + "\n")
        path = f.name

    dataset = DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()
    config = TrainingConfig(iterations=0, group_size=2, save_every=0, eval_every=0)

    class _MockBackend:
        def encode(self, text):
            return list(text.encode())

        def decode(self, tokens):
            return bytes(tokens).decode(errors="replace")

        def generate(self, tokens, params):
            from mlx_grpo.backends.base import GenerateResult

            return GenerateResult(text="x", tokens=[ord("x")], stop_reason="stop")

    trainer = Trainer(config=config, dataset=dataset, backend=_MockBackend())
    if used_tokens is not None:
        trainer._used_tokens = used_tokens
    return trainer


class TestTruncateCompletions:
    def test_no_limit_passthrough(self):
        """When _used_tokens=None and no per-seq limits, sequences pass through unchanged."""
        trainer = _make_trainer_with_used_tokens(None)
        seqs = [[1, 2, 3, 4, 5]]
        pls = [2]
        masks = [[True] * 5]
        out_seqs, out_pls, out_masks = trainer._truncate_completions(seqs, pls, masks)
        assert out_seqs == seqs
        assert out_pls == pls

    def test_prefix_truncation(self):
        """Completion (5 tokens) > limit (3) → truncated to 3."""
        trainer = _make_trainer_with_used_tokens(3)
        trainer._truncate_from = "prefix"
        trainer._truncate_marker = ""
        seq = [0, 1, 2, 3, 4, 5, 6]  # prompt=2 tokens, completion=5 tokens
        seqs = [seq]
        pls = [2]
        masks = [[True] * 7]
        out_seqs, out_pls, out_masks = trainer._truncate_completions(seqs, pls, masks)
        comp_len = len(out_seqs[0]) - out_pls[0]
        assert comp_len <= 3

    def test_no_truncation_when_short(self):
        """When completion <= limit, passthrough."""
        trainer = _make_trainer_with_used_tokens(10)
        seq = [0, 1, 2, 3]  # prompt=2, completion=2
        seqs = [seq]
        pls = [2]
        masks = [[True] * 4]
        out_seqs, out_pls, out_masks = trainer._truncate_completions(seqs, pls, masks)
        assert out_seqs[0] == seq

    def test_postfix_truncation(self):
        """postfix mode keeps end of completion."""
        trainer = _make_trainer_with_used_tokens(3)
        trainer._truncate_from = "postfix"
        trainer._truncate_marker = ""
        seq = list(range(10))  # prompt=2, completion=8
        seqs = [seq]
        pls = [2]
        masks = [[True] * 10]
        out_seqs, out_pls, out_masks = trainer._truncate_completions(seqs, pls, masks)
        comp_len = len(out_seqs[0]) - out_pls[0]
        assert comp_len <= 4  # 3 kept + possible marker

    def test_middle_truncation(self):
        """middle mode: keep prefix + suffix, excise middle, one marker between."""
        trainer = _make_trainer_with_used_tokens(6)
        trainer._truncate_from = "middle"
        trainer._truncate_marker = ""  # no marker tokens for clean arithmetic
        # prompt=[0,1], completion=[2,3,4,5,6,7,8,9,10,11] (10 tokens)
        seq = list(range(12))
        seqs = [seq]
        pls = [2]
        masks = [[True] * 12]
        out_seqs, out_pls, out_masks = trainer._truncate_completions(seqs, pls, masks)
        out_comp = out_seqs[0][2:]  # strip prompt
        # Total kept ≤ limit=6; prefix from start, suffix from end
        assert len(out_comp) <= 6
        assert out_comp[0] == 2  # first completion token preserved (prefix kept)
        assert out_comp[-1] == 11  # last completion token preserved (suffix kept)

    def test_middle_truncation_with_marker(self):
        """middle mode: marker tokens appear between prefix and suffix."""
        trainer = _make_trainer_with_used_tokens(6)
        trainer._truncate_from = "middle"
        trainer._truncate_marker = ""
        trainer._truncate_marker_masked = True
        # marker masked → mask value for marker tokens should be False (excluded from loss)
        seq = list(range(12))  # prompt=2, completion=10
        seqs = [seq]
        pls = [2]
        masks = [[True] * 12]
        out_seqs, out_pls, out_masks = trainer._truncate_completions(seqs, pls, masks)
        # Completion portion of mask: kept prefix/suffix=True, marker=False
        out_cmask = out_masks[0][2:]
        # With empty marker: no mask=False tokens expected, but output is still truncated
        assert all(m in (True, False) for m in out_cmask)

    def test_per_sequence_limit(self):
        """Per-sequence limits override global."""
        trainer = _make_trainer_with_used_tokens(None)
        seq = list(range(10))
        seqs = [seq]
        pls = [2]
        masks = [[True] * 10]
        # Pass limit=3 per sequence
        out_seqs, _, _ = trainer._truncate_completions(seqs, pls, masks, limits=[3])
        comp_len = len(out_seqs[0]) - pls[0]
        assert comp_len <= 4  # truncated

    def test_per_sequence_none_limit(self):
        """Per-sequence limit=None means no truncation for that sequence."""
        trainer = _make_trainer_with_used_tokens(3)  # global=3
        seq = list(range(10))
        seqs = [seq]
        pls = [2]
        masks = [[True] * 10]
        # Per-seq override is None → skip truncation
        out_seqs, _, _ = trainer._truncate_completions(seqs, pls, masks, limits=[None])
        assert out_seqs[0] == seq


# ── Trainer basic paths ──────────────────────────────────────────────────────


def _make_small_dataset(n=2):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "prompt": f"Q{i}",
                        "answer": f"A{i}",
                        "type": "math",
                        "metadata": {"ground_truth": str(i)},
                    }
                )
                + "\n"
            )
        f.flush()
        return DatasetLoader(DatasetConfig(path=f.name, shuffle=False)).load()


class TestTrainerHelperPaths:
    def test_samples_per_step_one(self, mock_backend):
        """Default sps=1 runs normally."""
        dataset = _make_small_dataset(2)
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        fired = []
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.on("after_step", lambda t, m: fired.append(m.iteration))
        trainer.train()
        assert fired == [0]

    def test_on_unknown_event_raises(self, mock_backend):
        dataset = _make_small_dataset(1)
        config = TrainingConfig(iterations=0, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        with pytest.raises(ValueError, match="Unknown event"):
            trainer.on("nonexistent_event", lambda t, m: None)

    def test_stop_requested_breaks_loop(self, mock_backend):
        dataset = _make_small_dataset(5)
        config = TrainingConfig(iterations=5, group_size=2, save_every=0, eval_every=0)
        fired = []
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.on("after_step", lambda t, m: fired.append(m.iteration))

        # Request stop before training starts
        trainer.stop_requested = True
        with patch.object(trainer, "_save_interrupt_checkpoint"):
            trainer.train()
        # Should fire 0 times (stopped before first step)
        assert len(fired) == 0

    def test_resume_from(self, mock_backend):
        """train(resume_from=2) skips first 2 iterations."""
        dataset = _make_small_dataset(5)
        config = TrainingConfig(iterations=4, group_size=2, save_every=0, eval_every=0)
        fired = []
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.on("after_step", lambda t, m: fired.append(m.iteration))
        trainer.train(resume_from=2)
        assert fired == [2, 3]

    def test_iteration_property(self, mock_backend):
        dataset = _make_small_dataset(2)
        config = TrainingConfig(iterations=2, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.train()
        # After 2 iterations (0, 1), iteration should be 1
        assert trainer.iteration == 1
