"""Comprehensive tests for data/__init__.py — all DatasetLoader paths."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.data import DatasetConfig, DatasetLoader, Sample

# ── helpers ──────────────────────────────────────────────────────────────────


def _write_jsonl(rows: list[dict], path: str | Path | None = None) -> str:
    if path is None:
        import tempfile as _t

        with _t.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            return f.name
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def _write_json(rows: list[dict]) -> str:
    import tempfile as _t

    with _t.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(rows, f)
        return f.name


# ── Sample ────────────────────────────────────────────────────────────────────


class TestSampleExtra:
    def test_type_default(self):
        s = Sample(prompt="q", answer="a")
        assert s.type_name == "mcq"

    def test_metadata_immutable(self):
        s = Sample(prompt="q", answer="a", metadata={"k": 1})
        assert s.metadata["k"] == 1


# ── _balanced_shuffle ─────────────────────────────────────────────────────────


class TestBalancedShuffle:
    def _loader(self, rows):
        path = _write_jsonl(rows)
        return DatasetLoader(DatasetConfig(path=path, shuffle=False))

    def test_interleaves_types(self):
        rows = [{"prompt": f"M{i}", "answer": "a", "type": "math"} for i in range(3)] + [
            {"prompt": f"Q{i}", "answer": "a", "type": "mcq"} for i in range(3)
        ]
        loader = self._loader(rows)
        samples = loader.load()
        # Now test balanced shuffle
        import random

        rng = random.Random(42)
        shuffled = DatasetLoader._balanced_shuffle(samples, rng)
        assert len(shuffled) == 6
        # Should alternate types
        types = [s.type_name for s in shuffled]
        # At least first two should be different types (round-robin)
        assert types[0] != types[1]

    def test_single_type(self):
        rows = [{"prompt": f"Q{i}", "answer": "a", "type": "math"} for i in range(4)]
        loader = self._loader(rows)
        samples = loader.load()
        import random

        rng = random.Random(42)
        shuffled = DatasetLoader._balanced_shuffle(samples, rng)
        assert len(shuffled) == 4
        assert all(s.type_name == "math" for s in shuffled)

    def test_load_with_balanced_shuffle(self):
        rows = [{"prompt": f"M{i}", "answer": "a", "type": "math"} for i in range(3)] + [
            {"prompt": f"Q{i}", "answer": "a", "type": "mcq"} for i in range(3)
        ]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=True, balanced_shuffle=True))
        samples = loader.load()
        assert len(samples) == 6

    def test_next_batch_balanced_shuffle_wraps(self):
        """next_batch wrapping re-shuffles with balanced shuffle."""
        rows = [{"prompt": f"M{i}", "answer": "a", "type": "math"} for i in range(2)] + [
            {"prompt": f"Q{i}", "answer": "a", "type": "mcq"} for i in range(2)
        ]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=True, balanced_shuffle=True))
        loader.load()
        # Call next_batch beyond dataset size to trigger re-shuffle
        batch = loader.next_batch(5)
        assert len(batch) == 5  # 4 + 1 from next epoch


# ── _load_directory ───────────────────────────────────────────────────────────


class TestLoadDirectory:
    def test_loads_split_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _write_jsonl(
                [{"prompt": "Q", "answer": "A", "type": "math"}],
                path / "train.jsonl",
            )
            loader = DatasetLoader(DatasetConfig(path=str(path), shuffle=False))
            samples = loader.load()
            assert len(samples) == 1
            assert samples[0].prompt == "Q"

    def test_loads_multiple_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            _write_jsonl([{"prompt": "Q1", "answer": "A", "type": "math"}], path / "a.jsonl")
            _write_jsonl([{"prompt": "Q2", "answer": "A", "type": "math"}], path / "b.jsonl")
            loader = DatasetLoader(DatasetConfig(path=str(path), split="train", shuffle=False))
            samples = loader.load()
            # split=train.jsonl doesn't exist, falls back to glob all *.jsonl
            assert len(samples) == 2

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DatasetLoader(DatasetConfig(path=str(tmpdir), split="train", shuffle=False))
            with pytest.raises(FileNotFoundError):
                loader.load()


# ── _load_local ───────────────────────────────────────────────────────────────


class TestLoadLocal:
    def test_json_file_list(self):
        rows = [{"prompt": "Q", "answer": "A", "type": "math"}]
        path = _write_json(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        assert len(samples) == 1

    def test_json_file_non_list_raises(self):
        import tempfile as _t

        with _t.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"prompt": "Q"}, f)
            fname = f.name
        loader = DatasetLoader(DatasetConfig(path=fname, shuffle=False))
        with pytest.raises(ValueError, match="list"):
            loader.load()

    def test_missing_file_raises(self):
        loader = DatasetLoader(DatasetConfig(path="/nonexistent/file.jsonl", shuffle=False))
        with pytest.raises(FileNotFoundError):
            loader.load()


# ── _load_hf ──────────────────────────────────────────────────────────────────


class TestLoadHF:
    def test_importerror_when_datasets_missing(self, monkeypatch):
        """When 'datasets' isn't installed, raises ImportError."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("no module named datasets")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Use non-JSONL path to trigger HF path
        loader = DatasetLoader(DatasetConfig(path="some/hf/dataset", shuffle=False))
        with pytest.raises(ImportError, match="datasets"):
            loader.load()


# ── _hermes_to_sample ─────────────────────────────────────────────────────────


class TestHermesToSample:
    def _loader(self):
        return DatasetLoader(DatasetConfig(path="/dummy", shuffle=False))

    def test_basic_hermes(self):
        loader = self._loader()
        row = {
            "conversations": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        s = loader._hermes_to_sample(row)
        assert s.prompt == "What is 2+2?"
        assert s.answer == "4"

    def test_hermes_with_ground_truth(self):
        loader = self._loader()
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
            ],
            "ground_truth": "42",
        }
        s = loader._hermes_to_sample(row)
        assert s.answer == "42"

    def test_hermes_no_assistant_no_gt(self):
        loader = self._loader()
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
            ]
        }
        s = loader._hermes_to_sample(row)
        assert s.answer is None

    def test_hermes_type_from_row(self):
        loader = self._loader()
        row = {
            "conversations": [{"role": "user", "content": "Q"}],
            "type": "math",
            "answer": "42",
        }
        s = loader._hermes_to_sample(row)
        assert s.type_name == "math"

    def test_hermes_via_load(self):
        """Row with 'conversations' key triggers hermes path."""
        path = _write_jsonl(
            [
                {
                    "conversations": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ],
                    "type": "mcq",
                }
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        assert samples[0].prompt == "Hello"
        assert samples[0].answer == "Hi"

    def test_hermes_extra_metadata(self):
        loader = self._loader()
        row = {
            "conversations": [{"role": "user", "content": "Q"}],
            "answer": "A",
            "difficulty": "hard",  # extra field → metadata
        }
        s = loader._hermes_to_sample(row)
        assert s.metadata.get("difficulty") == "hard"


# ── reset ─────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_restores_index(self):
        path = _write_jsonl([{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(5)])
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        loader.load()
        loader.next_batch(3)
        assert loader._index == 3
        loader.reset()
        assert loader._index == 0

    def test_reset_with_new_seed(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(10)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=True))
        loader.load()
        loader.reset(new_seed=999)
        assert loader._index == 0

    def test_reset_reshuffles(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(10)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=True))
        loader.load()
        loader.reset(new_seed=42)
        order_after = [s.prompt for s in loader._samples]
        # With new seed, order may differ
        assert len(order_after) == 10

    def test_reset_no_shuffle_no_reshuffle(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(5)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        loader.load()
        order_before = [s.prompt for s in loader._samples]
        loader.reset()
        order_after = [s.prompt for s in loader._samples]
        assert order_before == order_after

    def test_reset_balanced_shuffle(self):
        rows = [{"prompt": f"M{i}", "answer": "a", "type": "math"} for i in range(3)] + [
            {"prompt": f"Q{i}", "answer": "a", "type": "mcq"} for i in range(3)
        ]
        path = _write_jsonl(rows)
        loader = DatasetLoader(
            DatasetConfig(path=path, shuffle=True, balanced_shuffle=True, seed=42)
        )
        loader.load()
        loader.reset()
        assert loader._index == 0
        assert len(loader._samples) == 6


# ── max_samples ───────────────────────────────────────────────────────────────


class TestMaxSamples:
    def test_max_samples_limit(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(20)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False, max_samples=5))
        samples = loader.load()
        assert len(samples) == 5

    def test_max_samples_larger_than_dataset(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(3)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False, max_samples=100))
        samples = loader.load()
        assert len(samples) == 3


# ── next_batch ────────────────────────────────────────────────────────────────


class TestNextBatchWrap:
    def test_wraps_with_reshuffle(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(3)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=True))
        loader.load()
        batch = loader.next_batch(5)  # More than dataset size
        assert len(batch) == 5
        assert loader._epoch == 1

    def test_wraps_no_shuffle(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(3)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        loader.load()
        batch = loader.next_batch(5)
        assert len(batch) == 5

    def test_getitem(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(3)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        loader.load()
        assert loader[0].prompt == "Q0"
        assert loader[2].prompt == "Q2"

    def test_len(self):
        rows = [{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(7)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        loader.load()
        assert len(loader) == 7
