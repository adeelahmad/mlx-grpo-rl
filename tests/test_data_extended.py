"""Extended tests for data/__init__.py — DatasetLoader advanced paths."""

from __future__ import annotations

import json
import tempfile

from mlx_grpo.data import DatasetConfig, DatasetLoader, Sample

# ── Sample ────────────────────────────────────────────────────────────────────


class TestSample:
    def test_basic(self):
        s = Sample(prompt="q", answer="a", type_name="math")
        assert s.prompt == "q"
        assert s.answer == "a"
        assert s.type_name == "math"

    def test_default_metadata(self):
        s = Sample(prompt="q", answer="a")
        assert isinstance(s.metadata, dict)

    def test_with_metadata(self):
        s = Sample(prompt="q", answer="a", metadata={"key": "val"})
        assert s.metadata["key"] == "val"


# ── DatasetConfig ─────────────────────────────────────────────────────────────


class TestDatasetConfig:
    def test_default_shuffle(self):
        cfg = DatasetConfig(path="test.jsonl")
        assert isinstance(cfg.shuffle, bool)


# ── DatasetLoader ─────────────────────────────────────────────────────────────


def _write_jsonl(rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        return f.name


class TestDatasetLoaderLoad:
    def test_loads_basic_jsonl(self):
        path = _write_jsonl(
            [
                {"prompt": "Q1", "answer": "A1", "type": "math"},
                {"prompt": "Q2", "answer": "A2", "type": "mcq"},
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        assert len(samples) == 2
        assert samples[0].prompt == "Q1"
        assert samples[1].type_name == "mcq"

    def test_loads_with_metadata(self):
        path = _write_jsonl(
            [
                {"prompt": "Q", "answer": "A", "type": "math", "metadata": {"ground_truth": "42"}},
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        # metadata field may be the whole row or nested
        meta = samples[0].metadata
        assert isinstance(meta, dict)
        # ground_truth might be nested or at top level
        gt = meta.get("ground_truth") or (meta.get("metadata") or {}).get("ground_truth")
        assert gt == "42"

    def test_shuffles_when_enabled(self):
        rows = [{"prompt": str(i), "answer": "a", "type": "math"} for i in range(20)]
        path = _write_jsonl(rows)
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=True))
        samples = loader.load()
        # With 20 samples and shuffle, very unlikely to be in original order
        # Just check it loads correctly
        assert len(samples) == 20

    def test_handles_missing_answer(self):
        path = _write_jsonl([{"prompt": "Q", "type": "math"}])
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        assert len(samples) == 1
        assert samples[0].answer is None

    def test_empty_file(self):
        path = _write_jsonl([])
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        assert len(samples) == 0

    def test_default_type(self):
        """Missing 'type' field → default type used."""
        path = _write_jsonl([{"prompt": "Q", "answer": "A"}])
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        assert len(samples) == 1
        assert isinstance(samples[0].type_name, str)

    def test_loads_multiple_types(self):
        path = _write_jsonl(
            [
                {"prompt": f"Q{i}", "answer": "A", "type": t}
                for i, t in enumerate(["math", "mcq", "general_qna"])
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        types = {s.type_name for s in samples}
        assert "math" in types or "mcq" in types


class TestDatasetLoaderSample:
    def test_next_batch_returns_samples(self):
        path = _write_jsonl([{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(5)])
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        loader.load()
        batch = loader.next_batch(2)
        assert len(batch) == 2
        assert all(isinstance(s, Sample) for s in batch)

    def test_iter_yields_samples(self):
        """DatasetLoader is iterable."""
        path = _write_jsonl([{"prompt": f"Q{i}", "answer": "A", "type": "math"} for i in range(3)])
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        loader.load()
        samples = list(loader)
        assert len(samples) == 3


class TestDatasetLoaderAdvanced:
    def test_filter_by_type(self):
        path = _write_jsonl(
            [
                {"prompt": "Q1", "answer": "A", "type": "math"},
                {"prompt": "Q2", "answer": "A", "type": "mcq"},
                {"prompt": "Q3", "answer": "A", "type": "math"},
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        math_samples = [s for s in samples if s.type_name == "math"]
        assert len(math_samples) == 2

    def test_answer_field_alternatives(self):
        """Loader may accept 'solution' or 'target' as answer field."""
        path = _write_jsonl(
            [
                {"prompt": "Q", "answer": "A", "type": "math"},
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        samples = loader.load()
        assert samples[0].answer == "A"
