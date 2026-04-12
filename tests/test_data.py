"""Tests for dataset loading."""

import json
import tempfile

import pytest

from mlx_grpo.data import DatasetConfig, DatasetLoader, Sample


class TestSample:
    def test_defaults(self):
        s = Sample(prompt="hello")
        assert s.answer is None
        assert s.type_name == "mcq"
        assert s.metadata == {}


class TestLocalDataset:
    def _write_jsonl(self, samples: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            f.flush()
            return f.name

    def test_load_jsonl(self):
        path = self._write_jsonl(
            [
                {"prompt": "Q1", "answer": "A"},
                {"prompt": "Q2", "answer": "B"},
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()
        assert len(loader) == 2
        assert loader[0].prompt == "Q1"
        assert loader[1].answer == "B"

    def test_custom_fields(self):
        path = self._write_jsonl(
            [
                {"question": "What?", "correct": "42", "topic": "math"},
            ]
        )
        loader = DatasetLoader(
            DatasetConfig(
                path=path,
                prompt_field="question",
                answer_field="correct",
                shuffle=False,
            )
        ).load()
        assert loader[0].prompt == "What?"
        assert loader[0].answer == "42"
        assert loader[0].metadata["topic"] == "math"

    def test_shuffle_seeded(self):
        data = [{"prompt": f"Q{i}", "answer": str(i)} for i in range(20)]
        path = self._write_jsonl(data)

        l1 = DatasetLoader(DatasetConfig(path=path, seed=42)).load()
        l2 = DatasetLoader(DatasetConfig(path=path, seed=42)).load()
        assert [s.prompt for s in l1] == [s.prompt for s in l2]

    def test_max_samples(self):
        data = [{"prompt": f"Q{i}", "answer": str(i)} for i in range(100)]
        path = self._write_jsonl(data)
        loader = DatasetLoader(DatasetConfig(path=path, max_samples=5, shuffle=False)).load()
        assert len(loader) == 5

    def test_next_batch_wraps(self):
        path = self._write_jsonl(
            [
                {"prompt": "Q1", "answer": "A"},
                {"prompt": "Q2", "answer": "B"},
            ]
        )
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()
        batch1 = loader.next_batch(2)
        assert len(batch1) == 2
        batch2 = loader.next_batch(2)  # Should wrap
        assert len(batch2) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DatasetLoader(DatasetConfig(path="/nonexistent.jsonl")).load()

    def test_json_array(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"prompt": "Q1", "answer": "A"}], f)
            f.flush()
            path = f.name
        loader = DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()
        assert len(loader) == 1
        assert loader[0].prompt == "Q1"
