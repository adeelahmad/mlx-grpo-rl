"""Tests for VLM image fields on Sample and DatasetLoader (Phase 1, Task 1.3)."""

from __future__ import annotations

import json
from pathlib import Path

from mlx_grpo.data import DatasetConfig, DatasetLoader, Sample

# ── Task 1.1: Sample dataclass image fields ──────────────────────────────────


def test_sample_image_fields_default_none():
    """Sample with no image args has image_path=None and image_b64=None."""
    s = Sample(prompt="What is 2+2?", answer="4")
    assert s.image_path is None
    assert s.image_b64 is None


def test_sample_with_image_path():
    """Sample(image_path=...) stores the path correctly."""
    s = Sample(prompt="Describe this image.", answer="A cat.", image_path="images/cat.jpg")
    assert s.image_path == "images/cat.jpg"
    assert s.image_b64 is None


def test_sample_with_image_b64():
    """Sample(image_b64=...) stores the base64 string correctly."""
    # Minimal valid 1×1 PNG in base64 (truncated for brevity)
    fake_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    s = Sample(prompt="What colour?", answer="Red.", image_b64=fake_b64)
    assert s.image_b64 == fake_b64
    assert s.image_path is None


# ── Task 1.2: DatasetLoader image field extraction ────────────────────────────


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows to a temporary JSONL file and return the path."""
    f = tmp_path / "data.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    return f


def test_dataloader_text_only_unchanged(tmp_path):
    """Loading a JSONL without image columns still works (backward compat)."""
    rows = [
        {"prompt": "Q1", "answer": "A1"},
        {"prompt": "Q2", "answer": "A2"},
    ]
    f = _write_jsonl(tmp_path, rows)
    cfg = DatasetConfig(path=str(f), shuffle=False)
    loader = DatasetLoader(cfg).load()
    assert len(loader) == 2
    for sample in loader:
        assert sample.image_path is None
        assert sample.image_b64 is None


def test_dataloader_with_image_path(tmp_path):
    """JSONL rows with image_path populate sample.image_path."""
    # Use a URL so the loader doesn't warn about missing local file
    rows = [
        {"prompt": "Describe", "answer": "A cat", "image_path": "https://example.com/cat.jpg"},
        {"prompt": "Describe", "answer": "A dog", "image_path": "https://example.com/dog.jpg"},
        {"prompt": "No image", "answer": "Text only"},
    ]
    f = _write_jsonl(tmp_path, rows)
    cfg = DatasetConfig(path=str(f), shuffle=False)
    loader = DatasetLoader(cfg).load()
    assert len(loader) == 3
    assert loader[0].image_path == "https://example.com/cat.jpg"
    assert loader[1].image_path == "https://example.com/dog.jpg"
    assert loader[2].image_path is None


def test_dataloader_with_image_b64(tmp_path):
    """JSONL rows with image_b64 populate sample.image_b64."""
    fake_b64 = "abc123=="
    rows = [
        {"prompt": "What colour?", "answer": "Red", "image_b64": fake_b64},
    ]
    f = _write_jsonl(tmp_path, rows)
    cfg = DatasetConfig(path=str(f), shuffle=False)
    loader = DatasetLoader(cfg).load()
    assert len(loader) == 1
    assert loader[0].image_b64 == fake_b64
    assert loader[0].image_path is None


def test_dataloader_mixed_image_text(tmp_path):
    """Dataset with some image rows and some text-only rows loads correctly."""
    fake_b64 = "ZGF0YQ=="
    rows = [
        {"prompt": "Q1", "answer": "A1"},
        {"prompt": "Q2", "answer": "A2", "image_path": "https://example.com/img.jpg"},
        {"prompt": "Q3", "answer": "A3", "image_b64": fake_b64},
        {"prompt": "Q4", "answer": "A4"},
    ]
    f = _write_jsonl(tmp_path, rows)
    cfg = DatasetConfig(path=str(f), shuffle=False)
    loader = DatasetLoader(cfg).load()
    assert len(loader) == 4

    assert loader[0].image_path is None
    assert loader[0].image_b64 is None

    assert loader[1].image_path == "https://example.com/img.jpg"
    assert loader[1].image_b64 is None

    assert loader[2].image_path is None
    assert loader[2].image_b64 == fake_b64

    assert loader[3].image_path is None
    assert loader[3].image_b64 is None


def test_dataloader_image_path_not_in_metadata(tmp_path):
    """image_path and image_b64 are NOT included in sample.metadata."""
    rows = [
        {
            "prompt": "Q",
            "answer": "A",
            "image_path": "https://example.com/x.jpg",
            "image_b64": "abc==",
            "extra": "value",
        }
    ]
    f = _write_jsonl(tmp_path, rows)
    cfg = DatasetConfig(path=str(f), shuffle=False)
    loader = DatasetLoader(cfg).load()
    sample = loader[0]
    # image_path / image_b64 should be proper fields, not in metadata
    assert "image_path" not in sample.metadata
    assert "image_b64" not in sample.metadata
    # Other extra fields still land in metadata
    assert sample.metadata.get("extra") == "value"
