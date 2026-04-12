"""Tests for the validate-data subcommand and validator module."""

from __future__ import annotations

import json
from pathlib import Path

from mlx_grpo.data import Sample
from mlx_grpo.data.validator import (
    format_report,
    save_valid_jsonl,
    validate_dataset,
    validate_sample,
)

# ── validate_sample ──────────────────────────────────────────────────


def test_valid_mcq_sample():
    s = Sample(prompt="What is 2+2?", answer="A", type_name="mcq")
    r = validate_sample(0, s)
    assert r.valid is True
    assert r.max_reward == 1.0
    assert len(r.issues) == 0


def test_empty_prompt():
    s = Sample(prompt="", answer="A", type_name="mcq")
    r = validate_sample(0, s)
    assert r.valid is False
    assert any(i.field == "prompt" and i.severity == "error" for i in r.issues)


def test_whitespace_prompt():
    s = Sample(prompt="   ", answer="A", type_name="mcq")
    r = validate_sample(0, s)
    assert r.valid is False


def test_unknown_type():
    s = Sample(prompt="hello", answer="world", type_name="nonexistent_type")
    r = validate_sample(0, s)
    assert r.valid is False
    assert any(i.field == "type_name" and i.severity == "error" for i in r.issues)


def test_alias_resolution():
    """'exam' alias should resolve to 'mcq'."""
    s = Sample(prompt="Pick the best answer", answer="B", type_name="exam")
    r = validate_sample(0, s)
    assert r.valid is True
    assert r.max_reward == 1.0


def test_math_no_answer():
    s = Sample(prompt="Solve x^2=4", answer=None, type_name="math")
    r = validate_sample(0, s)
    assert r.valid is True  # warning, not error
    assert r.max_reward == 0.0
    assert any(i.severity == "warning" for i in r.issues)


def test_math_with_answer():
    s = Sample(prompt="Solve x^2=4", answer="2", type_name="math")
    r = validate_sample(0, s)
    assert r.valid is True
    assert r.max_reward == 1.0
    assert len(r.issues) == 0


def test_tool_call_bad_json_answer():
    s = Sample(prompt="Call the weather API", answer="not json", type_name="tool_call")
    r = validate_sample(0, s)
    assert r.valid is True  # warning, not error
    assert r.max_reward == 0.5


def test_tool_call_valid_answer():
    answer = json.dumps({"name": "get_weather", "arguments": {"city": "Melbourne"}})
    s = Sample(prompt="Call the weather API", answer=answer, type_name="tool_call")
    r = validate_sample(0, s)
    assert r.valid is True
    assert r.max_reward == 1.0


def test_lotto_no_answer():
    s = Sample(prompt="Pick lotto numbers", answer=None, type_name="lotto")
    r = validate_sample(0, s)
    assert r.valid is False
    assert any(i.severity == "error" for i in r.issues)


def test_general_qna_no_answer():
    s = Sample(prompt="Explain gravity", answer=None, type_name="general_qna")
    r = validate_sample(0, s)
    assert r.valid is True
    assert r.max_reward == 0.3


def test_python_code_no_answer():
    s = Sample(prompt="Write hello world", answer=None, type_name="python_code")
    r = validate_sample(0, s)
    assert r.valid is True
    assert r.max_reward == 0.5


def test_python_verified_no_answer():
    s = Sample(prompt="Write a function", answer=None, type_name="python_verified")
    r = validate_sample(0, s)
    assert r.valid is True
    assert r.max_reward == 0.9


# ── validate_dataset ─────────────────────────────────────────────────


def test_dataset_counts():
    samples = [
        Sample(prompt="Q1", answer="A", type_name="mcq"),
        Sample(prompt="", answer="A", type_name="mcq"),  # invalid: empty prompt
        Sample(prompt="Q3", answer=None, type_name="lotto"),  # invalid: lotto needs answer
        Sample(prompt="Q4", answer="42", type_name="math"),  # valid
    ]
    report = validate_dataset("test", samples)
    assert report.total_rows == 4
    assert report.valid_rows == 2
    assert report.invalid_rows == 2
    assert "mcq" in report.per_type
    assert "math" in report.per_type


def test_dataset_per_type_stats():
    samples = [
        Sample(prompt="Q1", answer="A", type_name="mcq"),
        Sample(prompt="Q2", answer="B", type_name="mcq"),
        Sample(prompt="Q3", answer=None, type_name="mcq"),  # warning
    ]
    report = validate_dataset("test", samples)
    mcq = report.per_type["mcq"]
    assert mcq.count == 3
    assert mcq.valid == 3
    assert mcq.invalid == 0
    assert mcq.warnings == 1


# ── save_valid_jsonl ─────────────────────────────────────────────────


def test_save_valid_jsonl(tmp_path: Path):
    samples = [
        Sample(prompt="Q1", answer="A", type_name="mcq"),
        Sample(prompt="", answer="A", type_name="mcq"),  # invalid
        Sample(prompt="Q3", answer="42", type_name="math"),
    ]
    report = validate_dataset("test", samples)
    counts = save_valid_jsonl(report, tmp_path / "clean")

    assert counts == {"mcq": 1, "math": 1}
    assert (tmp_path / "clean" / "mcq.jsonl").exists()
    assert (tmp_path / "clean" / "math.jsonl").exists()

    with open(tmp_path / "clean" / "mcq.jsonl") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 1
    assert rows[0]["prompt"] == "Q1"


# ── format_report ────────────────────────────────────────────────────


def test_format_report_output():
    samples = [Sample(prompt="Q1", answer="A", type_name="mcq")]
    report = validate_dataset("demo", samples)
    text = format_report(report)
    assert "demo" in text
    assert "Valid: 1" in text
    assert "mcq" in text


def test_format_report_verbose():
    samples = [Sample(prompt="Q1", answer=None, type_name="math")]
    report = validate_dataset("demo", samples)
    text = format_report(report, verbose=True)
    assert "WRN" in text
    assert "reward will be 0.0" in text


# ── CLI integration ──────────────────────────────────────────────────


def test_cli_validate_data(tmp_path: Path):
    """End-to-end: write a config + dataset, run validate-data via main()."""
    # Write dataset
    ds_path = tmp_path / "data.jsonl"
    rows = [
        {"prompt": "What is 1+1?", "answer": "A", "type": "mcq"},
        {"prompt": "", "answer": "B", "type": "mcq"},
    ]
    with open(ds_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Write config
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"training:\n  model_path: test\n  iterations: 1\n"
        f"datasets:\n  test:\n    path: {ds_path}\n    type: mcq\n"
    )

    from mlx_grpo.cli import main

    rc = main(["validate-data", "--config", str(cfg_path)])
    assert rc == 1  # one invalid row
