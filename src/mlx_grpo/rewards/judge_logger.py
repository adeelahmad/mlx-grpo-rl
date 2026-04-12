"""Judge call logger — raw JSONL + aggregate stats for CSV/WandB.

Writes one JSONL entry per LLM judge API call (or cache hit) for
post-hoc analysis. Exposes aggregate stats for integration with
StatsLogger (CSV) and WandbLogger.

Config::

    reward_kwargs:
      judge_log_path: "checkpoints/judge_calls.jsonl"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class JudgeCallRecord:
    """One judge API call (or cache hit)."""

    timestamp: float
    ground_truth: str
    predicted: str
    problem: str
    correct: bool | None = None
    confidence: float | None = None
    extracted_answer: str = ""
    reward: float = 0.0
    cache_hit: bool = False
    latency_ms: float = 0.0
    fallback: bool = False
    error: str = ""


class JudgeLogger:
    """Append raw judge calls to JSONL and track aggregate stats."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a")  # noqa: SIM115
        # Running aggregates for current step
        self._step_calls = 0
        self._step_hits = 0
        self._step_failures = 0
        self._step_confidences: list[float] = []
        self._step_correct = 0
        self._step_latency_ms: list[float] = []

    def log(self, record: JudgeCallRecord) -> None:
        """Write one record and update step aggregates."""
        self._step_calls += 1
        if record.cache_hit:
            self._step_hits += 1
        if record.fallback:
            self._step_failures += 1
        if record.confidence is not None:
            self._step_confidences.append(record.confidence)
        if record.correct:
            self._step_correct += 1
        self._step_latency_ms.append(record.latency_ms)

        entry: dict[str, Any] = {
            "ts": record.timestamp,
            "gt": record.ground_truth,
            "pred": record.predicted,
            "problem": record.problem[:200],
            "correct": record.correct,
            "confidence": record.confidence,
            "extracted": record.extracted_answer,
            "reward": round(record.reward, 4),
            "cache_hit": record.cache_hit,
            "latency_ms": round(record.latency_ms, 1),
            "fallback": record.fallback,
        }
        if record.error:
            entry["error"] = record.error
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def step_stats(self) -> dict[str, float]:
        """Return aggregate stats for current step, then reset."""
        confs = self._step_confidences
        lats = self._step_latency_ms
        stats = {
            "judge_calls": self._step_calls,
            "judge_cache_hits": self._step_hits,
            "judge_failures": self._step_failures,
            "judge_correct": self._step_correct,
            "judge_mean_confidence": sum(confs) / len(confs) if confs else 0.0,
            "judge_mean_latency_ms": sum(lats) / len(lats) if lats else 0.0,
        }
        self._step_calls = 0
        self._step_hits = 0
        self._step_failures = 0
        self._step_confidences = []
        self._step_correct = 0
        self._step_latency_ms = []
        return stats

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
