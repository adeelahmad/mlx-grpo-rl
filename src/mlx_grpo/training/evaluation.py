"""Evaluation pass — periodic model assessment during training.

Runs the model on a held-out eval set with greedy decoding
(temperature=0) to track training progress objectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..data import DatasetLoader, Sample
from ..engine.callbacks import CallbackRegistry
from ..engine.context import BatchContext
from ..engine.executor import RolloutExecutor
from ..types.base import TaskType


@dataclass(slots=True)
class EvalResult:
    """Result of an evaluation pass."""

    mean_reward: float = 0.0
    per_type_reward: dict[str, float] = field(default_factory=dict)
    per_type_count: dict[str, int] = field(default_factory=dict)
    total_samples: int = 0
    total_tokens: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)


class Evaluator:
    """Runs evaluation passes during training.

    Uses greedy decoding (temperature=0) for deterministic results.
    Supports per-type metric tracking.

    Usage:
        evaluator = Evaluator(executor, eval_dataset)
        result = evaluator.evaluate(iteration=100, max_samples=50)
    """

    def __init__(
        self,
        executor: RolloutExecutor,
        eval_dataset: DatasetLoader | None = None,
        eval_samples: list[Sample] | None = None,
        type_overrides: dict[str, dict[str, Any]] | None = None,
        system_prompt: str = "",
    ):
        self.executor = executor
        self.eval_dataset = eval_dataset
        self._static_samples = eval_samples
        self._type_overrides = type_overrides or {}
        self._system_prompt = system_prompt

    def evaluate(
        self,
        iteration: int = 0,
        max_samples: int = 50,
        temperature: float = 0.0,
    ) -> EvalResult:
        """Run evaluation on eval set.

        Args:
            iteration: Current training iteration
            max_samples: Maximum samples to evaluate
            temperature: Generation temperature (0 for greedy)
        """
        samples = self._get_samples(max_samples)
        if not samples:
            return EvalResult()

        rewards_by_type: dict[str, list[float]] = {}
        all_results: list[dict[str, Any]] = []
        total_tokens = 0

        for sample in samples:
            try:
                task_type = TaskType.get(sample.type_name)
            except KeyError:
                continue

            # Build eval context
            batch_ctx = BatchContext(
                iteration=iteration,
                total_iters=1,
                group_index=0,
                group_size=1,
            )

            cb = CallbackRegistry(seed=iteration)
            custom = task_type.callbacks()
            if custom:
                cb.register_many(custom)

            # Resolve per-type system prompt
            overrides = self._type_overrides.get(sample.type_name, {})
            merged = {**task_type.config(), **overrides} if overrides else task_type.config()
            system_prompt = (
                sample.metadata.get("system_prompt")
                or merged.get("system_prompt")
                or task_type.chat_params().system_prompt
                or self._system_prompt
                or "You are a helpful assistant."
            )

            # Execute rollout with low/zero temperature
            result = self.executor.execute(
                rollout_fn=task_type.rollout,
                prompt=sample.prompt,
                answer=sample.answer,
                type_name=sample.type_name,
                type_config={**merged, "temperature": temperature},
                batch=batch_ctx,
                callbacks=cb,
            )
            result.system_message = system_prompt

            # Score
            reward = task_type.reward(result, sample.answer)
            rewards_by_type.setdefault(sample.type_name, []).append(reward)
            total_tokens += result.generated_token_count

            all_results.append(
                {
                    "type": sample.type_name,
                    "reward": reward,
                    "prompt": sample.prompt[:100],
                    "completion": result.completion_text[:200],
                }
            )

        # Aggregate
        per_type_reward = {t: sum(rs) / len(rs) for t, rs in rewards_by_type.items()}
        per_type_count = {t: len(rs) for t, rs in rewards_by_type.items()}
        all_rewards = [r for rs in rewards_by_type.values() for r in rs]
        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)

        return EvalResult(
            mean_reward=mean_reward,
            per_type_reward=per_type_reward,
            per_type_count=per_type_count,
            total_samples=len(all_results),
            total_tokens=total_tokens,
            results=all_results,
        )

    def _get_samples(self, max_samples: int) -> list[Sample]:
        """Get evaluation samples."""
        if self._static_samples:
            return self._static_samples[:max_samples]
        if self.eval_dataset:
            return self.eval_dataset.next_batch(max_samples)
        return []


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None
# Usage: evaluator = Evaluator(executor, eval_dataset=ds)
#        result = evaluator.evaluate(max_samples=50)
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
