"""Scaffold-stratified accuracy probe — measures true model capability.

Runs the same eval samples at fixed scaffold ratios (0%, 25%, 50%, 75%, 100%)
to separate genuine learning from scaffold dependence. The key metric is
scaffold=0.0 (unassisted accuracy) — what the model can do on its own.

    probe = ScaffoldProbe(executor, eval_dataset=ds)
    result = probe.run(iteration=50)
    print(result.unassisted_accuracy)  # scaffold=0.0 accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data import DatasetLoader, Sample
from ..engine.callbacks import CallbackRegistry
from ..engine.context import BatchContext
from ..engine.executor import RolloutExecutor
from ..scaffold import ScaffoldConfig
from ..types.base import TaskType


def _metal_clear_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


@dataclass(slots=True, frozen=True)
class ScaffoldProbeResult:
    """Result of a scaffold-stratified evaluation."""

    per_ratio_accuracy: dict[float, float] = field(default_factory=dict)
    per_ratio_reward_std: dict[float, float] = field(default_factory=dict)
    per_ratio_count: dict[float, int] = field(default_factory=dict)
    per_ratio_by_type: dict[float, dict[str, float]] = field(default_factory=dict)
    unassisted_accuracy: float = 0.0
    total_samples: int = 0
    total_tokens: int = 0


class ScaffoldProbe:
    """Probes model accuracy at fixed scaffold ratios.

    At each eval interval, runs the same samples at multiple fixed
    scaffold levels to track how much the model relies on scaffolding
    vs. genuine learned capability.
    """

    def __init__(
        self,
        executor: RolloutExecutor,
        eval_dataset: DatasetLoader | None = None,
        train_dataset: DatasetLoader | None = None,
        ratios: list[float] | None = None,
        samples_per_ratio: int = 20,
        scaffold_target: str = "both",
    ) -> None:
        self.executor = executor
        self._eval_dataset = eval_dataset
        self._train_dataset = train_dataset
        self.ratios = ratios or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.samples_per_ratio = samples_per_ratio
        self.scaffold_target = scaffold_target

    def run(self, iteration: int = 0) -> ScaffoldProbeResult:
        """Run scaffold-stratified evaluation."""
        samples = self._get_samples()
        if not samples:
            return ScaffoldProbeResult()

        per_ratio_acc: dict[float, float] = {}
        per_ratio_std: dict[float, float] = {}
        per_ratio_count: dict[float, int] = {}
        per_ratio_by_type: dict[float, dict[str, float]] = {}
        total_tokens = 0

        for ratio in self.ratios:
            rewards: list[float] = []
            type_rewards: dict[str, list[float]] = {}

            for sample in samples:
                try:
                    task_type = TaskType.get(sample.type_name)
                except KeyError:
                    continue

                batch_ctx = BatchContext(
                    scaffold=ScaffoldConfig(
                        initial_ratio=ratio,
                        decay_rate=0.0,
                        end_progress=1.0,
                        target=self.scaffold_target,
                    ),
                    iteration=0,
                    total_iters=1,
                    group_index=0,
                    group_size=1,
                )
                cb = CallbackRegistry(seed=iteration)
                custom = task_type.callbacks()
                if custom:
                    cb.register_many(custom)

                result = self.executor.execute(
                    rollout_fn=task_type.rollout,
                    prompt=sample.prompt,
                    answer=sample.answer,
                    type_name=sample.type_name,
                    type_config={**task_type.config(), "temperature": 0.0},
                    batch=batch_ctx,
                    callbacks=cb,
                )

                ground_truth = sample.metadata.get("ground_truth", sample.answer)
                reward = task_type.reward(result, ground_truth)
                rewards.append(reward)
                total_tokens += result.generated_token_count
                type_rewards.setdefault(sample.type_name, []).append(reward)

            n = len(rewards)
            mean = sum(rewards) / max(n, 1)
            std = (sum((r - mean) ** 2 for r in rewards) / max(n, 1)) ** 0.5
            per_ratio_acc[ratio] = mean
            per_ratio_std[ratio] = std
            per_ratio_count[ratio] = n
            per_ratio_by_type[ratio] = {t: sum(rs) / len(rs) for t, rs in type_rewards.items()}
            _metal_clear_cache()

        unassisted = per_ratio_acc.get(0.0, 0.0)

        return ScaffoldProbeResult(
            per_ratio_accuracy=per_ratio_acc,
            per_ratio_reward_std=per_ratio_std,
            per_ratio_count=per_ratio_count,
            per_ratio_by_type=per_ratio_by_type,
            unassisted_accuracy=unassisted,
            total_samples=sum(per_ratio_count.values()),
            total_tokens=total_tokens,
        )

    def _get_samples(self) -> list[Sample]:
        """Get evaluation samples, preferring eval_dataset."""
        n = self.samples_per_ratio
        if self._eval_dataset and len(self._eval_dataset) > 0:
            return list(self._eval_dataset)[:n]
        if self._train_dataset and len(self._train_dataset) > 0:
            return list(self._train_dataset)[:n]
        return []
