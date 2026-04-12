"""Process-based rollout worker — rollout-only, no gradient ops.

Each worker loads its own model copy, receives weight-sync + sample tasks
via multiprocessing queues, and returns WorkerResult to the main process.

Metal + fork is unsafe on macOS; workers MUST be spawned (not forked).
Startup cost: ~10-30s per worker (full model load). Amortized over training.
"""

from __future__ import annotations

import logging
import signal
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# Magic sample_idx used in result_queue to tag GradientResult messages.
_GRADIENT_RESULT_IDX: int = -2


@dataclass
class WorkerTask:
    """Task sent from main process to a worker."""

    sample: Any = None  # Sample | None
    iteration: int = 0
    sample_idx: int = -1  # batch position → result reordering
    weight_update: dict[str, Any] | None = None  # trainable param numpy arrays
    gradient_task: Any | None = None  # GradientTask | None (worker 0 only)
    save_checkpoint: bool = False  # request checkpoint save before exit


@dataclass
class WorkerResult:
    """Result sent from worker to main process. Duck-types SampleGroupResult."""

    sample_idx: int = -1
    group_rollouts: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    advantages: list = field(default_factory=list)
    mean_reward: float = 0.0
    prompts: list = field(default_factory=list)
    answers: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    pending_events: list = field(default_factory=list)
    tui_ctx: Any = None  # always None — TUI is single-process


@dataclass
class GradientTask:
    """Gradient step request sent from pool to gradient worker (worker 0)."""

    rollouts: list = field(default_factory=list)  # list[RolloutResult]
    advantages: list = field(default_factory=list)  # list[float]
    rewards: list = field(default_factory=list)  # list[float]
    iteration: int = 0
    reset_optimizer: bool = False  # Clear optimizer momentum after gradient step


@dataclass
class GradientResult:
    """Gradient step result sent from worker 0 to pool / main."""

    loss: float = 0.0
    kl: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    updated_weights: dict[str, Any] = field(default_factory=dict)  # numpy arrays
    spike_skipped: bool = False
    spike_action: str | None = None
    is_weight_stats: list[dict[str, float]] = field(default_factory=list)
    metal_memory_mb: float = 0.0  # peak Metal GPU memory from gradient worker process


def _build_trainer_kwargs(trainer: Any) -> dict[str, Any]:
    """Extract picklable Trainer init kwargs for worker processes.

    All values are frozen dataclasses, plain Python scalars, or plain dicts
    — all picklable. Model, tokenizer, optimizer excluded (worker loads its own).
    """
    return {
        "max_completion_tokens": trainer.max_completion_tokens,
        "scaffold": trainer._global_scaffold,
        "max_grad_norm": trainer._max_grad_norm,
        "used_tokens": trainer._used_tokens,
        "truncate_from": trainer._truncate_from,
        "truncate_marker": trainer._truncate_marker,
        "truncate_marker_masked": trainer._truncate_marker_masked,
        "snap_to_boundary": trainer._snap_to_boundary,
        "type_overrides": trainer._type_overrides,
        "system_prompt": trainer._system_prompt,
        "is_cfg": trainer._is_cfg,
        "dv_cfg": trainer._dv_cfg,
        "esc_cfg": trainer._esc_cfg,
        "gen_defaults": dict(trainer._gen_defaults),
    }


def _apply_weights(model: Any, weights: dict[str, Any]) -> float | None:  # pragma: no cover
    """Apply numpy weight arrays to an MLX model. Returns reward_baseline if present."""
    baseline: float | None = None
    _rb = weights.get("__reward_baseline__")
    if _rb is not None:
        baseline = float(_rb[0])
    try:
        import mlx.core as mx

        weight_list = [(k, mx.array(v)) for k, v in weights.items() if k != "__reward_baseline__"]
        if weight_list:
            model.load_weights(weight_list, strict=False)
            mx.eval(model.parameters())
    except Exception as exc:  # pragma: no cover
        _log.warning("Weight sync failed: %s", exc)
    return baseline


def worker_main(  # pragma: no cover
    model_path: str,
    cfg: Any,
    yaml_path: str | None,
    trainer_kwargs: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Entry point for rollout-only worker processes (workers 1..N-1).

    Loads base model, no optimizer, no gradient ops.
    Processes weight-sync tasks and rollout tasks only.
    """
    try:
        import mlx_grpo.rewards.accuracy  # noqa: F401
        import mlx_grpo.types.builtins  # noqa: F401

        if yaml_path:
            from mlx_grpo.config.loader import load_config

            load_config(yaml_path)

        from mlx_lm import load as _mlx_load

        model, tokenizer = _mlx_load(model_path)

        from mlx_grpo.backends.base import BackendRegistry

        backend = BackendRegistry.get("mlx", model=model, tokenizer=tokenizer)

        from dataclasses import replace as _replace

        from mlx_grpo.training import Trainer

        worker_cfg = _replace(cfg, rollout_worker_mode="thread", rollout_workers=1)
        trainer: Any = Trainer(
            config=worker_cfg,
            dataset=None,  # type: ignore[arg-type]
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            **trainer_kwargs,
        )
        trainer._adaptive = None

        result_queue.put(("ready", None))
        # Ignore SIGINT — main process handles interrupt and calls pool.shutdown()
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    except Exception as exc:
        result_queue.put(("startup_error", str(exc)))
        return

    while True:
        try:
            task: WorkerTask | None = task_queue.get()
        except KeyboardInterrupt:
            break
        if task is None:
            break
        try:
            # Save checkpoint: rollout-only workers have no model state to save
            if task.save_checkpoint:
                continue

            if task.weight_update is not None and task.sample is None:
                baseline = _apply_weights(model, task.weight_update)
                if baseline is not None:
                    trainer._reward_baseline = baseline
                continue

            if task.sample is None:
                continue

            sgr = trainer._run_sample_group(task.sample, task.iteration)
            if sgr is None:
                result_queue.put((task.sample_idx, None))
            else:
                result_queue.put(
                    (
                        task.sample_idx,
                        WorkerResult(
                            sample_idx=task.sample_idx,
                            group_rollouts=sgr.group_rollouts,
                            rewards=sgr.rewards,
                            advantages=sgr.advantages,
                            mean_reward=sgr.mean_reward,
                            prompts=sgr.prompts,
                            answers=sgr.answers,
                            targets=sgr.targets,
                            pending_events=sgr.pending_events,
                            tui_ctx=None,
                        ),
                    )
                )
        except KeyboardInterrupt:
            break
        except Exception as exc:
            _log.warning("Rollout worker error on sample_idx=%d: %s", task.sample_idx, exc)
            result_queue.put((task.sample_idx, None))
