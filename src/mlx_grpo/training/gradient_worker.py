"""Gradient worker process — worker 0 in process pool mode.

Owns the policy model + optimizer. Handles rollout generation AND gradient
computation, checkpointing, and spike guard. Main process holds no model.

Total model copies = N (not N+1): 1 gradient worker + (N-1) rollout workers.
"""

from __future__ import annotations

import logging
import signal
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def gradient_worker_main(  # pragma: no cover
    model_path: str,
    full_config: Any,  # full Config object
    yaml_path: str | None,
    trainer_kwargs: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
    ckpt_dir: str,
) -> None:
    """Entry point for gradient worker (worker 0).

    Sets up full training pipeline: model + LoRA adapters + optimizer +
    CheckpointManager + SpikeGuard. Handles rollout tasks and GradientTasks.
    Exports initial weights in the ready signal so rollout workers can sync.
    """
    try:
        import mlx_grpo.rewards.accuracy  # noqa: F401
        import mlx_grpo.types.builtins  # noqa: F401

        if yaml_path:
            from mlx_grpo.config.loader import load_config

            load_config(yaml_path)

        cfg = full_config.training
        gcfg = full_config.generation

        # ── Load model with adapters (same as CLI) ──
        from mlx_grpo.training.model_loader import (
            AdapterConfig,
            AdapterType,
            ModelConfig,
            ModelLoader,
        )

        adapter_type_str = getattr(cfg, "adapter_type", "lora")
        adapter_cfg = AdapterConfig(
            rank=getattr(cfg, "lora_rank", 8),
            alpha=getattr(cfg, "lora_alpha", 16.0),
            target_modules=list(
                getattr(cfg, "lora_targets", ["self_attn.q_proj", "self_attn.v_proj"])
            ),
            layers=getattr(cfg, "lora_layers", None),
        )
        _ckpt = Path(ckpt_dir)
        _adapter_path = str(_ckpt) if (_ckpt / "adapters.safetensors").exists() else None
        model_cfg = ModelConfig(
            model_path=model_path,
            adapter_path=_adapter_path,
            adapter_type=adapter_type_str,
            adapter=adapter_cfg,
            freeze_base=(AdapterType.from_str(adapter_type_str) != AdapterType.FULL),
            grad_checkpoint=getattr(cfg, "grad_checkpoint", True),
            moe_cfg=full_config.moe,
        )
        model, ref_model, tokenizer = ModelLoader(model_cfg).load()
        model.eval()

        import mlx.core as mx
        from mlx.utils import tree_flatten as _tf

        _all = [v for _, v in _tf(model.parameters())]
        for _ci in range(0, len(_all), 50):
            mx.eval(*_all[_ci : _ci + 50])
        mx.synchronize()
        mx.clear_cache()

        # ── Backend + optimizer ──
        from mlx_grpo.backends.base import BackendRegistry

        backend = BackendRegistry.get(
            "mlx",
            model=model,
            tokenizer=tokenizer,
            kv_bits=gcfg.kv_bits,
            kv_group_size=gcfg.kv_group_size,
            max_kv_size=gcfg.max_kv_size,
            prefill_step_size=gcfg.prefill_step_size,
            quantized_kv_start=gcfg.quantized_kv_start,
        )
        from mlx_grpo.training.optimizer import build_optimizer

        optimizer = build_optimizer(cfg)

        # ── Trainer (full gradient ops) ──
        from mlx_grpo.training import StepMetrics, Trainer

        worker_cfg = _replace(cfg, rollout_worker_mode="thread", rollout_workers=1)
        trainer: Any = Trainer(
            config=worker_cfg,
            dataset=None,  # type: ignore[arg-type]
            backend=backend,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            **trainer_kwargs,
        )
        trainer._adaptive = None

        # ── CheckpointManager ──
        from mlx_grpo.training.checkpoint import CheckpointManager

        ckpt_mgr = CheckpointManager(
            _ckpt,
            keep_recent=getattr(cfg, "keep_last_n_checkpoints", 5),
            keep_best=getattr(cfg, "keep_best_n_checkpoints", 3),
            save_optimizer=getattr(cfg, "save_optimizer", True),
        )

        def _ckpt_hook(t: Any, m: Any) -> None:
            ckpt_mgr.save(
                m.iteration,
                t.model,
                optimizer=t.optimizer,
                learning_rate=cfg.learning_rate,
                metrics={
                    "mean_reward": m.mean_reward,
                    "loss": m.loss,
                    "kl_divergence": m.kl_divergence,
                },
            )

        trainer.on("on_checkpoint", _ckpt_hook)

        # ── SpikeGuard ──
        if full_config.spike_guard is not None:
            from mlx_grpo.training.spike_guard import SpikeGuard

            trainer.set_spike_guard(
                SpikeGuard(config=full_config.spike_guard, adapter_path=str(_ckpt))
            )

        # Signal ready + export initial weights for rollout workers to sync
        import numpy as np

        initial_weights: dict[str, Any] = {
            n: np.array(p) for n, p in _tf(trainer.model.trainable_parameters())
        }
        initial_weights["__reward_baseline__"] = np.array([trainer._reward_baseline])
        result_queue.put(("ready", initial_weights))
        # Ignore SIGINT — main process handles interrupt and calls pool.shutdown()
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    except Exception as exc:
        result_queue.put(("startup_error", str(exc)))
        return

    from . import StepMetrics
    from .process_worker import (
        _GRADIENT_RESULT_IDX,
        GradientResult,
        WorkerResult,
    )

    _last_gt_iteration: int = 0  # track for forced checkpoint on shutdown

    while True:
        try:
            task: Any = task_queue.get()
        except KeyboardInterrupt:
            break
        if task is None:
            break
        try:
            # Forced checkpoint save (e.g. on Ctrl+C shutdown)
            if task.save_checkpoint:
                _m = StepMetrics(
                    iteration=_last_gt_iteration,
                    mean_reward=0.0,
                    loss=0.0,
                    kl_divergence=0.0,
                )
                trainer._fire("on_checkpoint", _m)
                _log.info("Gradient worker: checkpoint saved at iteration %d", _last_gt_iteration)
                continue

            # Weight-sync: gradient worker already has current weights — ignore
            if task.weight_update is not None and task.sample is None and not task.gradient_task:
                continue

            # Rollout task
            if task.sample is not None:
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

            # Gradient task
            elif task.gradient_task is not None:
                gt = task.gradient_task
                _last_gt_iteration = gt.iteration
                loss, kl, grad_norm = trainer._do_gradient_step(gt.rollouts, gt.advantages)
                # Collect IS weights stamped by gradient step on gt.rollouts
                _is_stats: list[dict[str, float]] = []
                for _r in gt.rollouts:
                    _iws: list[float] = _r.metadata.get("_is_weights") or []
                    if _iws:
                        _is_stats.append(
                            {"mean": sum(_iws) / len(_iws), "min": min(_iws), "max": max(_iws)}
                        )
                if gt.reset_optimizer and trainer.optimizer is not None:
                    # Preserve scalar schedule state (learning_rate, step, etc.) so the
                    # cosine decay schedule keeps its position. Only clear per-parameter
                    # moment buffers (stored as dict values: {"m": ..., "v": ...}).
                    _preserved = {
                        k: v for k, v in trainer.optimizer.state.items() if not isinstance(v, dict)
                    }
                    trainer.optimizer.state = _preserved
                    _log.info("Optimizer state reset at iteration %d", gt.iteration)
                lr = 0.0
                if trainer.optimizer is not None:
                    try:
                        _lr = trainer.optimizer.learning_rate
                        lr = float(_lr.item()) if hasattr(_lr, "item") else float(_lr)
                    except (AttributeError, TypeError, KeyError):
                        pass

                spike_skipped, spike_action = False, None
                if trainer._spike_guard is not None:
                    from mlx_grpo.training.spike_guard import SpikeShutdownError

                    try:
                        spike_action = trainer._spike_guard.check(loss, kl, gt.iteration, {})
                        spike_skipped = spike_action in ("skip", "revert")
                    except SpikeShutdownError:
                        pass

                updated: dict[str, Any] = {
                    n: np.array(p) for n, p in _tf(trainer.model.trainable_parameters())
                }
                updated["__reward_baseline__"] = np.array([trainer._reward_baseline])

                # Update reward baseline EMA so synced workers stay calibrated
                mean_r = sum(gt.rewards) / len(gt.rewards) if gt.rewards else 0.0
                trainer._reward_baseline = 0.99 * trainer._reward_baseline + 0.01 * mean_r
                # Autonomous checkpointing (main skips on_checkpoint in process mode)
                _m = StepMetrics(
                    iteration=gt.iteration, mean_reward=mean_r, loss=loss, kl_divergence=kl
                )
                if cfg.save_every and (gt.iteration + 1) % cfg.save_every == 0:
                    trainer._fire("on_checkpoint", _m)

                # Report Metal memory from this process (main process has no model)
                _metal_mb = 0.0
                try:
                    _metal_mb = mx.metal.get_peak_memory() / (1024 * 1024)
                    mx.metal.reset_peak_memory()
                except (AttributeError, Exception):
                    pass

                result_queue.put(
                    (
                        _GRADIENT_RESULT_IDX,
                        GradientResult(
                            loss=loss,
                            kl=kl,
                            grad_norm=grad_norm,
                            learning_rate=lr,
                            updated_weights=updated,
                            spike_skipped=spike_skipped,
                            spike_action=spike_action,
                            is_weight_stats=_is_stats,
                            metal_memory_mb=_metal_mb,
                        ),
                    )
                )

        except KeyboardInterrupt:
            break
        except Exception as exc:
            _log.warning("Gradient worker error: %s", exc)
            if task.gradient_task is not None:
                result_queue.put((_GRADIENT_RESULT_IDX, GradientResult()))
            elif task.sample is not None:
                result_queue.put((task.sample_idx, None))
