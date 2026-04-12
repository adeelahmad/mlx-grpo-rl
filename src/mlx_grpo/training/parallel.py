"""Metal-safe parallel rollout dispatch.

_ROLLOUT_LOCK serializes all executor.execute() calls (one Metal submit at a time).
Cooldown sleeps run OUTSIDE the lock → hidden by other threads' GPU work.

Speedup = (n_rollouts - 1) × T_cooldown  +  (n_threads - 1) × T_reward_cpu
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

_log = logging.getLogger(__name__)


class _NoOpLock:
    """No-op lock for CUDA/ROCm where Metal command-buffer serialization is unnecessary."""

    def __enter__(self) -> _NoOpLock:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


_ROLLOUT_LOCK: threading.RLock | _NoOpLock = threading.RLock()

# Lock for boundary optimizer to prevent race conditions when multiple threads
# call observe_reward() concurrently. Used in training/__init__.py _run_sample_group.
_BOUNDARY_OPTIMIZER_LOCK: threading.Lock = threading.Lock()

# Lock for _prompt_masked_this_sample to prevent race when multiple _run_sample_group
# threads write to this attribute concurrently (via dispatch_parallel). Low severity
# but prevents stale reads in event firing code.
_PROMPT_MASK_STATE_LOCK: threading.Lock = threading.Lock()


def set_rollout_lock_enabled(enabled: bool) -> None:
    """Disable the rollout lock for non-Metal backends (CUDA, CPU).

    Metal requires serialized command buffer submission; CUDA does not.
    Call with enabled=False before training to allow parallel generation.
    """
    global _ROLLOUT_LOCK
    if enabled:
        _ROLLOUT_LOCK = threading.RLock()
    else:
        _ROLLOUT_LOCK = _NoOpLock()
        _log.info("Rollout lock disabled (CUDA/CPU — no Metal serialization needed)")


T = TypeVar("T")


def is_metal_error(e: Exception) -> bool:
    """Return True if this exception looks like a Metal GPU error.

    Uses word-boundary-aware checks so "non-metal" does not match.
    """
    msg = str(e).lower()
    # "metal" must appear at start or after a space/punctuation (not in "non-metal")
    metal_word = msg.startswith("metal") or " metal" in msg or "\nmetal" in msg
    return metal_word or any(
        k in msg for k in ("command buffer", "out of memory", "failed to execute")
    )


def dispatch_parallel(
    fn: Callable[..., T],
    items: list,
    max_workers: int,
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[T | None]:
    """Call fn(item) for each item using up to max_workers threads in parallel.

    Returns a list aligned with items. Metal errors → None (with on_error callback).
    Non-Metal errors cancel pending futures and re-raise.
    """
    if max_workers <= 1 or len(items) <= 1:
        return _serial(fn, items, on_error)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: list[Future[T]] = [pool.submit(fn, item) for item in items]
        results: list[T | None] = []
        for i, fut in enumerate(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                if is_metal_error(e):
                    _log.warning("Metal error in rollout thread %d, skipping: %s", i, e)
                    if on_error:
                        on_error(i, e)
                    results.append(None)
                else:
                    for f in futures[i + 1 :]:
                        f.cancel()
                    raise
    return results


def _serial(
    fn: Callable[..., T],
    items: list,
    on_error: Callable[[int, Exception], None] | None,
) -> list[T | None]:
    """Serial fallback — same contract as dispatch_parallel."""
    results: list[T | None] = []
    for i, item in enumerate(items):
        try:
            results.append(fn(item))
        except Exception as e:
            if is_metal_error(e):
                _log.warning("Metal error on item %d, skipping: %s", i, e)
                if on_error:
                    on_error(i, e)
                results.append(None)
            else:
                raise
    return results


class ProcessWorkerPool:  # pragma: no cover
    """N worker processes each with their own model copy — N-copy design.

    Worker 0 = gradient worker (owns policy model + optimizer + checkpointing).
    Workers 1..N-1 = rollout-only (no optimizer, weight-synced from worker 0).
    Main process = pure coordinator (no model, no GPU ops).

    Uses 'spawn' context — Metal + fork is unsafe on macOS.
    Worker 0 auto-respawns after Metal crashes; weights reload from last checkpoint.
    """

    def __init__(
        self,
        n_workers: int,
        model_path: str,
        full_config: Any,
        yaml_path: str | None,
        trainer_kwargs: dict[str, Any],
        ckpt_dir: str,
    ) -> None:
        from .gradient_worker import gradient_worker_main
        from .process_worker import WorkerTask, worker_main

        ctx = multiprocessing.get_context("spawn")
        self._ctx = ctx
        self._n = n_workers
        self._result_q: Any = ctx.Queue()
        self._task_qs: list[Any] = [ctx.Queue() for _ in range(n_workers)]
        self._procs: list[Any] = []
        self._last_weights: dict[str, Any] = {}
        self._last_metal_mb: float = 0.0  # last reported Metal memory from gradient worker
        self._gradient_worker_args = (model_path, full_config, yaml_path, trainer_kwargs, ckpt_dir)

        p0 = ctx.Process(
            target=gradient_worker_main,
            args=(
                model_path,
                full_config,
                yaml_path,
                trainer_kwargs,
                self._task_qs[0],
                self._result_q,
                ckpt_dir,
            ),
        )
        p0.start()
        self._procs.append(p0)

        for i in range(1, n_workers):
            p = ctx.Process(
                target=worker_main,
                args=(
                    model_path,
                    full_config.training,
                    yaml_path,
                    trainer_kwargs,
                    self._task_qs[i],
                    self._result_q,
                ),
            )
            p.start()
            self._procs.append(p)

        ready = 0
        while ready < n_workers:
            msg, payload = self._result_q.get(timeout=180)
            if msg == "startup_error":
                self.shutdown()
                raise RuntimeError(f"Worker startup failed: {payload}")
            if msg == "ready":
                if payload is not None:
                    self._last_weights = payload
                ready += 1
                _log.info("Worker %d/%d ready", ready, n_workers)

        if self._last_weights and n_workers > 1:
            sync_task = WorkerTask(
                sample=None, iteration=0, sample_idx=-1, weight_update=self._last_weights
            )
            for q in self._task_qs[1:]:
                q.put(sync_task)

    def _respawn_gradient_worker(self) -> bool:
        """Restart worker 0 after Metal crash. Reloads from latest checkpoint on disk."""
        from .gradient_worker import gradient_worker_main

        model_path, full_config, yaml_path, trainer_kwargs, ckpt_dir = self._gradient_worker_args
        _log.warning("Metal crash detected — respawning gradient worker (worker 0)...")
        old = self._procs[0]
        old.join(timeout=5)
        if old.is_alive():
            old.terminate()
        # Drain stale items so result_q is clean for the "ready" handshake
        with contextlib.suppress(Exception):
            while True:
                self._task_qs[0].get_nowait()
        with contextlib.suppress(Exception):
            while True:
                self._result_q.get_nowait()
        p0 = self._ctx.Process(
            target=gradient_worker_main,
            args=(
                model_path,
                full_config,
                yaml_path,
                trainer_kwargs,
                self._task_qs[0],
                self._result_q,
                ckpt_dir,
            ),
        )
        p0.start()
        self._procs[0] = p0
        try:
            msg, payload = self._result_q.get(timeout=120)
            if msg == "ready":
                if payload is not None:
                    self._last_weights = payload
                _log.info("Gradient worker respawned successfully")
                return True
            _log.error("Unexpected message during respawn: %s", msg)
        except Exception:
            _log.error("Gradient worker respawn timed out")
        return False

    def dispatch(
        self,
        batch: list[Any],
        iteration: int,
        reward_baseline: float,
        reset_optimizer: bool = False,
    ) -> tuple[list[Any], Any]:
        """Sync weights, distribute rollouts, run gradient step, return results."""
        from .process_worker import _GRADIENT_RESULT_IDX, GradientResult, GradientTask, WorkerTask

        # 1. Sync weights to rollout workers (1..N-1)
        if self._last_weights and self._n > 1:
            sync_task = WorkerTask(
                sample=None,
                iteration=iteration,
                sample_idx=-1,
                weight_update=self._last_weights,
            )
            for q in self._task_qs[1:]:
                q.put(sync_task)

        # 2. Round-robin rollout tasks; track which worker owns each sample_idx
        task_to_worker: dict[int, int] = {}
        for i, sample in enumerate(batch):
            wi = i % self._n
            task_to_worker[i] = wi
            self._task_qs[wi].put(
                WorkerTask(sample=sample, iteration=iteration, sample_idx=i, weight_update=None)
            )

        # 3. Collect rollout results; when a worker dies, fill its pending slots with None
        results: list[Any] = [None] * len(batch)
        n_received = 0
        dead: set[int] = set()
        while n_received < len(batch):
            for wi, p in enumerate(self._procs):
                if wi not in dead and not p.is_alive():
                    dead.add(wi)
                    _log.warning("Worker %d died during rollouts; skipping its samples", wi)
                    for si, owner in task_to_worker.items():
                        if owner == wi and results[si] is None:
                            n_received += 1
            if n_received >= len(batch):
                break
            try:
                sample_idx, wr = self._result_q.get(timeout=1.0)
                if 0 <= sample_idx < len(results) and results[sample_idx] is None:
                    results[sample_idx] = wr
                    n_received += 1
            except Exception:
                continue

        # 4. Build GradientTask; respawn worker 0 if it died during rollouts
        all_rollouts: list[Any] = []
        all_advantages: list[float] = []
        all_rewards: list[float] = []
        for wr in results:
            if wr is not None:
                all_rollouts.extend(wr.group_rollouts)
                all_advantages.extend(wr.advantages)
                all_rewards.extend(wr.rewards)

        if not self._procs[0].is_alive() and not self._respawn_gradient_worker():
            return results, GradientResult()

        gt = GradientTask(
            rollouts=all_rollouts,
            advantages=all_advantages,
            rewards=all_rewards,
            iteration=iteration,
            reset_optimizer=reset_optimizer,
        )
        self._task_qs[0].put(WorkerTask(gradient_task=gt))

        # 5. Collect GradientResult from worker 0; respawn on crash during gradient step
        grad_result: Any = GradientResult()
        while True:
            if not self._procs[0].is_alive():
                _log.warning("Worker 0 died during gradient step; respawning for next iteration")
                self._respawn_gradient_worker()
                break
            try:
                sample_idx, payload = self._result_q.get(timeout=5.0)
                if sample_idx == _GRADIENT_RESULT_IDX:
                    grad_result = payload
                    break
            except Exception:
                continue

        if grad_result is not None and grad_result.updated_weights:
            self._last_weights = dict(grad_result.updated_weights)
        if grad_result is not None and grad_result.metal_memory_mb > 0:
            self._last_metal_mb = grad_result.metal_memory_mb
        return results, grad_result

    def shutdown(self) -> None:
        """Send shutdown sentinels and join worker processes.

        Sends a save_checkpoint request to the gradient worker (worker 0) before
        sending the None sentinel, so Ctrl+C saves the last checkpoint.
        """
        from .process_worker import WorkerTask

        # Ask gradient worker to save before exiting
        with contextlib.suppress(Exception):
            self._task_qs[0].put(WorkerTask(save_checkpoint=True))
        # Shutdown sentinels to all workers
        for q in self._task_qs:
            with contextlib.suppress(Exception):
                q.put(None)
        for p in self._procs:
            p.join(timeout=15)  # extra time for gradient worker to save
            if p.is_alive():
                p.terminate()
