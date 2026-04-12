"""Tests for Metal-safe parallel rollout dispatch."""

from __future__ import annotations

import threading

import pytest

from mlx_grpo.training.parallel import (
    _ROLLOUT_LOCK,
    _serial,
    dispatch_parallel,
    is_metal_error,
)

# ── is_metal_error ────────────────────────────────────────────────────────────


def test_is_metal_error_metal_keyword():
    assert is_metal_error(RuntimeError("Metal command buffer failed"))


def test_is_metal_error_command_buffer():
    assert is_metal_error(RuntimeError("command buffer execution error"))


def test_is_metal_error_oom():
    assert is_metal_error(RuntimeError("out of memory on device"))


def test_is_metal_error_failed_to_execute():
    assert is_metal_error(RuntimeError("failed to execute kernel"))


def test_is_metal_error_false_for_normal():
    assert not is_metal_error(ValueError("bad value"))
    assert not is_metal_error(RuntimeError("division by zero"))


# ── _serial ────────────────────────────────────────────────────────────────────


def test_serial_returns_results():
    results = _serial(lambda x: x * 2, [1, 2, 3], None)
    assert results == [2, 4, 6]


def test_serial_empty_list():
    results = _serial(lambda x: x, [], None)
    assert results == []


def test_serial_metal_error_returns_none():
    def fn(x):
        if x == 2:
            raise RuntimeError("Metal GPU crash")
        return x

    errors = []
    results = _serial(fn, [1, 2, 3], lambda i, e: errors.append((i, e)))
    assert results == [1, None, 3]
    assert len(errors) == 1
    assert errors[0][0] == 1  # index 1 = item 2


def test_serial_non_metal_error_reraises():
    def fn(x):
        raise ValueError("non-metal error")

    with pytest.raises(ValueError, match="non-metal error"):
        _serial(fn, [1], None)


# ── dispatch_parallel ──────────────────────────────────────────────────────────


def test_dispatch_parallel_serial_fallback_max_workers_1():
    """max_workers=1 should use serial path."""
    results = dispatch_parallel(lambda x: x + 10, [1, 2, 3], max_workers=1)
    assert results == [11, 12, 13]


def test_dispatch_parallel_serial_fallback_single_item():
    """Single item always uses serial path."""
    results = dispatch_parallel(lambda x: x * 3, [7], max_workers=4)
    assert results == [21]


def test_dispatch_parallel_runs_in_parallel():
    results = dispatch_parallel(lambda x: x**2, [1, 2, 3, 4], max_workers=2)
    assert results == [1, 4, 9, 16]


def test_dispatch_parallel_metal_error_returns_none():
    errors = []

    def fn(x):
        if x == 2:
            raise RuntimeError("command buffer execution failed")
        return x * 10

    results = dispatch_parallel(
        fn,
        [1, 2, 3],
        max_workers=2,
        on_error=lambda i, e: errors.append(i),
    )
    assert results[0] == 10
    assert results[1] is None
    assert results[2] == 30
    assert 1 in errors  # index 1 triggered the error


def test_dispatch_parallel_non_metal_reraises():
    def fn(x):
        if x == 2:
            raise ValueError("non-metal problem")
        return x

    with pytest.raises(ValueError, match="non-metal problem"):
        dispatch_parallel(fn, [1, 2, 3], max_workers=2)


def test_dispatch_parallel_empty_list():
    results = dispatch_parallel(lambda x: x, [], max_workers=2)
    assert results == []


# ── _ROLLOUT_LOCK ──────────────────────────────────────────────────────────────


def test_rollout_lock_is_module_singleton():
    """Import from two paths — should be the same object."""
    from mlx_grpo.training import parallel as parallel_mod

    assert _ROLLOUT_LOCK is parallel_mod._ROLLOUT_LOCK


def test_rollout_lock_is_threading_lock():
    assert isinstance(_ROLLOUT_LOCK, type(threading.Lock()) | type(threading.RLock()))


def test_rollout_lock_serializes_access():
    """Two threads cannot hold the lock simultaneously."""
    results: list[bool] = []
    lock_acquired = threading.Event()
    second_acquired = threading.Event()

    def holder():
        with _ROLLOUT_LOCK:
            lock_acquired.set()
            second_acquired.wait(timeout=0.1)
            results.append(True)

    def waiter():
        lock_acquired.wait(timeout=0.5)
        acquired = _ROLLOUT_LOCK.acquire(blocking=False)
        results.append(acquired)
        if acquired:
            _ROLLOUT_LOCK.release()
        second_acquired.set()

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    t1.join(timeout=1)
    t2.join(timeout=1)
    # holder should succeed (True), waiter should fail to acquire (False)
    assert True in results
    assert False in results


# ── Integration: Trainer with rollout_workers=2 ───────────────────────────────


def test_trainer_parallel_workers_completes(mock_backend, tmp_path):
    """Trainer with rollout_workers=2 processes a batch without crashing."""
    import json

    import mlx_grpo.types.builtins  # noqa: F401 — ensure builtins registered
    from mlx_grpo.config.loader import TrainingConfig
    from mlx_grpo.data import DatasetConfig, DatasetLoader
    from mlx_grpo.training import Trainer

    ds_file = tmp_path / "data.jsonl"
    rows = [
        {"prompt": "What is 2+2?", "answer": "4", "type": "general_qna"},
        {"prompt": "What is 3+3?", "answer": "6", "type": "general_qna"},
    ]
    ds_file.write_text("\n".join(json.dumps(r) for r in rows))

    dataset = DatasetLoader(DatasetConfig(path=str(ds_file), shuffle=False))
    dataset.load()

    config = TrainingConfig(
        iterations=1,
        batch_size=2,
        group_size=2,
        rollout_workers=2,
        rollout_cooldown=0.0,
        phase_cooldown=0.0,
    )
    trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
    metrics = trainer._train_step(iteration=0)
    assert metrics is not None
    assert len(metrics.rewards) > 0
