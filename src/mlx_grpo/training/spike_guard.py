"""Spike guard — detect and mitigate sudden KL / loss spikes.

A spike is detected when the current step's KL or loss exceeds either:
  - An absolute threshold (kl_threshold / loss_threshold), OR
  - A delta above the recent rolling-window mean (kl_delta / loss_delta).

On detection, one of three actions fires:
  skip     — restore weights from pre-step in-memory snapshot, discard accum.
  revert   — restore weights from last checkpoint on disk, discard accum.
  shutdown — raise SpikeShutdownError to stop training cleanly.

Consecutive spike counter resets on any clean step; if it reaches
max_consecutive, shutdown fires regardless of configured action.

Skipped samples are tracked in-memory and optionally written to a JSON log.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SpikeShutdownError(RuntimeError):
    """Raised when spike guard decides to shut down training."""


@dataclass(slots=True, frozen=True)
class SpikeConfig:
    """Spike-guard configuration."""

    # ── Detection ──────────────────────────────────────────────────────
    kl_threshold: float = 0.5  # absolute KL → spike
    kl_delta: float | None = None  # KL above window-mean by this → spike
    loss_threshold: float | None = None  # absolute loss → spike
    loss_delta: float | None = None  # loss above window-mean by this → spike
    window: int = 20  # rolling window size
    min_window: int = 5  # min steps before delta checks activate

    # ── Action ─────────────────────────────────────────────────────────
    action: str = "skip"  # skip | revert | shutdown
    max_consecutive: int = 5  # consecutive spikes → forced shutdown

    # ── Logging ────────────────────────────────────────────────────────
    log_file: str | None = None  # JSON-lines path for skipped samples


@dataclass
class SkipRecord:
    """Record of a single skipped training step."""

    iteration: int
    loss: float
    kl: float
    reason: str
    consecutive: int
    sample: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "loss": self.loss,
            "kl": self.kl,
            "reason": self.reason,
            "consecutive": self.consecutive,
            "sample": self.sample,
            "timestamp": self.timestamp,
        }


def _copy_tree(tree: Any) -> Any:
    """Recursively copy an MLX parameter tree (nested dict of mx.array)."""
    if isinstance(tree, dict):
        return {k: _copy_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_copy_tree(v) for v in tree]
    try:
        import mlx.core as mx

        return mx.array(tree)
    except Exception:
        return tree


class SpikeGuard:
    """Detects KL / loss spikes and takes corrective action."""

    def __init__(self, config: SpikeConfig, adapter_path: str | None = None):
        self.config = config
        self._adapter_path = adapter_path
        self._loss_win: deque[float] = deque(maxlen=config.window)
        self._kl_win: deque[float] = deque(maxlen=config.window)
        self._snapshot: Any = None  # in-memory param snapshot
        self._consecutive: int = 0
        self._total_skipped: int = 0
        self._skips: list[SkipRecord] = []

    # ── Public API ─────────────────────────────────────────────────────

    def snapshot(self, model: Any) -> None:
        """Copy trainable parameters before the gradient step."""
        try:
            import mlx.core as mx

            self._snapshot = _copy_tree(model.trainable_parameters())
            flat = [v for v in _flatten_values(self._snapshot)]
            if flat:
                mx.eval(*flat)
        except Exception:
            self._snapshot = None

    def check(
        self,
        loss: float,
        kl: float,
        iteration: int,
        sample_info: dict[str, Any] | None = None,
    ) -> str | None:
        """Evaluate current step. Returns action string or None if clean.

        Windows are updated AFTER the spike check so the window always
        reflects the baseline *before* the current step.
        """
        reason = self._spike_reason(loss, kl)

        # Advance windows regardless
        self._loss_win.append(loss)
        self._kl_win.append(kl)

        if reason is None:
            self._consecutive = 0
            return None

        self._consecutive += 1
        self._total_skipped += 1
        rec = SkipRecord(
            iteration=iteration,
            loss=loss,
            kl=kl,
            reason=reason,
            consecutive=self._consecutive,
            sample=sample_info or {},
        )
        self._skips.append(rec)
        self._write_log(rec)

        print(
            f"  [spike-guard] {reason} | loss={loss:.4f} kl={kl:.4f}"
            f" | consecutive={self._consecutive} | action=",
            end="",
            flush=True,
        )

        if self._consecutive >= self.config.max_consecutive:
            print("SHUTDOWN (max consecutive reached)")
            raise SpikeShutdownError(
                f"Spike guard: {self._consecutive} consecutive spikes — shutting down. "
                f"Last: loss={loss:.4f} kl={kl:.4f}"
            )

        action = self.config.action
        print(action)
        return action

    def restore_snapshot(self, model: Any) -> bool:
        """Restore model to pre-step in-memory snapshot. Returns success."""
        if self._snapshot is None:
            print("  [spike-guard] Warning: no snapshot available, continuing")
            return False
        try:
            import mlx.core as mx

            model.update(self._snapshot)
            mx.eval(model.trainable_parameters())
            return True
        except Exception as e:
            print(f"  [spike-guard] Warning: snapshot restore failed: {e}")
            return False

    def restore_checkpoint(self, model: Any, optimizer: Any | None = None) -> bool:
        """Restore from last checkpoint on disk (adapter_path/adapters.safetensors)."""
        if not self._adapter_path:
            print("  [spike-guard] revert: no adapter_path configured, falling back to snapshot")
            return self.restore_snapshot(model)
        try:
            import mlx.core as mx

            weights_path = Path(self._adapter_path) / "adapters.safetensors"
            if not weights_path.exists():
                print(f"  [spike-guard] revert: {weights_path} not found, falling back to snapshot")
                return self.restore_snapshot(model)
            weights = mx.load(str(weights_path))
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())
            print(f"  [spike-guard] Reverted weights from {weights_path}")
            if optimizer is not None:
                opt_path = Path(self._adapter_path) / "optimizer_state.safetensors"
                if opt_path.exists():
                    opt_state = dict(mx.load(str(opt_path)))
                    optimizer.state.update(opt_state)
                    mx.eval(optimizer.state)
                    print("  [spike-guard] Restored optimizer state")
            return True
        except Exception as e:
            print(f"  [spike-guard] revert failed ({e}), falling back to snapshot")
            return self.restore_snapshot(model)

    def reset_accum(self, trainer: Any) -> None:
        """Clear gradient accumulation state on the trainer."""
        trainer._accum_grads = None
        trainer._accum_count = 0

    @property
    def total_skipped(self) -> int:
        return self._total_skipped

    @property
    def skip_records(self) -> list[SkipRecord]:
        return list(self._skips)

    # ── Internal ───────────────────────────────────────────────────────

    def _spike_reason(self, loss: float, kl: float) -> str | None:
        cfg = self.config
        if kl >= cfg.kl_threshold:
            return f"kl={kl:.4f} >= threshold={cfg.kl_threshold}"
        if cfg.loss_threshold is not None and loss >= cfg.loss_threshold:
            return f"loss={loss:.4f} >= threshold={cfg.loss_threshold}"
        if len(self._kl_win) >= cfg.min_window:
            mean_kl = sum(self._kl_win) / len(self._kl_win)
            if cfg.kl_delta is not None and kl > mean_kl + cfg.kl_delta:
                return f"kl={kl:.4f} > mean({mean_kl:.4f}) + delta({cfg.kl_delta})"
            if cfg.loss_delta is not None and len(self._loss_win) >= cfg.min_window:
                mean_loss = sum(self._loss_win) / len(self._loss_win)
                if loss > mean_loss + cfg.loss_delta:
                    return f"loss={loss:.4f} > mean({mean_loss:.4f}) + delta({cfg.loss_delta})"
        return None

    def _write_log(self, rec: SkipRecord) -> None:
        if not self.config.log_file:
            return
        try:
            p = Path(self.config.log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(json.dumps(rec.to_dict()) + "\n")
        except Exception:
            pass


def _flatten_values(tree: Any):
    """Yield all leaf values from a nested dict/list tree."""
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _flatten_values(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from _flatten_values(v)
    else:
        yield tree
