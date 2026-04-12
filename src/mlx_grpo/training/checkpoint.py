"""Checkpoint management — save, load, and rotate adapter weights.

Flat-file format matching mlx-lm conventions:
  adapter_dir/
  ├── adapters.safetensors              # Latest weights (always updated)
  ├── adapter_config.json               # Training config (saved once)
  ├── optimizer_state.safetensors       # Optimizer momentum/variance
  ├── training_state.json               # Iteration, tokens, lr, wandb_run_id
  ├── 0000100_adapters.safetensors      # Numbered checkpoint at iter 100
  └── ...
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^(\d{7})_adapters\.safetensors$")


class CheckpointManager:
    """Manages adapter checkpoints in mlx-lm-compatible flat format."""

    def __init__(
        self,
        adapter_dir: str | Path,
        keep_recent: int = 5,
        keep_best: int = 3,
        save_optimizer: bool = True,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.keep_recent = keep_recent
        self.keep_best = keep_best
        self.save_optimizer = save_optimizer
        self._best: list[tuple[int, float]] = []

    def save_config(self, config: dict[str, Any]) -> None:
        """Save adapter config (adapter_config.json). Write-once: if the file
        already exists it is left unchanged to preserve the original config."""
        path = self.adapter_dir / "adapter_config.json"
        if path.exists():
            return
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def save(
        self,
        iteration: int | float,
        model: Any,
        optimizer: Any = None,
        trained_tokens: int = 0,
        learning_rate: float = 0.0,
        wandb_run_id: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save checkpoint: latest + numbered + optimizer + training state."""
        metrics = metrics or {}
        _iter_int = int(iteration)
        weights = self._get_trainable_weights(model)

        # Force-materialize all lazy arrays BEFORE serialization.
        # Without this, optimizer momentum (m/v) and adapter weights may be
        # saved as stale/incomplete tensors, corrupting the checkpoint.
        self._eval_before_save(weights, optimizer)

        # 1. Adapter weights — save numbered, update 'latest' symlink + flat copy.
        #    Symlink keeps the most-recent checkpoint name stable; flat copy
        #    preserves mlx-lm load compatibility (adapters.safetensors).
        numbered = self.adapter_dir / f"{_iter_int:07d}_adapters.safetensors"
        if weights:
            self._save_safetensors(weights, numbered)
            shutil.copy2(str(numbered), str(self.adapter_dir / "adapters.safetensors"))
            # Relative symlink: latest → 0000100_adapters.safetensors
            latest = self.adapter_dir / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(numbered.name)

        # 2. Optimizer state
        if self.save_optimizer and optimizer is not None:
            self._save_optimizer_state(optimizer)

        # 3. Training state (used for resume — stores NEXT iteration)
        state: dict[str, Any] = {
            "iteration": _iter_int + 1,
            "trained_tokens": trained_tokens,
            "learning_rate": learning_rate,
        }
        if wandb_run_id:
            state["wandb_run_id"] = wandb_run_id
        for k, v in metrics.items():
            state[f"metric_{k}"] = v
        with open(self.adapter_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # 4. Rotation
        reward = metrics.get("mean_reward", 0.0)
        self._best.append((_iter_int, reward))
        self._rotate()

        _log.info("Checkpoint saved: iter %d -> %s", _iter_int, numbered.name)
        print(f"  Saved checkpoint: {numbered.name}")
        return numbered

    def load_training_state(self) -> dict[str, Any]:
        """Load training state for resume."""
        path = self.adapter_dir / "training_state.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def load_optimizer_state(self, optimizer: Any) -> bool:
        """Load optimizer state into optimizer. Returns True if loaded."""
        path = self.adapter_dir / "optimizer_state.safetensors"
        if not path.exists():
            return False
        # Reject obviously corrupt files (real Adam state is >> 1KB)
        if path.stat().st_size < 1024:
            _log.warning("Optimizer state too small (%d bytes), skipping", path.stat().st_size)
            return False
        try:
            import mlx.core as mx
            from mlx.utils import tree_unflatten

            opt_state = mx.load(str(path))
            if not opt_state:
                _log.warning("Optimizer state is empty, skipping")
                return False
            optimizer.state = tree_unflatten(list(opt_state.items()))
            _log.info("Loaded optimizer state from %s", path)
            return True
        except Exception as e:
            _log.warning("Failed to load optimizer state: %s", e)
            return False

    def load_weights(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load adapter weights. Defaults to latest adapters.safetensors."""
        if path is None:
            path = self.adapter_dir / "adapters.safetensors"
        path = Path(path)
        if not path.exists():
            return {}
        try:
            import mlx.core as mx

            return dict(mx.load(str(path)))
        except ImportError:
            return {}

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """List all numbered checkpoints sorted by iteration."""
        results = []
        for f in self.adapter_dir.iterdir():
            m = CHECKPOINT_PATTERN.match(f.name)
            if m:
                results.append((int(m.group(1)), f))
        return sorted(results)

    # ── Internal ──

    def _get_trainable_weights(self, model: Any) -> dict[str, Any]:
        if model is None:
            return {}
        try:
            from mlx.utils import tree_flatten

            return dict(tree_flatten(model.trainable_parameters()))
        except ImportError:
            return {}

    @staticmethod
    def _eval_before_save(weights: dict[str, Any], optimizer: Any) -> None:
        """mx.eval all lazy arrays so serialization writes materialized data."""
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            arrays: list[mx.array] = []
            arrays.extend(v for v in weights.values() if isinstance(v, mx.array))
            if optimizer is not None and hasattr(optimizer, "state"):
                for _, v in tree_flatten(optimizer.state):
                    if isinstance(v, mx.array):
                        arrays.append(v)
            if arrays:
                mx.eval(*arrays)
        except ImportError:
            pass

    def _save_safetensors(self, weights: dict[str, Any], path: Path) -> None:
        try:
            import mlx.core as mx

            mx.save_safetensors(str(path), weights)
        except ImportError:
            _log.warning("MLX not available - skipping weight save")

    def _save_optimizer_state(self, optimizer: Any) -> None:
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            opt_state = dict(tree_flatten(optimizer.state))
            saveable = {k: v for k, v in opt_state.items() if isinstance(v, mx.array)}
            if saveable:
                mx.save_safetensors(
                    str(self.adapter_dir / "optimizer_state.safetensors"),
                    saveable,
                )
        except Exception as e:
            _log.warning("Failed to save optimizer state: %s", e)

    def _rotate(self) -> None:
        """Keep only recent + best numbered checkpoints."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.keep_recent + self.keep_best:
            return

        recent_iters = {it for it, _ in checkpoints[-self.keep_recent :]}
        by_reward = sorted(self._best, key=lambda x: x[1], reverse=True)
        best_iters = {it for it, _ in by_reward[: self.keep_best]}

        keep = recent_iters | best_iters
        for it, path in checkpoints:
            if it not in keep:
                path.unlink(missing_ok=True)
                _log.info("Rotated checkpoint: %s", path.name)


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None (mlx optional for weight save/load)
# Usage: mgr = CheckpointManager(Path("adapters"))
#        mgr.save_config(config_dict)
#        mgr.save(iteration, model, optimizer=opt, metrics={"mean_reward": 0.8})
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
