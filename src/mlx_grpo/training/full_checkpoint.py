"""Full fine-tuning checkpoint manager — directory-based with latest symlink.

Directory layout:
  checkpoints/
  ├── 0000100/
  │   ├── model.safetensors          # Complete model weights
  │   ├── config.json                # Copied from source (inference)
  │   ├── tokenizer.json             # Copied from source (inference)
  │   ├── tokenizer_config.json      # Copied from source (inference)
  │   └── optimizer_state.safetensors  # Optional (save_optimizer=True)
  ├── 0000200/
  │   └── ...
  ├── latest -> 0000200/             # Relative symlink to newest checkpoint
  ├── adapter_config.json            # Training metadata
  └── training_state.json            # Resume state
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointManager

_log = logging.getLogger(__name__)

_INFERENCE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
)

_DIR_RE = re.compile(r"^\d{7}$")


class FullFTCheckpointManager(CheckpointManager):
    """Checkpoint manager for full fine-tuning.

    Saves complete model weights in per-iteration subdirectories (not adapter
    deltas). Maintains a ``latest`` relative symlink so that resume and
    inference tools can always point at a stable path.
    """

    def __init__(
        self,
        adapter_dir: str | Path,
        keep_recent: int = 5,
        keep_best: int = 3,
        save_optimizer: bool = True,
        model_source_path: str | None = None,
    ) -> None:
        super().__init__(adapter_dir, keep_recent, keep_best, save_optimizer)
        self.model_source_path = model_source_path

    # ── Save ─────────────────────────────────────────────────────────

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
        """Save full model checkpoint in its own numbered directory."""
        metrics = metrics or {}
        _iter_int = int(iteration)
        weights = self._get_all_weights(model)
        opt_to_eval = optimizer if self.save_optimizer else None
        self._eval_before_save(weights, opt_to_eval)

        ckpt_dir = self.adapter_dir / f"{_iter_int:07d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if weights:
            self._save_safetensors(weights, ckpt_dir / "adapters.safetensors")

        if self.save_optimizer and optimizer is not None:
            self._save_optimizer_to(optimizer, ckpt_dir / "optimizer_state.safetensors")

        if self.model_source_path:
            self._copy_inference_files(ckpt_dir)

        # Copy adapter_config.json into the checkpoint dir so the directory
        # is fully self-contained for inference (mlx-lm needs both files).
        self._copy_adapter_config(ckpt_dir)

        self._update_latest_symlink(ckpt_dir)

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

        reward = metrics.get("mean_reward", 0.0)
        self._best.append((_iter_int, reward))
        self._rotate()

        saved = ckpt_dir / "adapters.safetensors"
        _log.info("Full FT checkpoint: iter %d -> %s", _iter_int, ckpt_dir.name)
        print(f"  Saved checkpoint: {ckpt_dir.name}/adapters.safetensors")
        return saved

    # ── Load ─────────────────────────────────────────────────────────

    def load_weights(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load from latest/adapters.safetensors or an explicit file/dir path.

        Search order for a directory path:
          1. adapters.safetensors   (current naming)
          2. model.safetensors      (previous naming — migration compat)

        If no directory checkpoint exists, falls back to the old flat
        ``adapters.safetensors`` at the root (pre-refactor runs).
        """
        if path is None:
            resolved = self._latest_model_path()
            if resolved is None:
                # Migration: old flat format saved at root by CheckpointManager
                flat = self.adapter_dir / "adapters.safetensors"
                resolved = flat if flat.exists() else None
        else:
            p = Path(path)
            if p.is_dir():
                resolved = (
                    (p / "adapters.safetensors")
                    if (p / "adapters.safetensors").exists()
                    else (p / "model.safetensors")
                )
            else:
                resolved = p
        if resolved is None or not resolved.exists():
            return {}
        try:
            import mlx.core as mx

            return dict(mx.load(str(resolved)))
        except ImportError:
            return {}

    def load_optimizer_state(self, optimizer: Any) -> bool:
        """Load optimizer from latest/optimizer_state.safetensors."""
        latest_dir = self._latest_dir()
        if latest_dir is None:
            _log.info("No checkpoint directory found — starting fresh optimizer")
            return False
        path = latest_dir / "optimizer_state.safetensors"
        if not path.exists():
            _log.info("No optimizer state at %s — starting fresh", path)
            return False
        if path.stat().st_size < 1024:
            _log.warning("Optimizer state too small (%d bytes), skipping", path.stat().st_size)
            return False
        try:
            import mlx.core as mx
            from mlx.utils import tree_unflatten

            opt_state = mx.load(str(path))
            if not opt_state:
                return False
            optimizer.state = tree_unflatten(list(opt_state.items()))
            _log.info("Loaded optimizer state from %s", path)
            return True
        except Exception as e:
            _log.warning("Failed to load optimizer state: %s", e)
            return False

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """List all numbered checkpoint directories, sorted by iteration."""
        results = []
        for d in self.adapter_dir.iterdir():
            if d.is_dir() and not d.is_symlink() and _DIR_RE.match(d.name):
                results.append((int(d.name), d))
        return sorted(results)

    # ── Internal ─────────────────────────────────────────────────────

    def _get_all_weights(self, model: Any) -> dict[str, Any]:
        if model is None:
            return {}
        try:
            from mlx.utils import tree_flatten

            return dict(tree_flatten(model.parameters()))
        except ImportError:
            return {}

    def _update_latest_symlink(self, ckpt_dir: Path) -> None:
        """Point latest → ckpt_dir.name (relative, so path-portable)."""
        latest = self.adapter_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_dir.name)

    def _copy_inference_files(self, ckpt_dir: Path) -> None:
        """Copy tokenizer + config files from the source model directory."""
        src = Path(self.model_source_path)  # type: ignore[arg-type]
        if not src.is_dir():
            return
        for fname in _INFERENCE_FILES:
            src_file = src / fname
            if src_file.exists() and not (ckpt_dir / fname).exists():
                shutil.copy2(str(src_file), str(ckpt_dir / fname))

    def _copy_adapter_config(self, ckpt_dir: Path) -> None:
        """Copy adapter_config.json from checkpoint root into the iteration dir."""
        src = self.adapter_dir / "adapter_config.json"
        dst = ckpt_dir / "adapter_config.json"
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))

    def _save_optimizer_to(self, optimizer: Any, path: Path) -> None:
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            saveable = {k: v for k, v in tree_flatten(optimizer.state) if isinstance(v, mx.array)}
            if saveable:
                mx.save_safetensors(str(path), dict(saveable))
        except Exception as e:
            _log.warning("Failed to save optimizer state: %s", e)

    def _latest_dir(self) -> Path | None:
        latest = self.adapter_dir / "latest"
        if latest.exists():  # follows symlink
            return latest.resolve()
        ckpts = self.list_checkpoints()
        return ckpts[-1][1] if ckpts else None

    def _latest_model_path(self) -> Path | None:
        d = self._latest_dir()
        if d is None:
            return None
        # Prefer current naming; fall back to previous naming for migration.
        for name in ("adapters.safetensors", "model.safetensors"):
            p = d / name
            if p.exists():
                return p
        return d / "adapters.safetensors"  # will not exist; caller handles

    def _rotate(self) -> None:
        """Keep only recent + best checkpoint directories."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.keep_recent + self.keep_best:
            return
        recent_iters = {it for it, _ in checkpoints[-self.keep_recent :]}
        by_reward = sorted(self._best, key=lambda x: x[1], reverse=True)
        best_iters = {it for it, _ in by_reward[: self.keep_best]}
        keep = recent_iters | best_iters
        for it, path in checkpoints:
            if it not in keep:
                shutil.rmtree(str(path), ignore_errors=True)
                _log.info("Rotated checkpoint dir: %s", path.name)


# ── End-of-File ──────────────────────────────────────────────────────
# Usage: mgr = FullFTCheckpointManager(Path("checkpoints"), model_source_path="Qwen/...")
#        mgr.save_config({"fine_tune_type": "full", "model": "..."})
#        mgr.save(iteration, model, optimizer=opt)
# ─────────────────────────────────────────────────────────────────────
