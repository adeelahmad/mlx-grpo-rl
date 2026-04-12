"""PyTorch checkpoint management — save, load, and rotate PEFT adapters.

Matches MLX CheckpointManager format: flat layout with symlinks.
Uses PEFT save_pretrained for adapter weights, torch.save for optimizer.

Directory layout:
    adapter_dir/
    +-- latest                   # Symlink → 0000100/ (latest checkpoint dir)
    +-- 0000100/                 # Numbered checkpoint directory (iter 100)
    |   +-- adapter_config.json
    |   +-- adapter_model.safetensors (or other PEFT weights)
    +-- 0000200/                 # Previous checkpoint
    +-- optimizer_state.pt       # Optimizer state_dict
    +-- training_state.json      # Iteration, tokens, lr, wandb_run_id
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class TorchCheckpointManager:
    """Manages PyTorch/PEFT adapter checkpoints."""

    def __init__(
        self,
        adapter_dir: str | Path,
        keep_recent: int = 5,
        keep_best: int = 3,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.keep_recent = keep_recent
        self.keep_best = keep_best
        self._best: list[tuple[int, float]] = []

    def save_config(self, config: dict[str, Any]) -> None:
        """Save training config once at start."""
        path = self.adapter_dir / "adapter_config.json"
        if not path.exists():
            with open(path, "w") as f:
                json.dump(config, f, indent=2, default=str)

    def save(
        self,
        iteration: int | float,
        model: Any,
        optimizer: Any = None,
        metrics: dict[str, float] | None = None,
        trained_tokens: int = 0,
        learning_rate: float = 0.0,
        wandb_run_id: str | None = None,
    ) -> Path:
        """Save checkpoint: numbered dir + latest symlink + optimizer + training state."""
        import torch

        metrics = metrics or {}
        _iter_int = int(iteration)
        numbered_dir = self.adapter_dir / f"{_iter_int:07d}"

        # 1. Save adapter weights (PEFT save_pretrained or full state_dict)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(numbered_dir))
            # Create symlink latest → numbered dir (matches MLX pattern)
            latest_link = self.adapter_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(numbered_dir.name)
        else:
            # Full model: save state_dict
            numbered_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), numbered_dir / "model.pt")
            # Create symlink latest → numbered dir (matches MLX pattern)
            latest_link = self.adapter_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(numbered_dir.name)

        # 2. Optimizer state
        if optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                self.adapter_dir / "optimizer_state.pt",
            )

        # 3. Training state
        state: dict[str, Any] = {
            "iteration": _iter_int,
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

        _log.info("Checkpoint saved: iter %d -> %s", _iter_int, numbered_dir.name)
        print(f"  Saved checkpoint: {numbered_dir.name}")
        return numbered_dir

    def load_weights(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load adapter weights from latest checkpoint. Returns empty dict if not found."""
        import torch

        if path is None:
            # Resolve latest symlink if it exists
            latest = self.adapter_dir / "latest"
            if latest.is_symlink():
                # Follow symlink to actual checkpoint directory
                path = latest.resolve()
            elif latest.exists():
                path = latest
            else:
                return {}
        path = Path(path)

        if not path.exists():
            return {}

        try:
            # Try PEFT load_pretrained format (adapter_config.json + adapter weights)
            if (path / "adapter_config.json").exists():
                # Return config and weight info for the caller to load
                return {"type": "peft", "path": str(path)}
            # Try torch.save format (model.pt)
            elif (path / "model.pt").exists():
                state = torch.load(str(path / "model.pt"), map_location="cpu", weights_only=True)
                return state if isinstance(state, dict) else {}
        except Exception as e:
            _log.warning("Failed to load weights from %s: %s", path, e)
            return {}

    def load_training_state(self) -> dict[str, Any]:
        """Load training state for resume."""
        path = self.adapter_dir / "training_state.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def load_optimizer_state(self, optimizer: Any) -> bool:
        """Load optimizer state into optimizer. Returns True if loaded."""
        import torch

        path = self.adapter_dir / "optimizer_state.pt"
        if not path.exists():
            return False
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state)
            _log.info("Loaded optimizer state from %s", path)
            return True
        except Exception as e:
            _log.warning("Failed to load optimizer state: %s", e)
            return False

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """List all numbered checkpoint dirs sorted by iteration."""
        results = []
        for d in self.adapter_dir.iterdir():
            if d.is_dir() and d.name.isdigit() and len(d.name) == 7:
                results.append((int(d.name), d))
        return sorted(results)

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
                shutil.rmtree(path, ignore_errors=True)
                _log.info("Rotated checkpoint: %s", path.name)
