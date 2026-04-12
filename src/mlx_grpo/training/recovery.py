"""Training recovery — crash recovery and auto-resume.

Provides automatic recovery from crashes, OOM, and interrupts.
Uses checkpoint + state files to resume training seamlessly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXIT_CODE_CORRUPTION = 42  # Special exit code for model corruption


@dataclass(slots=True)
class RecoveryState:
    """Persistent training state for recovery."""

    iteration: int = 0
    epoch: int = 0
    best_reward: float = float("-inf")
    total_tokens: int = 0
    last_checkpoint: str = ""
    crash_count: int = 0
    last_crash_time: float = 0.0
    config_hash: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """Manages training state persistence and crash recovery.

    On startup, checks for existing state files and determines
    whether to resume or start fresh.

    Usage:
        recovery = RecoveryManager(checkpoint_dir="./checkpoints")
        state = recovery.load_or_create()

        # During training
        recovery.save_state(state)

        # On crash
        recovery.record_crash(state, error)
    """

    STATE_FILE = "recovery_state.json"
    MAX_CRASHES = 5
    CRASH_COOLDOWN = 60.0  # seconds

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_path = self.checkpoint_dir / self.STATE_FILE

    def load_or_create(self, config_hash: str = "") -> RecoveryState:
        """Load existing state or create new one.

        If config hash doesn't match, starts fresh (config changed).
        """
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                state = RecoveryState(
                    **{k: v for k, v in data.items() if k in RecoveryState.__dataclass_fields__}
                )
                # Check config compatibility
                if config_hash and state.config_hash and state.config_hash != config_hash:
                    return RecoveryState(config_hash=config_hash)
                return state
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Corrupted recovery state at %s, starting fresh: %s",
                    self.state_path,
                    e,
                )

        return RecoveryState(config_hash=config_hash)

    def save_state(self, state: RecoveryState) -> None:
        """Persist training state to disk."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration": state.iteration,
            "epoch": state.epoch,
            "best_reward": state.best_reward,
            "total_tokens": state.total_tokens,
            "last_checkpoint": state.last_checkpoint,
            "crash_count": state.crash_count,
            "last_crash_time": state.last_crash_time,
            "config_hash": state.config_hash,
            "extras": state.extras,
        }
        # Atomic write
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(self.state_path)

    def record_crash(self, state: RecoveryState, error: str = "") -> bool:
        """Record a crash. Returns False if max crashes exceeded.

        When max crashes is exceeded, the caller should exit with
        EXIT_CODE_CORRUPTION to signal unrecoverable failure.
        """
        state.crash_count += 1
        state.last_crash_time = time.time()
        state.extras["last_error"] = error[:500]
        self.save_state(state)

        return state.crash_count < self.MAX_CRASHES

    def should_resume(self, state: RecoveryState) -> bool:
        """Check if we should resume from checkpoint."""
        return (
            state.iteration > 0
            and state.last_checkpoint != ""
            and state.crash_count < self.MAX_CRASHES
        )

    def find_latest_checkpoint(self) -> str | None:
        """Find the most recent checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = sorted(
            [
                d
                for d in self.checkpoint_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        return str(checkpoints[0]) if checkpoints else None

    def cleanup_old_crashes(self, state: RecoveryState, cooldown: float | None = None) -> None:
        """Reset crash counter if enough time has passed."""
        cooldown = cooldown or self.CRASH_COOLDOWN
        if state.last_crash_time and time.time() - state.last_crash_time > cooldown:
            state.crash_count = 0


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None (uses json stdlib)
# Usage: mgr = RecoveryManager(checkpoint_dir); state = mgr.load_or_create()
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
