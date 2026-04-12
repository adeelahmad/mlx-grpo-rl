"""EMA state management and checkpoint persistence for dynamic gradient scaling.

This module provides exponential moving average (EMA) smoothing of per-layer gradient
multipliers, with support for checkpoint save/restore.

Integration into Trainer._train_step:
  1. After computing raw angle drifts [L20, ..., L35]
  2. Call DynamicGradientScaler.compute_multipliers(angle_drift) → raw_mults
  3. Call ema_buffer.update(raw_mults) → smoothed_mults (EMA applied)
  4. Pass smoothed_mults to DynamicGradientScaler.apply_to_gradients()
  5. On checkpoint save: ckpt["ema_state"] = ema_buffer.to_dict()
  6. On resume: ema_buffer = EMABuffer.from_dict(ckpt["ema_state"])

Example:
  >>> buffer = EMABuffer(ema_alpha=0.25)
  >>> raw = [2.0, 0.5, 1.5, 1.0] * 4  # 16 layers
  >>> smoothed = buffer.update(raw)
  >>> # First call: smoothed == raw (no history)
  >>> # Second call: smoothed[i] = 0.25 * raw[i] + 0.75 * smoothed_prev[i]
  >>> state_dict = buffer.to_dict()
  >>> buffer2 = EMABuffer.from_dict(state_dict)
  >>> assert buffer2.update(raw) == smoothed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class EMAState:
    """Snapshot of EMA state for checkpointing.

    Attributes:
        ema_multipliers: Current EMA smoothed multipliers (16 values for L20-L35)
        ema_alpha: Smoothing factor (0 < alpha ≤ 1). Higher = more weight to recent
        step_count: Number of update() calls since initialization (for diagnostics)
    """

    ema_multipliers: list[float] = field(default_factory=lambda: [1.0] * 16)
    ema_alpha: float = 0.25
    step_count: int = 0


class EMABuffer:
    """Exponential moving average buffer for per-layer gradient multipliers.

    Applies EMA smoothing to raw multipliers. On first call, uses raw values
    (no history). On subsequent calls, applies the EMA formula:
      ema[i] = alpha * raw[i] + (1 - alpha) * ema[i]

    Supports checkpoint serialization via to_dict()/from_dict().
    """

    def __init__(self, ema_alpha: float = 0.25) -> None:
        """Initialize EMA buffer.

        Args:
            ema_alpha: Smoothing factor. Default 0.25 (75% weight on history).
                Must be in (0, 1].
        """
        if not (0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self._ema_multipliers: list[float] = [1.0] * 16
        self._ema_alpha = ema_alpha
        self._step_count: int = 0

    def update(self, raw_multipliers: list[float]) -> list[float]:
        """Apply EMA smoothing to raw multipliers.

        On first call: returns raw multipliers unchanged (initializes history).
        On subsequent calls: applies EMA formula.

        Args:
            raw_multipliers: Raw multipliers from DynamicGradientScaler
              (16 values for layers L20-L35)

        Returns:
            Smoothed multipliers after EMA filtering

        Raises:
            ValueError: If raw_multipliers length != 16

        Example:
            >>> buffer = EMABuffer(ema_alpha=0.25)
            >>> # First step: initialize with raw values
            >>> raw1 = [2.0, 0.5, 1.5, 1.0] * 4
            >>> smoothed1 = buffer.update(raw1)
            >>> assert smoothed1 == raw1  # No history yet
            >>>
            >>> # Second step: apply EMA formula
            >>> raw2 = [3.0, 0.3, 2.0, 1.2] * 4
            >>> smoothed2 = buffer.update(raw2)
            >>> # smoothed2[0] = 0.25 * 3.0 + 0.75 * 2.0 = 2.25
            >>> # smoothed2[1] = 0.25 * 0.3 + 0.75 * 0.5 = 0.525
            >>> assert abs(smoothed2[0] - 2.25) < 1e-6
            >>> assert abs(smoothed2[1] - 0.525) < 1e-6
        """
        if len(raw_multipliers) != 16:
            raise ValueError(f"Expected 16 multipliers (L20-L35), got {len(raw_multipliers)}")

        self._step_count += 1

        # First call: initialize with raw values
        if self._step_count == 1:
            self._ema_multipliers = list(raw_multipliers)
            return list(self._ema_multipliers)

        # Apply EMA formula: ema[i] = alpha * raw[i] + (1 - alpha) * ema[i]
        alpha = self._ema_alpha
        smoothed = [
            alpha * raw + (1.0 - alpha) * ema
            for raw, ema in zip(raw_multipliers, self._ema_multipliers, strict=True)
        ]

        self._ema_multipliers = smoothed
        return list(smoothed)

    def to_dict(self) -> dict[str, Any]:
        """Serialize EMA state for checkpoint persistence.

        Returns:
            Dictionary with keys: ema_multipliers, ema_alpha, step_count
        """
        return {
            "ema_multipliers": list(self._ema_multipliers),
            "ema_alpha": self._ema_alpha,
            "step_count": self._step_count,
        }

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> EMABuffer:
        """Restore EMA buffer from checkpoint.

        Args:
            state_dict: Dictionary from to_dict() (or training_state.json)

        Returns:
            Restored EMABuffer with saved state

        Raises:
            KeyError: If required keys missing
            ValueError: If multipliers length != 16

        Example:
            >>> buffer = EMABuffer(ema_alpha=0.25)
            >>> raw = [2.0, 0.5, 1.5, 1.0] * 4
            >>> _ = buffer.update(raw)
            >>> raw2 = [3.0, 0.3, 2.0, 1.2] * 4
            >>> smoothed = buffer.update(raw2)
            >>>
            >>> # Save checkpoint
            >>> ckpt = buffer.to_dict()
            >>>
            >>> # Resume from checkpoint
            >>> buffer2 = EMABuffer.from_dict(ckpt)
            >>> assert buffer2._step_count == buffer._step_count
            >>> assert buffer2._ema_multipliers == buffer._ema_multipliers
            >>>
            >>> # Next update uses restored state
            >>> raw3 = [1.5, 0.4, 1.8, 0.9] * 4
            >>> smoothed3 = buffer2.update(raw3)
            >>> smoothed3_fresh = buffer.update(raw3)
            >>> assert smoothed3 == smoothed3_fresh  # Identical continuity
        """
        restored = cls(ema_alpha=state_dict["ema_alpha"])
        restored._ema_multipliers = list(state_dict["ema_multipliers"])
        restored._step_count = state_dict["step_count"]
        return restored

    @property
    def step_count(self) -> int:
        """Number of update() calls (for diagnostics)."""
        return self._step_count

    @property
    def ema_alpha(self) -> float:
        """Current smoothing factor."""
        return self._ema_alpha

    @property
    def current_multipliers(self) -> list[float]:
        """Read current EMA multipliers (read-only)."""
        return list(self._ema_multipliers)
