"""Dynamic per-layer gradient scaling configuration.

Automatically scales gradients per layer based on angle drift to achieve
balanced convergence across all LoRA layers (L20-L35, 16 layers total).

Usage in YAML:
  dynamic_gradient_scaling:
    enabled: true
    angle_drift_threshold: 0.005      # degrees
    max_multiplier: 5.0
    min_multiplier: 0.1
    ema_alpha: 0.25
    update_interval: 1                # always 1, reserved for future
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DynamicGradientScalingConfig:
    """Configuration for per-layer gradient scaling based on angle drift.

    Attributes:
        enabled: Whether to enable dynamic gradient scaling.
        angle_drift_threshold: Angle drift threshold in degrees. Layers below
            this threshold get min_multiplier; above get scaled between min and
            max based on observed drift.
        max_multiplier: Maximum scaling multiplier (clipped at upper bound).
        min_multiplier: Minimum scaling multiplier (clipped at lower bound).
        ema_alpha: EMA smoothing factor (0.0-1.0). Higher = faster response to
            new measurements; lower = more stable, less noisy.
        update_interval: Steps between recompute. Reserved for future; currently
            always 1.
    """

    enabled: bool = False
    angle_drift_threshold: float = 0.005
    max_multiplier: float = 5.0
    min_multiplier: float = 0.1
    ema_alpha: float = 0.25
    update_interval: int = 1
