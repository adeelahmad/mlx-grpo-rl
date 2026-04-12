"""Per-layer gradient scaling based on angle drift measurements.

Computes multipliers for each layer (L20-L35) based on angle drift, then applies
them multiplicatively to the gradient dict to balance convergence.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DynamicGradientScaler:
    """Scales gradients per layer based on angle drift measurements.

    The scaler computes multipliers by comparing each layer's angle drift to
    a threshold. Layers with low drift get min_multiplier; high drift layers
    are scaled up to max_multiplier proportionally.

    Formula:
        if drift < threshold:
            multiplier = min_multiplier
        else:
            multiplier = min_multiplier + (drift / max_observed_drift)
                         * (max_multiplier - min_multiplier)

    The multiplier is always clamped to [min_multiplier, max_multiplier] and
    checked for NaN/inf before application.
    """

    def __init__(
        self,
        angle_drift_threshold: float = 0.005,
        max_multiplier: float = 5.0,
        min_multiplier: float = 0.1,
    ):
        """Initialize the scaler.

        Args:
            angle_drift_threshold: Threshold in degrees for "low drift" detection.
            max_multiplier: Upper bound for scaling multipliers.
            min_multiplier: Lower bound for scaling multipliers.

        Raises:
            ValueError: If bounds are invalid (min >= max, or non-positive).
        """
        if min_multiplier <= 0:
            raise ValueError(f"min_multiplier must be > 0, got {min_multiplier}")
        if max_multiplier <= 0:
            raise ValueError(f"max_multiplier must be > 0, got {max_multiplier}")
        if min_multiplier >= max_multiplier:
            raise ValueError(
                f"min_multiplier ({min_multiplier}) must be < max_multiplier ({max_multiplier})"
            )

        self.angle_drift_threshold = angle_drift_threshold
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier

    def compute_multipliers(self, angle_drifts: list[float]) -> list[float]:
        """Compute per-layer scaling multipliers from angle drifts.

        Args:
            angle_drifts: List of angle drift values in degrees for each layer
                (L20-L35, 16 values expected).

        Returns:
            List of scaling multipliers (same length as input), one per layer.
            Values are bounded in [min_multiplier, max_multiplier] and safe
            (no NaN/inf).

        Algorithm:
            1. Identify layers below threshold → min_multiplier
            2. For layers above threshold:
               - Find max observed drift (among high-drift layers)
               - Scale linearly: min + (drift / max_drift) * (max - min)
            3. Clamp all values to bounds
            4. Check for and prevent NaN/inf
        """
        if not angle_drifts:
            return []

        multipliers: list[float] = []
        drifts_array = np.array(angle_drifts, dtype=np.float64)

        # Find max drift among layers strictly above threshold
        high_drift = drifts_array[drifts_array > self.angle_drift_threshold]
        max_drift = float(np.max(high_drift)) if len(high_drift) > 0 else 0.0

        for drift in angle_drifts:
            if drift < self.angle_drift_threshold:
                # Low drift: use minimum
                mult = self.min_multiplier
            elif max_drift > 0:
                # High drift: scale proportionally
                ratio = min(drift / max_drift, 1.0)  # Clamp to [0, 1]
                mult = self.min_multiplier + ratio * (self.max_multiplier - self.min_multiplier)
            else:
                # Should not reach here (drift >= threshold but max_drift == 0)
                mult = self.min_multiplier

            # Clamp to bounds and sanitize
            mult = np.clip(mult, self.min_multiplier, self.max_multiplier)
            mult = float(mult)

            # Guard against NaN/inf
            if not (math.isfinite(mult)):
                mult = self.min_multiplier

            multipliers.append(mult)

        return multipliers

    def apply_to_gradients(
        self,
        grads: dict[str, Any],
        layer_multipliers: list[float],
        layer_names: list[str],
    ) -> dict[str, Any]:
        """Apply per-layer multipliers to gradients.

        Multiplies gradients only for layers in layer_names, skipping layers
        not in the list. Skips any multiplier that would produce NaN/inf.

        Layer name matching supports both exact names (e.g., "L20") and numeric
        extraction (e.g., "L20" matches "model.layers.20.*").

        Args:
            grads: Dictionary of gradient tensors keyed by layer name
                (e.g., {"model.layers.20.self_attn.q_proj": array(...)}).
            layer_multipliers: List of scaling multipliers, one per layer
                (16 values for L20-L35).
            layer_names: List of layer names (e.g., ["L20", "L21", ...]).
                Must be same length as layer_multipliers.

        Returns:
            Dictionary with same structure as grads, but with multiplied
            gradients for matched layers. Non-matched layers are unchanged.

        Raises:
            ValueError: If layer_multipliers and layer_names have different
                lengths.
        """
        if len(layer_multipliers) != len(layer_names):
            raise ValueError(
                f"layer_multipliers ({len(layer_multipliers)}) must match"
                f" layer_names ({len(layer_names)})"
            )

        result = dict(grads)

        # Build a map of layer number → multiplier for quick lookup
        # Extract numeric layer IDs from layer names (e.g., "L20" → 20)
        layer_num_to_mult: dict[int, float] = {}
        for name, mult in zip(layer_names, layer_multipliers, strict=True):
            if math.isfinite(mult) and mult > 0:
                # Extract numeric part from names like "L20" → 20
                try:
                    layer_num = int(name.lstrip("L"))
                    layer_num_to_mult[layer_num] = mult
                except ValueError:
                    logger.warning(f"Could not extract layer number from {name}")

        # Apply multipliers to matching layers
        # Use regex to extract exact layer number from grad keys like "model.layers.20.*"
        layer_pattern = re.compile(r"\.(\d+)\.")
        for grad_key, grad_val in result.items():
            # Extract all layer numbers from grad_key
            matches = layer_pattern.findall(grad_key)
            if matches:
                # Use the first (usually only) match
                grad_layer_num = int(matches[0])
                if grad_layer_num in layer_num_to_mult:
                    multiplier = layer_num_to_mult[grad_layer_num]
                    try:
                        result[grad_key] = grad_val * multiplier
                    except (TypeError, ValueError) as e:
                        logger.warning(
                            f"Could not apply multiplier {multiplier} to gradient {grad_key}: {e}"
                        )

        return result
