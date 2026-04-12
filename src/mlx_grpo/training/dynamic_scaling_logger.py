"""Dynamic gradient scaling logger — per-layer multipliers to W&B, CSV, JSONL, console.

Provides standalone logging functions for gradient multipliers computed by
DynamicGradientScaler. These functions are pure (no state mutations) and handle
missing wandb gracefully.

Integration points:
  - log_multipliers_to_wandb(): call in training/__init__.py after gradient scaling
  - format_multipliers_for_console(): call in training/__init__.py, print result
  - multipliers_to_csv_row(): call in stats_logger.py step_hook()
  - multipliers_to_jsonl_dict(): call in rollout_logger.py log() method
"""

from __future__ import annotations

from typing import Any


def log_multipliers_to_wandb(
    multipliers: list[float],
    layer_names: list[str],
    step: int,
    mean_angle_drift: float | None = None,
) -> None:
    """Log per-layer gradient multipliers and aggregates to W&B.

    Logs:
    - Per-layer keys: "gradient_multipliers/L20", ..., "gradient_multipliers/L35"
    - Aggregates: "gradient_scaling/max_multiplier_used", "gradient_scaling/mean_angle_drift"

    Handles missing wandb gracefully (no-op if not available).

    Args:
        multipliers: Multiplier values [2.1, 0.5, ...] for each layer (len 16).
        layer_names: Layer names ["L20", "L21", ...] (len 16).
        step: Training iteration number.
        mean_angle_drift: Optional mean angle drift value to log.

    Example:
        >>> log_multipliers_to_wandb(
        ...     [2.1, 0.5, 3.5, 1.0, ...],
        ...     ["L20", "L21", "L22", "L23", ...],
        ...     step=100,
        ...     mean_angle_drift=0.04,
        ... )
        # W&B receives:
        # {"gradient_multipliers/L20": 2.1, ..., "gradient_scaling/max_multiplier_used": 3.5,
        #  "gradient_scaling/mean_angle_drift": 0.04}
    """
    if not multipliers or not layer_names:
        return

    # Attempt W&B import; no-op if unavailable
    try:
        import wandb
    except ImportError:
        return

    # Build per-layer dict
    per_layer_dict = {
        f"gradient_multipliers/{name}": mult
        for name, mult in zip(layer_names, multipliers, strict=False)
    }

    # Add aggregates
    max_mult = max(multipliers) if multipliers else 1.0
    per_layer_dict["gradient_scaling/max_multiplier_used"] = max_mult
    if mean_angle_drift is not None:
        per_layer_dict["gradient_scaling/mean_angle_drift"] = mean_angle_drift

    # Log to W&B
    wandb.log(per_layer_dict, step=step)


def format_multipliers_for_console(
    multipliers: list[float],
    layer_names: list[str],
) -> str:
    """Format top-3 multipliers for console output.

    Returns a string like "[DYNAMIC_SCALING] Top 3 Layers: L27:4.5 | L31:4.1 | L21:3.8"

    Returns empty string if multipliers all equal 1.0 (no scaling applied).

    Args:
        multipliers: Multiplier values [2.1, 0.5, ...] (len 16).
        layer_names: Layer names ["L20", "L21", ...] (len 16).

    Returns:
        Console-formatted string, or empty string if all multipliers == 1.0.

    Example:
        >>> format_multipliers_for_console(
        ...     [2.1, 0.5, 3.5, 1.0, 1.0, 1.0, ...],
        ...     ["L20", "L21", "L22", "L23", "L24", "L25", ...],
        ... )
        '[DYNAMIC_SCALING] Top 3 Layers: L22:3.5 | L20:2.1 | L21:0.5'
    """
    if not multipliers or not layer_names:
        return ""

    # Skip if all multipliers are 1.0 (no scaling)
    if all(abs(m - 1.0) < 1e-6 for m in multipliers):
        return ""

    # Pair up names and multipliers, sort by magnitude (descending)
    pairs = sorted(
        zip(layer_names, multipliers, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )

    # Top 3
    top_3 = pairs[:3]
    if not top_3:
        return ""

    # Format: "L27:4.5 | L31:4.1 | L21:3.8"
    parts = [f"{name}:{mult:.1f}" for name, mult in top_3]
    return f"[DYNAMIC_SCALING] Top 3 Layers: {' | '.join(parts)}"


def multipliers_to_csv_row(
    multipliers: list[float],
    layer_names: list[str],
    max_mult: float,
    mean_drift: float,
) -> dict[str, str]:
    """Convert multipliers to CSV row dict for stats.csv.

    Returns dict with keys:
    - "L20_mult", "L21_mult", ..., "L35_mult" (16 columns)
    - "max_multiplier_used" (float, max across all multipliers)
    - "mean_angle_drift" (float, average angle drift across all layers)

    Args:
        multipliers: Multiplier values [2.1, 0.5, ...] (len 16).
        layer_names: Layer names ["L20", "L21", ...] (len 16).
        max_mult: Maximum multiplier value (for logging; typically max(multipliers)).
        mean_drift: Mean angle drift across all layers (in degrees).

    Returns:
        Dict ready to unpack into csv.DictWriter.writerow({**existing, **result}).

    Example:
        >>> multipliers_to_csv_row(
        ...     [2.1, 0.5, 3.5, 1.0, ...],
        ...     ["L20", "L21", "L22", "L23", ...],
        ...     max_mult=3.5,
        ...     mean_drift=0.042,
        ... )
        {
            'L20_mult': '2.1',
            'L21_mult': '0.5',
            'L22_mult': '3.5',
            'L23_mult': '1.0',
            ...,
            'max_multiplier_used': '3.5',
            'mean_angle_drift': '0.0420',
        }
    """
    row: dict[str, str] = {}

    # Per-layer columns
    for name, mult in zip(layer_names, multipliers, strict=False):
        row[f"{name}_mult"] = f"{mult:.4f}"

    # Aggregate columns
    row["max_multiplier_used"] = f"{max_mult:.4f}"
    row["mean_angle_drift"] = f"{mean_drift:.4f}"

    return row


def multipliers_to_jsonl_dict(
    multipliers: list[float],
    layer_names: list[str],
    max_mult: float,
) -> dict[str, Any]:
    """Convert multipliers to JSONL dict for rollouts.jsonl.

    Returns dict with keys:
    - "gradient_multipliers": {"L20": 2.1, "L21": 0.5, ...}
    - "max_multiplier_used": 3.5

    Suitable for merging into rollout entry dict before JSON serialization.

    Args:
        multipliers: Multiplier values [2.1, 0.5, ...] (len 16).
        layer_names: Layer names ["L20", "L21", ...] (len 16).
        max_mult: Maximum multiplier value (for logging).

    Returns:
        Dict ready to unpack into rollout entry: entry.update(multipliers_to_jsonl_dict(...))

    Example:
        >>> multipliers_to_jsonl_dict(
        ...     [2.1, 0.5, 3.5, 1.0, ...],
        ...     ["L20", "L21", "L22", "L23", ...],
        ...     max_mult=3.5,
        ... )
        {
            'gradient_multipliers': {
                'L20': 2.1,
                'L21': 0.5,
                'L22': 3.5,
                'L23': 1.0,
                ...
            },
            'max_multiplier_used': 3.5,
        }
    """
    return {
        "gradient_multipliers": {
            name: mult for name, mult in zip(layer_names, multipliers, strict=False)
        },
        "max_multiplier_used": max_mult,
    }
