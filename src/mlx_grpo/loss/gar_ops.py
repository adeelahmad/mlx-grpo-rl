"""GAR (Guided Activation Reinforcement) loss — MLX-native.

Penalizes cosine angle drift between scaffolded ref and naked policy
activations above a dead-zone threshold. Phase 1: core ops only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass(slots=True)
class GARGradStepResult:
    """Result from a GAR gradient step."""

    loss: float = 0.0
    grad_norm: float = 0.0
    effective_tokens: int = 0
    per_layer_metrics: dict[str, dict[str, float]] | None = None


def _require_mlx() -> None:
    if not HAS_MLX:
        raise NotImplementedError("GAR ops require MLX. Install with: pip install mlx")


def capture_activations_forward(
    model: Any,
    token_ids: list[int],
    lora_layers: list[int],
) -> dict[str, Any]:
    """Forward pass capturing hidden states at specified layers.

    No mx.eval calls to support usage inside nn.value_and_grad scope.
    Input is stopped with mx.stop_gradient to prevent unintended gradients.
    """
    _require_mlx()
    from ..inspect.collector import _get_base_model, _get_embed_func, _layer_forward

    base = _get_base_model(model)
    embed_fn = _get_embed_func(base)
    if embed_fn is None:
        return {}

    x = mx.stop_gradient(mx.array([token_ids]))
    h = embed_fn(x)

    result: dict[str, Any] = {}
    layer_set = set(lora_layers)
    for i, layer in enumerate(base.layers):
        h = _layer_forward(layer, h)
        if i in layer_set:
            result[str(i)] = h[0]  # squeeze batch dim

    return result


def _build_weight_vector(
    seq_len: int,
    masks: list[float] | None,
    proximity_weights: list[float] | None,
) -> Any:
    """Build per-token weight vector from masks and proximity weights."""
    if masks:
        m = mx.array(masks[:seq_len], dtype=mx.float32)
    else:
        m = mx.ones((seq_len,), dtype=mx.float32)
    if proximity_weights is not None:
        m = m * mx.array(proximity_weights[:seq_len], dtype=mx.float32)
    return m


def _layer_metrics(
    angles_deg: Any,
    active_mask: Any,
    n_act: float,
    threshold_deg: float,
) -> dict[str, float]:
    """Compute per-layer angle statistics."""
    if n_act <= 0:
        return {
            "angle_mean": 0.0,
            "angle_std": 0.0,
            "angle_max": 0.0,
            "tokens_above": 0.0,
            "tokens_below": 0.0,
        }
    masked = angles_deg * active_mask
    mean = float((mx.sum(masked) / n_act).item())
    sq_mean = float((mx.sum(masked * masked) / n_act).item())
    std = math.sqrt(max(sq_mean - mean**2, 0.0))
    above = float(mx.sum((angles_deg > threshold_deg) * active_mask).item())
    return {
        "angle_mean": mean,
        "angle_std": std,
        "angle_max": float(mx.max(angles_deg).item()),
        "tokens_above": above,
        "tokens_below": n_act - above,
    }


def compute_activation_drift_loss(
    ref_acts_dict: dict[str, Any],
    policy_acts_dict: dict[str, Any],
    threshold_deg: float = 5.0,
    normalize: bool = True,
    masks: list[float] | None = None,
    proximity_weights: list[float] | None = None,
) -> tuple[Any, dict[str, dict[str, float]]]:
    """Cosine angle drift loss with dead-zone threshold. Returns (loss, metrics).

    Loss is averaged across layers to keep scale similar to GRPO loss (~0-1 range).
    Each layer contributes: sum(excess_angles * weight) / num_active_tokens.
    Final loss: average across all captured layers.
    """
    _require_mlx()
    eps = 1e-8
    threshold_rad = mx.array(threshold_deg * math.pi / 180.0)
    sum_of_squares = mx.array(0.0)
    layer_count = 0
    metrics: dict[str, dict[str, float]] = {}

    common_keys = sorted(set(ref_acts_dict) & set(policy_acts_dict))
    if not common_keys:
        return mx.array(0.0), {}

    for key in common_keys:
        ref, pol = ref_acts_dict[key], policy_acts_dict[key]
        seq_len = min(ref.shape[0], pol.shape[0])
        if seq_len == 0:
            continue

        ref_s, pol_s = ref[:seq_len], pol[:seq_len]
        if normalize:
            ref_s = ref_s / (mx.linalg.norm(ref_s, axis=-1, keepdims=True) + eps)
            pol_s = pol_s / (mx.linalg.norm(pol_s, axis=-1, keepdims=True) + eps)

        dot = mx.clip(mx.sum(ref_s * pol_s, axis=-1), -1.0 + eps, 1.0 - eps)
        angles_rad = mx.arccos(dot)
        excess = mx.maximum(angles_rad - threshold_rad, 0.0)

        m = _build_weight_vector(seq_len, masks, proximity_weights)
        n_active = mx.sum((m > 0).astype(mx.float32))
        layer_loss = mx.sum(excess * m) / mx.maximum(n_active, mx.array(1.0))
        sum_of_squares = sum_of_squares + (layer_loss**2)
        layer_count += 1

        active_mask = (m > 0).astype(mx.float32)
        n_act_val = float(mx.sum(active_mask).item())
        angles_deg = angles_rad * (180.0 / math.pi)
        metrics[key] = _layer_metrics(angles_deg, active_mask, n_act_val, threshold_deg)

    # RMS across layers: sqrt(sum_of_squares / layer_count)
    # Preserves gradient signal while normalizing scale-invariantly
    rms_loss = mx.sqrt(sum_of_squares / float(layer_count)) if layer_count > 0 else mx.array(0.0)

    return rms_loss, metrics


def gar_gradient_step(
    model: Any,
    ref_acts: dict[str, Any],
    lora_layers: list[int],
    optimizer: Any,
    token_ids: list[int],
    threshold_deg: float = 5.0,
    normalize: bool = True,
    masks: list[float] | None = None,
    proximity_weights: list[float] | None = None,
    max_grad_norm: float = 3.0,
) -> GARGradStepResult:
    """GAR gradient step: forward -> drift loss -> backward -> update.

    Captures policy activations BEFORE gradient step to ensure metrics
    use the same activation values as the loss computation.
    """
    _require_mlx()
    from .mlx_ops import _tree_eval, _tree_norm, _tree_scale, apply_grads

    # Capture policy activations BEFORE gradient step (while model state is stable)
    policy_acts = capture_activations_forward(model, token_ids, lora_layers)

    def loss_fn(
        m: Any,
        _pol_acts: dict[str, Any] = policy_acts,
        _r: dict[str, Any] = ref_acts,
        _td: float = threshold_deg,
        _n: bool = normalize,
        _m: list[float] | None = masks,
        _p: list[float] | None = proximity_weights,
    ) -> Any:
        # Use pre-captured policy activations to avoid double forward pass
        return compute_activation_drift_loss(_r, _pol_acts, _td, _n, _m, _p)[0]

    loss_val, grads = nn.value_and_grad(model, loss_fn)(model)  # type: ignore[union-attr]
    mx.eval(loss_val)
    _tree_eval(grads)

    # Compute metrics from same activations used in loss
    _, metrics = compute_activation_drift_loss(
        ref_acts, policy_acts, threshold_deg, normalize, masks, proximity_weights
    )

    grad_norm = _tree_norm(grads)
    if max_grad_norm > 0 and grad_norm > max_grad_norm:
        grads = _tree_scale(grads, max_grad_norm / grad_norm)
        _tree_eval(grads)

    apply_grads(model, optimizer, grads)
    eff_tokens = sum(1 for w in masks if w > 0) if masks is not None else len(token_ids)
    return GARGradStepResult(
        loss=float(loss_val),
        grad_norm=grad_norm,
        effective_tokens=eff_tokens,
        per_layer_metrics=metrics,
    )
