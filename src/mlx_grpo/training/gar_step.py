"""GAR gradient step — called from Trainer when mode='gar'.

Takes scaffolded + naked rollouts, captures activations from both,
and computes the activation drift loss to align naked toward scaffolded.
The gradient flows only through the naked (policy) forward pass.
"""

from __future__ import annotations

from typing import Any

from mlx_grpo.engine.result import RolloutResult


def gar_gradient_step(
    model: Any,
    optimizer: Any,
    rollout_pairs: list[tuple[RolloutResult, RolloutResult]],
    lora_layers: list[int],
    threshold_deg: float = 5.0,
    normalize: bool = True,
    max_grad_norm: float = 3.0,
    masks: list[list[float] | None] | None = None,
    proximity_weights: list[list[float] | None] | None = None,
) -> dict[str, Any]:
    """Run GAR gradient step over paired rollouts.

    Args:
        model: MLX model with LoRA adapters.
        optimizer: MLX optimizer instance.
        rollout_pairs: List of (scaffolded, naked) RolloutResult pairs.
        lora_layers: Layer indices for activation capture.
        threshold_deg: Dead zone threshold in degrees.
        normalize: L2-normalize activations before angle computation.
        max_grad_norm: Gradient clipping threshold.
        masks: Per-pair per-token loss weights (from IS/segment weighting).
        proximity_weights: Per-pair per-token importance weights.

    Returns:
        Dict with: loss, grad_norm, per_layer_angles, scaffold_fraction.
    """
    import mlx.core as mx  # noqa: I001  # pragma: no cover

    from mlx_grpo.loss.gar_ops import GARGradStepResult  # pragma: no cover
    from mlx_grpo.loss.gar_ops import capture_activations_forward  # pragma: no cover
    from mlx_grpo.loss.gar_ops import gar_gradient_step as _gar_step  # pragma: no cover

    total_loss = 0.0  # pragma: no cover
    total_grad_norm = 0.0  # pragma: no cover
    all_angles: dict[str, dict[str, float]] = {}  # pragma: no cover
    total_scaffold_tokens = 0  # pragma: no cover
    total_tokens = 0  # pragma: no cover

    for i, (scaffolded, naked) in enumerate(rollout_pairs):  # pragma: no cover
        scaf_tokens = scaffolded.completion_tokens
        naked_tokens = naked.completion_tokens

        total_scaffold_tokens += scaffolded.injected_token_count
        total_tokens += len(scaf_tokens) + len(naked_tokens)

        # Capture reference activations from scaffolded rollout (detached)
        ref_acts = capture_activations_forward(model, scaf_tokens, lora_layers)
        for key in ref_acts:
            ref_acts[key] = mx.stop_gradient(ref_acts[key])

        # Per-pair masks/weights
        pair_masks = masks[i] if masks and i < len(masks) else None
        pair_pw = proximity_weights[i] if proximity_weights and i < len(proximity_weights) else None

        result: GARGradStepResult = _gar_step(
            model=model,
            ref_acts=ref_acts,
            lora_layers=lora_layers,
            optimizer=optimizer,
            token_ids=naked_tokens,
            threshold_deg=threshold_deg,
            normalize=normalize,
            masks=pair_masks,
            proximity_weights=pair_pw,
            max_grad_norm=max_grad_norm,
        )

        total_loss += result.loss
        total_grad_norm = max(total_grad_norm, result.grad_norm)
        if result.per_layer_metrics:
            all_angles.update(result.per_layer_metrics)

    n = max(len(rollout_pairs), 1)  # pragma: no cover
    scaffold_frac = total_scaffold_tokens / max(total_tokens, 1)  # pragma: no cover

    return {  # pragma: no cover
        "loss": total_loss / n,
        "grad_norm": total_grad_norm,
        "per_layer_angles": all_angles,
        "scaffold_fraction": scaffold_frac,
    }


def pair_rollouts(
    rollouts: list[RolloutResult],
) -> list[tuple[RolloutResult, RolloutResult]]:
    """Pair rollouts into (scaffolded, naked) tuples.

    Expects group_size=2 per sample. Within each pair, index [0] is
    scaffolded (has injected scaffold tokens), index [1] is naked.
    If metadata flags are present, uses _view_is_naked; otherwise
    falls back to injected_token_count heuristic.
    """
    pairs: list[tuple[RolloutResult, RolloutResult]] = []
    for i in range(0, len(rollouts), 2):
        if i + 1 >= len(rollouts):
            break
        a, b = rollouts[i], rollouts[i + 1]

        # Determine which is scaffolded vs naked
        a_naked = a.metadata.get("_view_is_naked", a.injected_token_count == 0)
        b_naked = b.metadata.get("_view_is_naked", b.injected_token_count == 0)

        if a_naked and not b_naked:
            pairs.append((b, a))  # b=scaffolded, a=naked
        elif b_naked and not a_naked:
            pairs.append((a, b))  # a=scaffolded, b=naked
        else:
            # Fallback: more injected tokens = scaffolded
            if a.injected_token_count >= b.injected_token_count:
                pairs.append((a, b))
            else:
                pairs.append((b, a))

    return pairs
