"""Gradient-level utilities for training stability.

Provides:
- Gradient projection / alignment between GRPO and SFT gradients
- Layer-selective gradient masking
- SFT anchor loss for preventing catastrophic forgetting
- Gradient norm computation and clipping helpers

These operate on nested dicts of numpy arrays (or MLX arrays in production).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GradientAlignmentResult:
    """Result of gradient alignment computation."""

    cosine_similarity: float
    projection_norm: float
    original_norm: float
    aligned_norm: float
    mode: str


def compute_gradient_norm(grads: dict[str, Any]) -> float:
    """Compute L2 norm of nested gradient dict.

    Works with any array-like that supports .flatten() and ** operator.
    Falls back to list flattening for plain lists.
    """
    total = 0.0
    for _key, val in grads.items():
        if isinstance(val, dict):
            total += compute_gradient_norm(val) ** 2
        elif hasattr(val, "flatten"):
            flat = val.flatten()
            total += sum(float(x) ** 2 for x in flat)
        elif isinstance(val, list | tuple):
            total += sum(float(x) ** 2 for x in _flatten(val))
    return math.sqrt(total)


def clip_gradient_norm(
    grads: dict[str, Any],
    max_norm: float,
) -> tuple[dict[str, Any], float]:
    """Clip gradient norm. Returns (clipped_grads, original_norm)."""
    norm = compute_gradient_norm(grads)
    if norm <= max_norm or norm == 0:
        return grads, norm

    scale = max_norm / norm
    return _scale_grads(grads, scale), norm


def compute_gradient_alignment(
    grpo_grads: dict[str, Any],
    sft_grads: dict[str, Any],
    mode: str = "project",
    alpha: float = 0.5,
) -> tuple[dict[str, Any], GradientAlignmentResult]:
    """Align GRPO gradients with SFT gradients to prevent forgetting.

    Modes:
        project   — Project GRPO gradient onto SFT gradient direction
        interpolate — Linear interpolation: alpha * grpo + (1-alpha) * sft
        kl        — KL-divergence based alignment (adds SFT gradient scaled by KL)
        reject    — Zero out GRPO gradient components opposing SFT direction

    Args:
        grpo_grads: Gradients from GRPO loss
        sft_grads: Gradients from SFT anchor loss
        mode: Alignment mode
        alpha: Interpolation weight (for interpolate mode)

    Returns:
        (aligned_grads, alignment_info)
    """
    # Flatten both gradient dicts for vector operations
    grpo_flat = _flatten_grads(grpo_grads)
    sft_flat = _flatten_grads(sft_grads)

    if not grpo_flat or not sft_flat:
        return grpo_grads, GradientAlignmentResult(0.0, 0.0, 0.0, 0.0, mode)

    # Ensure same length (pad shorter with zeros)
    max_len = max(len(grpo_flat), len(sft_flat))
    grpo_flat.extend([0.0] * (max_len - len(grpo_flat)))
    sft_flat.extend([0.0] * (max_len - len(sft_flat)))

    # Compute cosine similarity
    dot = sum(a * b for a, b in zip(grpo_flat, sft_flat, strict=True))
    grpo_norm = math.sqrt(sum(x**2 for x in grpo_flat))
    sft_norm = math.sqrt(sum(x**2 for x in sft_flat))
    cos_sim = dot / max(grpo_norm * sft_norm, 1e-12)

    if mode == "project":
        # Project GRPO onto SFT direction
        if sft_norm > 0:
            proj_scale = dot / (sft_norm**2)
            aligned = [proj_scale * s for s in sft_flat]
        else:
            aligned = grpo_flat
    elif mode == "interpolate":
        aligned = [alpha * g + (1 - alpha) * s for g, s in zip(grpo_flat, sft_flat, strict=True)]
    elif mode == "reject":
        # Zero out components opposing SFT direction
        if sft_norm > 0 and cos_sim < 0:
            proj_scale = dot / (sft_norm**2)
            aligned = [g - proj_scale * s for g, s in zip(grpo_flat, sft_flat, strict=True)]
        else:
            aligned = grpo_flat
    elif mode == "kl":
        # Add SFT gradient scaled by KL divergence
        kl_scale = max(0.0, -cos_sim)  # Scale up when opposing
        aligned = [g + kl_scale * s for g, s in zip(grpo_flat, sft_flat, strict=True)]
    else:
        aligned = grpo_flat

    aligned_norm = math.sqrt(sum(x**2 for x in aligned))
    proj_norm = math.sqrt(sum(x**2 for x in aligned)) if mode == "project" else 0.0

    # Reconstruct gradient dict with aligned values
    aligned_grads = _unflatten_grads(aligned, grpo_grads)

    result = GradientAlignmentResult(
        cosine_similarity=cos_sim,
        projection_norm=proj_norm,
        original_norm=grpo_norm,
        aligned_norm=aligned_norm,
        mode=mode,
    )
    return aligned_grads, result


def create_layer_gradient_mask(
    layer_names: list[str],
    freeze_patterns: list[str] | None = None,
    train_patterns: list[str] | None = None,
) -> dict[str, bool]:
    """Create a per-layer mask for selective gradient computation.

    Args:
        layer_names: All layer names in the model
        freeze_patterns: Layer name patterns to freeze (substring match)
        train_patterns: Layer name patterns to train (overrides freeze)

    Returns:
        Dict mapping layer name -> True (trainable) / False (frozen)
    """
    mask = {name: True for name in layer_names}

    if freeze_patterns:
        for name in layer_names:
            if any(p in name for p in freeze_patterns):
                mask[name] = False

    if train_patterns:
        for name in layer_names:
            if any(p in name for p in train_patterns):
                mask[name] = True

    return mask


def apply_gradient_mask(
    grads: dict[str, Any],
    mask: dict[str, bool],
) -> dict[str, Any]:
    """Zero out gradients for frozen layers."""
    result = {}
    for key, val in grads.items():
        if isinstance(val, dict):
            result[key] = apply_gradient_mask(val, mask)
        else:
            if mask.get(key, True):
                result[key] = val
            else:
                result[key] = _zeros_like(val)
    return result


# ── Helpers ──


def _flatten(val: Any) -> list[float]:
    """Flatten nested lists/arrays to a flat list of floats."""
    if isinstance(val, int | float):
        return [float(val)]
    if hasattr(val, "flatten"):
        return [float(x) for x in val.flatten()]
    if isinstance(val, list | tuple):
        out = []
        for item in val:
            out.extend(_flatten(item))
        return out
    return []


def _flatten_grads(grads: dict[str, Any]) -> list[float]:
    """Flatten a gradient dict to a single vector."""
    out: list[float] = []
    for key in sorted(grads.keys()):
        val = grads[key]
        if isinstance(val, dict):
            out.extend(_flatten_grads(val))
        else:
            out.extend(_flatten(val))
    return out


def _unflatten_grads(flat: list[float], template: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct gradient dict from flat vector using template for structure."""
    offset = [0]

    def _fill(src: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key in sorted(src.keys()):
            val = src[key]
            if isinstance(val, dict):
                out[key] = _fill(val)
            else:
                flat_val = _flatten(val)
                n = len(flat_val)
                out[key] = flat[offset[0] : offset[0] + n]
                offset[0] += n
        return out

    return _fill(template)


def _scale_grads(grads: dict[str, Any], scale: float) -> dict[str, Any]:
    """Scale all gradient values by a constant."""
    result = {}
    for key, val in grads.items():
        if isinstance(val, dict):
            result[key] = _scale_grads(val, scale)
        elif isinstance(val, list):
            result[key] = [x * scale for x in val]
        elif hasattr(val, "__mul__"):
            result[key] = val * scale
        else:
            result[key] = val
    return result


def _zeros_like(val: Any) -> Any:
    """Create a zero-valued version of the given value."""
    if isinstance(val, list):
        return [0.0] * len(val)
    if isinstance(val, int | float):
        return 0.0
    if hasattr(val, "shape"):
        try:
            import numpy as np

            return np.zeros_like(val)
        except ImportError:
            return val * 0
    return val
