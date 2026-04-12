"""Adapter migration — fuse checkpoint adapter into base model weights.

When resuming with a changed adapter config (rank change, DoRA->LoRA), instead
of discarding learned weights we absorb the old adapter's effective delta into
the frozen base weights in-memory. The new adapter (already applied to the
model) then starts from zero delta on top of this improved base — zero loss of
learned knowledge, regardless of the structural change.

Supports both float and 4/8-bit quantized base weights (QLoRA). Quantized
layers are dequantized, fused, then requantized back in-place.

Migrations handled:
  LoRA rank change  w_fused = w + old_scale * B.T @ A.T   (mlx_lm convention)
  DoRA -> LoRA      w_fused = m * normalize(w + old_scale * B.T @ A.T)
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


# ── Layer navigation ─────────────────────────────────────────────────


def _get_submodule(model: Any, dotted: str) -> Any | None:
    """Resolve a dotted path through a model tree, handling list indices."""
    obj: Any = model
    for part in dotted.split("."):
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


# ── Weight read / write (float + quantized) ──────────────────────────


def _get_float_weight(layer: Any) -> tuple[Any, Any, bool]:
    """Return (w_float, inner_layer, is_quantized) for a LoRA/DoRA layer.

    mlx_lm wraps the base linear as layer.linear (LoRALinear pattern).
    A QuantizedLinear has .scales and .biases alongside .weight; we
    dequantize on read so downstream code always works in float space.
    """
    inner = getattr(layer, "linear", layer)
    if not hasattr(inner, "weight"):
        return None, None, False

    w_raw = inner.weight
    if hasattr(inner, "scales") and hasattr(inner, "biases"):
        # QLoRA: base is a QuantizedLinear — dequantize to full precision
        try:
            import mlx.core as mx

            group_size: int = getattr(inner, "group_size", 64)
            bits: int = getattr(inner, "bits", 4)
            w_float = mx.dequantize(w_raw, inner.scales, inner.biases, group_size, bits)
            return w_float, inner, True
        except Exception as e:
            _log.warning("Dequantize failed: %s", e)
            return None, None, False

    return w_raw, inner, False


def _set_float_weight(inner: Any, w_fused: Any, is_quantized: bool) -> bool:
    """Write fused float weight back, requantizing if the layer was quantized."""
    if is_quantized:
        try:
            import mlx.core as mx

            group_size: int = getattr(inner, "group_size", 64)
            bits: int = getattr(inner, "bits", 4)
            new_w, new_scales, new_biases = mx.quantize(w_fused, group_size, bits)
            inner.weight = new_w
            inner.scales = new_scales
            inner.biases = new_biases
            return True
        except Exception as e:
            _log.warning("Requantize failed: %s", e)
            return False

    if hasattr(inner, "weight"):
        inner.weight = w_fused
        return True
    return False


# ── Convention detection ──────────────────────────────────────────────


def _compute_delta(lora_a: Any, lora_b: Any, scale: float, path: str) -> Any | None:
    """Compute the (out, in) weight delta, auto-detecting the LoRA convention.

    mlx_lm LoRALinear stores:
        lora_a: (in_features, rank)   forward: x @ lora_a @ lora_b
        lora_b: (rank, out_features)  delta  : lora_b.T @ lora_a.T -> (out, in)

    Custom DoRALinear stores:
        lora_a: (rank, in_features)   forward: lora_b @ lora_a @ x
        lora_b: (out_features, rank)  delta  : lora_b @ lora_a -> (out, in)

    Detection uses the shared "rank" dimension — the small dim that appears
    as lora_a.shape[1] == lora_b.shape[0] (mlx_lm) or
    as lora_a.shape[0] == lora_b.shape[1] (custom).
    This works for both float and quantized base weights.
    """
    if lora_a.shape[1] == lora_b.shape[0]:
        # mlx_lm: A(in, rank), B(rank, out) -> delta = B.T @ A.T
        return scale * (lora_b.T @ lora_a.T)
    if lora_a.shape[0] == lora_b.shape[1]:
        # custom DoRALinear: A(rank, in), B(out, rank) -> delta = B @ A
        return scale * (lora_b @ lora_a)
    _log.warning(
        "Ambiguous LoRA convention for %s (A=%s, B=%s) — skipping",
        path,
        lora_a.shape,
        lora_b.shape,
    )
    return None


# ── Public API ───────────────────────────────────────────────────────


def fuse_old_adapter(
    model: Any,
    checkpoint_weights: dict[str, Any],
    old_adapter_config: dict[str, Any],
    dropped_keys: set[str],
) -> int:
    """Fuse checkpoint adapter weights into the model's frozen base weights.

    For each adapter layer in the checkpoint, computes the effective weight
    and writes it back as the new frozen base. Works for both float and
    quantized (QLoRA) base models. The new adapter trains from zero delta.

    Args:
        model:               Model with new adapter layers already applied.
        checkpoint_weights:  Flat {key: array} from adapters.safetensors.
        old_adapter_config:  Parsed adapter_config.json (before overwrite).
        dropped_keys:        Keys in checkpoint absent from new model
                             (e.g. DoRA magnitude when switching to LoRA).
    Returns:
        Number of layers successfully fused.
    """
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
    except ImportError:
        _log.warning("MLX unavailable — skipping adapter fusion")
        return 0

    lp = old_adapter_config.get("lora_parameters", {})
    old_scale: float = float(lp.get("scale", 1.0))

    # Magnitude-like dropped keys (DoRA m/magnitude no longer in new model)
    mag_keys = {k for k in dropped_keys if not k.endswith((".lora_a", ".lora_b"))}

    # Unique layer paths from lora_a / lora_b checkpoint entries
    layer_paths: set[str] = set()
    for k in checkpoint_weights:
        if k.endswith((".lora_a", ".lora_b")):
            layer_paths.add(k.rsplit(".", 1)[0])

    fused = 0
    for path in sorted(layer_paths):
        lora_a = checkpoint_weights.get(f"{path}.lora_a")
        lora_b = checkpoint_weights.get(f"{path}.lora_b")
        if lora_a is None or lora_b is None:
            continue

        layer = _get_submodule(model, path)
        if layer is None:
            _log.debug("Layer not found: %s", path)
            continue

        w_float, inner, is_quantized = _get_float_weight(layer)
        if w_float is None:
            _log.debug("No accessible base weight on: %s", path)
            continue

        delta = _compute_delta(lora_a, lora_b, old_scale, path)
        if delta is None:
            continue

        # Find DoRA magnitude for this layer (dropped when switching to LoRA)
        mag = next(
            (checkpoint_weights[k] for k in mag_keys if k.startswith(f"{path}.")),
            None,
        )

        if mag is not None:
            # DoRA: w_fused = m * normalize(w + delta, row-wise)
            w_updated = w_float + delta
            row_norms = mx.sqrt(mx.sum(w_updated * w_updated, axis=1, keepdims=True))
            w_fused = mag[:, None] * w_updated / mx.maximum(row_norms, mx.array(1e-8))
        else:
            w_fused = w_float + delta

        if not _set_float_weight(inner, w_fused, is_quantized):
            _log.debug("Cannot write base weight on: %s", path)
            continue

        fused += 1

    if fused:
        all_w = [v for _, v in tree_flatten(model.parameters()) if hasattr(v, "shape")]
        for i in range(0, len(all_w), 50):
            mx.eval(*all_w[i : i + 50])
        _log.info("Fused %d adapter layers into frozen base weights", fused)

    return fused


def merge_adapter_into_base(model: Any, adapter_path: Any) -> int:
    """Load and fuse a single adapter checkpoint into the model's base weights.

    Public API for ``--merge-adapters`` / ``merge_adapters`` config.
    Supports LoRA and DoRA checkpoints, float and quantized (QLoRA) bases.

    The adapter directory must contain ``adapters.safetensors`` and ideally
    ``adapter_config.json`` (for the correct LoRA scale).

    Returns: number of adapter layers fused into the base.
    """
    import json
    from pathlib import Path as _Path

    path = _Path(adapter_path)
    weights_file = path / "adapters.safetensors"
    if not weights_file.exists():
        _log.warning("merge_adapter_into_base: no adapters.safetensors in %s — skipping", path)
        return 0

    try:
        import mlx.core as mx
    except ImportError:
        _log.warning("MLX unavailable — skipping merge")
        return 0

    weights: dict[str, Any] = dict(mx.load(str(weights_file)))

    cfg: dict[str, Any] = {}
    cfg_file = path / "adapter_config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
    else:
        _log.warning(
            "merge_adapter_into_base: no adapter_config.json in %s — scale defaults to 1.0",
            path,
        )

    # Treat all non-lora_a/lora_b keys as potential magnitude keys so the
    # DoRA formula is applied automatically for DoRA checkpoints.
    mag_candidates = {k for k in weights if not k.endswith((".lora_a", ".lora_b"))}

    return fuse_old_adapter(model, weights, cfg, dropped_keys=mag_candidates)


# ── End-of-File ──────────────────────────────────────────────────────
# Usage:
#   fuse_old_adapter(model, ckpt_weights, old_cfg, dropped_keys)  # resume migration
#   merge_adapter_into_base(model, "/path/to/adapter_dir")        # explicit merge
# ─────────────────────────────────────────────────────────────────────
