"""Adapter fusion for PyTorch — merge LoRA/DoRA weights into base model.

Mirrors the MLX adapter_migration.fuse_old_adapter() function exactly.
When resuming with a changed adapter config (rank change, DoRA→LoRA),
fuses the old adapter's delta into the frozen base weights in-memory.

Supports both float and 4/8-bit quantized (QLoRA) base weights via bitsandbytes.

Migrations handled:
  LoRA rank change  w_fused = w + old_scale * B.T @ A.T   (peft convention)
  DoRA -> LoRA      w_fused = m * normalize(w + old_scale * B.T @ A.T)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

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

    peft LoRALinear wraps the base linear as layer.base_layer.
    A bitsandbytes Linear8bitLt has .weight with device="meta" + .quant_state;
    we dequantize on read so downstream code always works in float space.
    """
    inner = getattr(layer, "base_layer", layer)
    if not hasattr(inner, "weight"):
        return None, None, False

    w_raw = inner.weight
    is_quantized = False

    # Check for bitsandbytes quantization (8-bit or NF4)
    if hasattr(inner, "quant_state") and inner.quant_state is not None:
        try:
            # bitsandbytes 8-bit linear: weight is a quantized linear8bit_state tuple
            # For simplicity, we denormalize the weight back to float.
            w_float = inner.weight.float()  # This triggers dequantization in-place
            is_quantized = True
            return w_float, inner, is_quantized
        except Exception as e:
            _log.warning("Bitsandbytes dequantize failed: %s", e)
            return None, None, False

    # Plain float weight
    w_float = w_raw.float() if w_raw.dtype != torch.float32 else w_raw.clone()

    return w_float, inner, is_quantized


def _set_float_weight(inner: Any, w_fused: torch.Tensor, is_quantized: bool) -> bool:
    """Write fused float weight back, re-quantizing if the layer was quantized."""
    if is_quantized:
        try:
            # bitsandbytes: convert back to quantized storage
            # For 8-bit, we quantize the float back into the original format
            from bitsandbytes.nn import Linear8bitLt

            if isinstance(inner, Linear8bitLt):
                # Re-quantize: move to CPU, quantize, move back to device
                w_cpu = w_fused.cpu().contiguous()
                from bitsandbytes.functional import quantize_blockwise

                # Use default blocksize 64 for 4-bit or 8-bit depending on quant_state
                quant_state = getattr(inner, "quant_state", None)
                blocksize = 64
                w_quant, quant_state = quantize_blockwise(w_cpu, blocksize=blocksize)
                inner.weight = w_quant.to(w_fused.device)
                inner.quant_state = quant_state
                return True
        except Exception as e:
            _log.warning("Bitsandbytes requantize failed: %s", e)
            return False

    # Plain float: write directly
    if hasattr(inner, "weight"):
        w_fused_on_device = w_fused.to(inner.weight.device)
        # Wrap in Parameter if it's a module parameter
        if isinstance(inner.weight, torch.nn.Parameter):
            inner.weight = torch.nn.Parameter(w_fused_on_device)
        else:
            inner.weight = w_fused_on_device
        return True
    return False


# ── Convention detection ──────────────────────────────────────────────


def _compute_delta(
    lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float, path: str
) -> torch.Tensor | None:
    """Compute the (out, in) weight delta, auto-detecting the LoRA convention.

    peft LoRA stores (following mlx_lm convention):
        lora_a: (in_features, rank)   forward: x @ lora_a @ lora_b
        lora_b: (rank, out_features)  delta  : lora_b.T @ lora_a.T -> (out, in)

    Custom DoRA may store:
        lora_a: (rank, in_features)   forward: lora_b @ lora_a @ x
        lora_b: (out_features, rank)  delta  : lora_b @ lora_a -> (out, in)

    Detection uses rank dimension: mlx_lm has lora_a.shape[1] == lora_b.shape[0].
    """
    if lora_a.shape[1] == lora_b.shape[0]:
        # mlx_lm/peft: A(in, rank), B(rank, out) -> delta = B.T @ A.T
        return scale * (lora_b.T @ lora_a.T)
    if lora_a.shape[0] == lora_b.shape[1]:
        # custom DoRA: A(rank, in), B(out, rank) -> delta = B @ A
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
    checkpoint_weights: dict[str, torch.Tensor],
    old_adapter_config: dict[str, Any],
    dropped_keys: set[str],
) -> int:
    """Fuse checkpoint adapter weights into the model's frozen base weights.

    For each adapter layer in the checkpoint, computes the effective weight
    and writes it back as the new frozen base. Works for both float and
    quantized (QLoRA via bitsandbytes) base models. The new adapter trains
    from zero delta.

    Args:
        model:               Model with new adapter layers already applied.
        checkpoint_weights:  Flat {key: tensor} from checkpoint (safetensors dict).
        old_adapter_config:  Parsed adapter_config.json (before overwrite).
        dropped_keys:        Keys in checkpoint absent from new model
                             (e.g. DoRA magnitude when switching to LoRA).
    Returns:
        Number of layers successfully fused.
    """
    lp = old_adapter_config.get("lora_parameters", {})
    old_scale: float = float(lp.get("scale", 1.0))

    # Magnitude-like dropped keys (DoRA m/magnitude no longer in new model)
    mag_keys = {k for k in dropped_keys if not k.endswith((".lora_a", ".lora_b"))}

    # Unique layer paths from lora_a / lora_b checkpoint entries
    # Maps checkpoint key -> (lora_a_key, lora_b_key, model_path)
    # Handles both MLX format (path.lora_a/b) and PEFT format (path.lora_A.weight/lora_B.weight)
    layer_lookup: dict[str, tuple[str, str, str]] = {}
    for k in checkpoint_weights:
        # PEFT: base_model.model.xxx.lora_A.weight or base_model.model.xxx.lora_B.weight
        # MLX: model.layers.xx.lora_a or model.layers.xx.lora_b
        if k.endswith(".lora_A.weight"):
            # PEFT LoRA A
            model_path = k[:-14]  # strip ".lora_A.weight"
            # Remove PEFT prefix from model path for model lookup
            lookup_path = model_path
            if lookup_path.startswith("base_model.model."):
                lookup_path = lookup_path.replace("base_model.model.", "", 1)
            # Construct corresponding lora_B key
            lora_b_key = k.replace(".lora_A.weight", ".lora_B.weight")
            layer_lookup[model_path] = (
                k,
                lora_b_key,
                lookup_path,
            )  # (lora_A_key, lora_B_key, model_path)
        elif k.endswith(".lora_B.weight"):
            pass  # Will be found via lora_A
        elif k.endswith(".lora_a"):
            # MLX LoRA A
            model_path = k[:-7]  # strip ".lora_a"
            # Construct corresponding lora_b key
            lora_b_key = k[:-7] + ".lora_b"
            layer_lookup[model_path] = (
                k,
                lora_b_key,
                model_path,
            )  # (lora_a_key, lora_b_key, model_path)
        elif k.endswith(".lora_b"):
            pass  # Will be found via lora_a

    fused = 0
    device = next(model.parameters()).device

    for ckpt_path, (lora_a_key, lora_b_key, model_path) in sorted(layer_lookup.items()):
        lora_a = checkpoint_weights.get(lora_a_key)
        lora_b = checkpoint_weights.get(lora_b_key)

        if lora_a is None or lora_b is None:
            continue

        # Ensure tensors are on the same device as model
        lora_a = lora_a.to(device)
        lora_b = lora_b.to(device)

        layer = _get_submodule(model, model_path)
        if layer is None:
            _log.debug("Layer not found: %s", model_path)
            continue

        w_float, inner, is_quantized = _get_float_weight(layer)
        if w_float is None:
            _log.debug("No accessible base weight on: %s", model_path)
            continue

        delta = _compute_delta(lora_a, lora_b, old_scale, model_path)
        if delta is None:
            continue

        # Find DoRA magnitude for this layer (dropped when switching to LoRA)
        # Check both checkpoint path formats
        mag = next(
            (
                checkpoint_weights[k].to(device)
                for k in mag_keys
                if k.startswith(f"{ckpt_path}.") or k.startswith(f"{model_path}.")
            ),
            None,
        )

        if mag is not None:
            # DoRA: w_fused = m * normalize(w + delta, row-wise)
            w_updated = w_float + delta
            row_norms = torch.sqrt(torch.sum(w_updated * w_updated, dim=1, keepdim=True))
            row_norms = torch.clamp(row_norms, min=1e-8)
            w_fused = mag[:, None] * w_updated / row_norms
        else:
            w_fused = w_float + delta

        if not _set_float_weight(inner, w_fused, is_quantized):
            _log.debug("Cannot write base weight on: %s", model_path)
            continue

        fused += 1

    if fused:
        _log.info("Fused %d adapter layers into frozen base weights", fused)

    return fused


def merge_adapter_into_base(model: Any, adapter_path: Any) -> int:
    """Load and fuse a single adapter checkpoint into the model's base weights.

    Public API for merging LoRA/DoRA checkpoints. Supports both:
    - MLX format: adapters.safetensors + adapter_config.json
    - PEFT format: adapter_model.safetensors + adapter_config.json

    Supports both float and quantized (QLoRA) base models.

    Returns: number of adapter layers fused into the base.
    """
    import json
    from pathlib import Path as _Path

    path = _Path(adapter_path)

    # Try MLX format first, then PEFT format
    weights_file = path / "adapters.safetensors"
    if not weights_file.exists():
        weights_file = path / "adapter_model.safetensors"
    if not weights_file.exists():
        _log.warning("merge_adapter_into_base: no adapter weights found in %s — skipping", path)
        return 0

    # Load safetensors checkpoint
    try:
        from safetensors.torch import load_file

        weights: dict[str, torch.Tensor] = load_file(str(weights_file))
    except ImportError:
        _log.warning("safetensors not available — cannot load checkpoint")
        return 0

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
    # Exclude both MLX format (lowercase) and PEFT format (capital + .weight)
    mag_candidates = {
        k
        for k in weights
        if not k.endswith((".lora_a", ".lora_b", ".lora_A.weight", ".lora_B.weight"))
    }

    return fuse_old_adapter(model, weights, cfg, dropped_keys=mag_candidates)


# ── End-of-File ──────────────────────────────────────────────────────
# Usage:
#   fuse_old_adapter(model, ckpt_weights, old_cfg, dropped_keys)  # resume migration
#   merge_adapter_into_base(model, "/path/to/adapter_dir")        # explicit merge
# ─────────────────────────────────────────────────────────────────────
