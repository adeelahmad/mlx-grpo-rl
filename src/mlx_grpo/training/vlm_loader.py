"""VLM loader — AutoProcessor loading and vision encoder configuration.

Handles the VLM-specific side of model loading:
  - Loading AutoProcessor alongside the tokenizer
  - Freezing/unfreezing vision encoder layers per VisionConfig
  - Computing VLM-aware LoRA target modules

Usage:
    from mlx_grpo.training.vlm_loader import load_processor, freeze_vision_layers
    processor = load_processor(vision_cfg)   # None if VLM disabled
    freeze_vision_layers(model, vision_cfg)  # no-op if disabled
"""

from __future__ import annotations

import logging
from typing import Any

from ..config.vision import VisionConfig

_log = logging.getLogger(__name__)

# Module-name patterns that identify vision encoder submodules
_VISION_PATTERNS = (
    "vision_tower",
    "vision_encoder",
    "vision_model",
    "image_encoder",
    "clip_vision",
    "siglip",
)

# Module-name patterns that identify vision→LM projector/connector
_PROJECTOR_PATTERNS = (
    "mm_projector",
    "vision_projector",
    "connector",
    "resampler",
    "image_projection",
)


# ── Processor Loading ─────────────────────────────────────────────────


def load_processor(vision_cfg: VisionConfig) -> Any | None:
    """Load AutoProcessor for VLM mode.

    Returns None if VLM is disabled (image_processor not set) or if
    ``transformers`` is not installed.

    Args:
        vision_cfg: VisionConfig with image_processor path/HF model ID.

    Returns:
        AutoProcessor instance, or None.
    """
    if not vision_cfg.is_enabled():
        return None

    try:
        from transformers import AutoProcessor as _AutoProcessor
    except ImportError:
        _log.warning(
            "transformers not installed; cannot load AutoProcessor for %s. "
            "Install with: pip install transformers",
            vision_cfg.image_processor,
        )
        return None

    _log.info("Loading AutoProcessor from %s", vision_cfg.image_processor)
    return _AutoProcessor.from_pretrained(
        vision_cfg.image_processor,
        trust_remote_code=True,
    )


# ── Vision Encoder Freezing ───────────────────────────────────────────


def freeze_vision_layers(model: Any, vision_cfg: VisionConfig) -> int:
    """Freeze or unfreeze vision encoder layers based on VisionConfig.

    Args:
        model: MLX model with named submodules.
        vision_cfg: VisionConfig controlling freeze behaviour.

    Returns:
        Number of top-level vision modules processed.
    """
    if not vision_cfg.is_enabled():
        return 0

    should_freeze = vision_cfg.freeze_vision_encoder and not vision_cfg.train_vision_encoder
    count = 0

    for name, module in _iter_named_modules(model):
        name_l = name.lower()
        if not any(p in name_l for p in _VISION_PATTERNS):
            continue
        if should_freeze:
            if hasattr(module, "freeze"):
                module.freeze()
                count += 1
        else:
            if hasattr(module, "unfreeze"):
                module.unfreeze()
                count += 1

    action = "Froze" if should_freeze else "Unfroze"
    _log.info("%s %d vision module(s) in model", action, count)
    return count


# ── VLM-aware LoRA Target Computation ────────────────────────────────


def get_vlm_lora_targets(
    vision_cfg: VisionConfig,
    base_targets: list[str],
) -> list[str]:
    """Compute LoRA target modules for VLM training.

    When VLM is enabled and projector_lr_scale != 1.0, or when
    train_vision_encoder=True, adds projector layers to the target set
    so the vision→LM connector is fine-tuned.

    The vision encoder backbone is typically frozen; the projector
    bridges vision embeddings into the LM's token embedding space.

    Args:
        vision_cfg: VisionConfig controlling which layers to adapt.
        base_targets: LoRA target module paths for the LM backbone.

    Returns:
        Extended list of LoRA target module paths.
    """
    if not vision_cfg.is_enabled():
        return base_targets

    targets = list(base_targets)

    # Add projector targets when projector is not locked at default LR
    # (i.e., when the caller wants to fine-tune the projector separately)
    add_projector = not vision_cfg.freeze_vision_encoder or vision_cfg.projector_lr_scale != 1.0
    if add_projector:
        for proj in _PROJECTOR_PATTERNS:
            candidate = f"{proj}.linear"
            if candidate not in targets:
                targets.append(candidate)

    return targets


# ── Internal Helpers ──────────────────────────────────────────────────


def _iter_named_modules(model: Any) -> list[tuple[str, Any]]:
    """Return (name, module) pairs for all named submodules."""
    if hasattr(model, "named_modules"):
        return list(model.named_modules())
    return []


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: transformers (optional — guarded by try/except)
# Usage: processor = load_processor(cfg); freeze_vision_layers(model, cfg)
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
