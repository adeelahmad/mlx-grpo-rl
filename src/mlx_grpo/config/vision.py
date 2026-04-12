"""VisionConfig — configuration for Vision-Language Model (VLM) support."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Vision encoder and processor configuration for VLM training.

    When image_processor is empty (default), VLM support is disabled and
    is_enabled() returns False. Set image_processor to a HuggingFace model
    ID or local path to enable VLM mode.

    Fields:
        image_processor:       Path or HF model ID for AutoProcessor. Empty = disabled.
        image_size:            Target image size (square) after preprocessing.
        max_image_tokens:      Max token budget for encoded image features.
        freeze_vision_encoder: Freeze vision encoder layers during training.
        train_vision_encoder:  Explicitly train vision encoder (overrides freeze).
        projector_lr_scale:    LR multiplier for vision projector layers.
        image_token_id:        Token ID used as image placeholder (-1 = model default).
        patch_size:            Vision encoder patch size in pixels.
    """

    image_processor: str = ""
    image_size: int = 336
    max_image_tokens: int = 256
    freeze_vision_encoder: bool = True
    train_vision_encoder: bool = False
    projector_lr_scale: float = 1.0
    image_token_id: int = -1
    patch_size: int = 14

    def is_enabled(self) -> bool:
        """Return True when VLM support is active (image_processor is set)."""
        return bool(self.image_processor)
