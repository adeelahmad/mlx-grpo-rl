"""Tests for VisionConfig dataclass and YAML loader integration (task 4.3)."""

from __future__ import annotations

import os
import tempfile

import pytest

from mlx_grpo.config import VisionConfig, load_config

# ── VisionConfig defaults ─────────────────────────────────────────────


def test_vision_config_defaults():
    cfg = VisionConfig()
    assert cfg.image_processor == ""
    assert cfg.image_size == 336
    assert cfg.max_image_tokens == 256
    assert cfg.freeze_vision_encoder is True
    assert cfg.train_vision_encoder is False
    assert cfg.projector_lr_scale == 1.0
    assert cfg.image_token_id == -1
    assert cfg.patch_size == 14


def test_vision_config_is_enabled_false():
    assert not VisionConfig().is_enabled()


def test_vision_config_is_enabled_true():
    assert VisionConfig(image_processor="openai/clip-vit-base-patch16").is_enabled()


def test_vision_config_frozen():
    cfg = VisionConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.image_size = 224  # type: ignore[misc]


def test_vision_config_custom_values():
    cfg = VisionConfig(
        image_processor="model/path",
        image_size=224,
        max_image_tokens=128,
        freeze_vision_encoder=False,
        train_vision_encoder=True,
        projector_lr_scale=0.5,
        image_token_id=32000,
        patch_size=16,
    )
    assert cfg.image_size == 224
    assert cfg.max_image_tokens == 128
    assert not cfg.freeze_vision_encoder
    assert cfg.train_vision_encoder
    assert cfg.projector_lr_scale == 0.5
    assert cfg.image_token_id == 32000
    assert cfg.patch_size == 16


# ── YAML round-trip ───────────────────────────────────────────────────


def _tmp_yaml(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(content)
        return f.name


def test_yaml_no_vision_block_gives_defaults():
    """YAML without 'vision:' key → VisionConfig() defaults used."""
    path = _tmp_yaml("training:\n  model_path: test\n")
    try:
        config = load_config(path)
        assert not config.training.vision.is_enabled()
        assert config.training.vision.image_size == 336
    finally:
        os.unlink(path)


def test_yaml_vision_block_loaded():
    """YAML with 'vision:' block → fields parsed correctly."""
    path = _tmp_yaml(
        "training:\n  model_path: test\n"
        "vision:\n"
        "  image_processor: openai/clip-vit-base-patch16\n"
        "  image_size: 224\n"
        "  max_image_tokens: 128\n"
        "  freeze_vision_encoder: false\n"
    )
    try:
        config = load_config(path)
        v = config.training.vision
        assert v.is_enabled()
        assert v.image_processor == "openai/clip-vit-base-patch16"
        assert v.image_size == 224
        assert v.max_image_tokens == 128
        assert not v.freeze_vision_encoder
    finally:
        os.unlink(path)


def test_yaml_vision_unknown_keys_ignored():
    """Unknown keys in vision block are silently ignored."""
    path = _tmp_yaml("vision:\n  image_processor: test/model\n  unknown_future_key: 999\n")
    try:
        config = load_config(path)
        assert config.training.vision.is_enabled()
        assert config.training.vision.image_processor == "test/model"
    finally:
        os.unlink(path)


def test_yaml_vision_partial_override():
    """Only specified vision fields are overridden; rest use defaults."""
    path = _tmp_yaml("vision:\n  image_processor: hf/model\n  patch_size: 32\n")
    try:
        config = load_config(path)
        v = config.training.vision
        assert v.patch_size == 32
        assert v.image_size == 336  # default preserved
        assert v.max_image_tokens == 256  # default preserved
    finally:
        os.unlink(path)


def test_yaml_vision_train_encoder_flag():
    """train_vision_encoder bool parses correctly from YAML."""
    path = _tmp_yaml(
        "vision:\n"
        "  image_processor: test/model\n"
        "  freeze_vision_encoder: true\n"
        "  train_vision_encoder: true\n"
    )
    try:
        config = load_config(path)
        v = config.training.vision
        assert v.freeze_vision_encoder
        assert v.train_vision_encoder
    finally:
        os.unlink(path)


def test_vision_config_accessible_from_training():
    """TrainingConfig.vision is a VisionConfig instance."""
    path = _tmp_yaml("training:\n  model_path: x\n")
    try:
        config = load_config(path)
        assert isinstance(config.training.vision, VisionConfig)
    finally:
        os.unlink(path)
