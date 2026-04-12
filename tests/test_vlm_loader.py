"""Tests for VLM loader utility functions (tasks 5.1–5.3)."""

from __future__ import annotations

import json
import sys
import tempfile
from unittest.mock import MagicMock, patch

from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.config.vision import VisionConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.training import Trainer
from mlx_grpo.training.vlm_loader import (
    freeze_vision_layers,
    get_vlm_lora_targets,
    load_processor,
)

# ── load_processor ────────────────────────────────────────────────────


def test_load_processor_disabled_returns_none():
    """load_processor returns None when VLM is disabled."""
    result = load_processor(VisionConfig())
    assert result is None


def test_load_processor_no_transformers_returns_none(monkeypatch):
    """load_processor returns None gracefully when transformers is missing."""
    cfg = VisionConfig(image_processor="openai/clip-vit-base")
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = load_processor(cfg)
    assert result is None


def test_load_processor_calls_from_pretrained():
    """load_processor calls AutoProcessor.from_pretrained with correct args."""
    cfg = VisionConfig(image_processor="openai/clip-vit-base")
    mock_proc = MagicMock()
    mock_ap_cls = MagicMock(return_value=mock_proc)
    mock_ap_cls.from_pretrained = MagicMock(return_value=mock_proc)

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor = mock_ap_cls

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        result = load_processor(cfg)

    mock_ap_cls.from_pretrained.assert_called_once_with(
        "openai/clip-vit-base", trust_remote_code=True
    )
    assert result is mock_proc


def test_load_processor_empty_processor_path_returns_none():
    """load_processor returns None for any empty image_processor string."""
    assert load_processor(VisionConfig(image_processor="")) is None


# ── freeze_vision_layers ──────────────────────────────────────────────


def _make_model(*module_names: str) -> MagicMock:
    """Build a mock model whose named_modules() returns (name, mock) pairs."""
    modules = [(name, MagicMock()) for name in module_names]
    model = MagicMock()
    model.named_modules.return_value = modules
    return model, {name: mod for name, mod in modules}


def test_freeze_vision_disabled_noop():
    """freeze_vision_layers is a no-op when VLM is disabled."""
    count = freeze_vision_layers(object(), VisionConfig())
    assert count == 0


def test_freeze_vision_freezes_matching_modules():
    """Modules whose names contain a vision pattern are frozen."""
    cfg = VisionConfig(image_processor="m", freeze_vision_encoder=True)
    model, mods = _make_model("vision_tower", "language_model.layers.0.q_proj")

    count = freeze_vision_layers(model, cfg)

    mods["vision_tower"].freeze.assert_called_once()
    mods["language_model.layers.0.q_proj"].freeze.assert_not_called()
    assert count == 1


def test_freeze_vision_train_encoder_unfreezes():
    """train_vision_encoder=True unfreezes vision modules instead of freezing."""
    cfg = VisionConfig(
        image_processor="m",
        freeze_vision_encoder=False,
        train_vision_encoder=True,
    )
    model, mods = _make_model("vision_encoder", "lm.layer.0")

    count = freeze_vision_layers(model, cfg)

    mods["vision_encoder"].unfreeze.assert_called_once()
    mods["vision_encoder"].freeze.assert_not_called()
    assert count == 1


def test_freeze_vision_multiple_patterns():
    """All recognised vision module name patterns are matched."""
    cfg = VisionConfig(image_processor="m", freeze_vision_encoder=True)
    pattern_names = [
        "vision_tower",
        "vision_encoder",
        "vision_model",
        "image_encoder",
        "clip_vision_model",
        "siglip_layer",
    ]
    model, mods = _make_model(*pattern_names)

    count = freeze_vision_layers(model, cfg)

    assert count == len(pattern_names)
    for name in pattern_names:
        mods[name].freeze.assert_called_once()


def test_freeze_vision_model_without_named_modules():
    """Models without named_modules() return 0 without error."""
    cfg = VisionConfig(image_processor="m")
    count = freeze_vision_layers(object(), cfg)
    assert count == 0


# ── get_vlm_lora_targets ──────────────────────────────────────────────

BASE = ["self_attn.q_proj", "self_attn.v_proj"]


def test_lora_targets_vlm_disabled_returns_base():
    """Returns base targets unchanged when VLM is disabled."""
    result = get_vlm_lora_targets(VisionConfig(), BASE)
    assert result == BASE


def test_lora_targets_frozen_default_scale_no_projector():
    """No projector targets added when freeze=True and scale=1.0 (defaults)."""
    cfg = VisionConfig(image_processor="m")  # freeze=True, scale=1.0
    result = get_vlm_lora_targets(cfg, BASE)
    assert result == BASE


def test_lora_targets_adds_projector_when_not_frozen():
    """Projector targets added when train_vision_encoder=True (not frozen)."""
    cfg = VisionConfig(
        image_processor="m",
        freeze_vision_encoder=False,
        train_vision_encoder=True,
    )
    result = get_vlm_lora_targets(cfg, BASE)
    assert len(result) > len(BASE)
    # At least one projector pattern should appear
    assert any("projector" in t or "connector" in t or "resampler" in t for t in result)


def test_lora_targets_adds_projector_when_lr_scaled():
    """Projector targets added when projector_lr_scale != 1.0."""
    cfg = VisionConfig(image_processor="m", projector_lr_scale=0.5)
    result = get_vlm_lora_targets(cfg, BASE)
    assert len(result) > len(BASE)


def test_lora_targets_no_duplicates():
    """Projector targets are never added twice if already in base list."""
    cfg = VisionConfig(image_processor="m", projector_lr_scale=0.5)
    base_with_proj = BASE + ["mm_projector.linear"]
    result = get_vlm_lora_targets(cfg, base_with_proj)
    assert result.count("mm_projector.linear") == 1


def test_lora_targets_preserves_base_order():
    """Base targets appear first in the returned list."""
    cfg = VisionConfig(image_processor="m", projector_lr_scale=0.1)
    result = get_vlm_lora_targets(cfg, BASE)
    assert result[: len(BASE)] == BASE


# ── Trainer VLM wiring ─────────────────────────────────────────────────────


def _make_dataset_and_backend():
    """Create a minimal dataset + mock backend for Trainer tests."""
    from conftest import MockBackend

    rows = [{"prompt": "Q1", "answer": "A1", "type": "mcq", "metadata": {"ground_truth": "A"}}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    dataset = DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()
    return dataset, MockBackend()


class TestTrainerVLMWiring:
    """Trainer.__init__ wires VLM processor and vision layer freezing."""

    def test_processor_none_when_vision_disabled(self):
        """_processor is None when VisionConfig has no image_processor."""
        dataset, backend = _make_dataset_and_backend()
        config = TrainingConfig(iterations=1, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=backend)
        assert trainer._processor is None

    def test_processor_loaded_when_vision_enabled(self):
        """load_processor is called when VisionConfig.image_processor is set."""
        dataset, backend = _make_dataset_and_backend()
        cfg = VisionConfig(image_processor="openai/clip-vit-base")
        config = TrainingConfig(iterations=1, save_every=0, eval_every=0, vision=cfg)
        mock_proc = MagicMock()
        with patch(
            "mlx_grpo.training.vlm_loader.load_processor", return_value=mock_proc
        ) as load_mock:
            trainer = Trainer(config=config, dataset=dataset, backend=backend)
        load_mock.assert_called_once_with(cfg)
        assert trainer._processor is mock_proc

    def test_freeze_vision_called_when_model_and_vision(self):
        """freeze_vision_layers is called when model and vision config are present."""
        dataset, backend = _make_dataset_and_backend()
        cfg = VisionConfig(image_processor="openai/clip-vit-base")
        config = TrainingConfig(iterations=1, save_every=0, eval_every=0, vision=cfg)
        mock_model = MagicMock()
        with (
            patch("mlx_grpo.training.vlm_loader.load_processor", return_value=None),
            patch("mlx_grpo.training.vlm_loader.freeze_vision_layers") as freeze_mock,
        ):
            Trainer(config=config, dataset=dataset, backend=backend, model=mock_model)
        freeze_mock.assert_called_once_with(mock_model, cfg)

    def test_freeze_vision_not_called_without_model(self):
        """freeze_vision_layers is skipped when model=None (rollout-only mode)."""
        dataset, backend = _make_dataset_and_backend()
        cfg = VisionConfig(image_processor="openai/clip-vit-base")
        config = TrainingConfig(iterations=1, save_every=0, eval_every=0, vision=cfg)
        with (
            patch("mlx_grpo.training.vlm_loader.load_processor", return_value=None),
            patch("mlx_grpo.training.vlm_loader.freeze_vision_layers") as freeze_mock,
        ):
            Trainer(config=config, dataset=dataset, backend=backend, model=None)
        freeze_mock.assert_not_called()

    def test_vlm_wiring_skipped_when_vision_not_enabled(self):
        """Neither load_processor nor freeze is called for default VisionConfig."""
        dataset, backend = _make_dataset_and_backend()
        config = TrainingConfig(iterations=1, save_every=0, eval_every=0)
        with (
            patch("mlx_grpo.training.vlm_loader.load_processor") as load_mock,
            patch("mlx_grpo.training.vlm_loader.freeze_vision_layers") as freeze_mock,
        ):
            Trainer(config=config, dataset=dataset, backend=backend)
        load_mock.assert_not_called()
        freeze_mock.assert_not_called()
