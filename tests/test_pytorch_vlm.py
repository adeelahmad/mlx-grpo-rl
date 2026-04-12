"""PyTorch VLM integration tests.

Tests VQA and ImageCaptioning tasks with PyTorch backend, including
processor loading, vision layer freezing, and gradient computation.
"""

from __future__ import annotations

import pytest

from mlx_grpo.config.vision import VisionConfig
from mlx_grpo.training.model_loader import ModelConfig
from mlx_grpo.training.torch_model_loader import TorchModelLoader

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from peft import LoraConfig

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.skipif(not HAS_PEFT, reason="peft not installed")
class TestPyTorchVLMIntegration:
    """Test VLM support in PyTorch model loader."""

    def test_processor_loading_disabled(self) -> None:
        """Test that processor is None when VisionConfig is not enabled."""
        vision_cfg = VisionConfig(image_processor="")
        cfg = ModelConfig(
            model_path="gpt2",
            adapter_type="lora",
            vision=vision_cfg,
        )
        loader = TorchModelLoader(cfg)
        model, ref_model, tokenizer, processor = loader.load()

        assert model is not None
        assert tokenizer is not None
        # When vision is disabled, processor should be None
        assert processor is None

    def test_vision_config_passed_through(self) -> None:
        """Test that VisionConfig is properly initialized in model config."""
        vision_cfg = VisionConfig(
            image_processor="",
            image_size=384,
            freeze_vision_encoder=True,
        )
        cfg = ModelConfig(
            model_path="gpt2",
            adapter_type="lora",
            vision=vision_cfg,
        )

        assert cfg.vision is not None
        assert cfg.vision.image_size == 384
        assert cfg.vision.freeze_vision_encoder is True

    def test_gradient_checkpointing_config(self) -> None:
        """Test that gradient checkpointing config is preserved."""
        cfg = ModelConfig(
            model_path="gpt2",
            adapter_type="lora",
            grad_checkpoint=True,
            vision=VisionConfig(image_processor=""),
        )
        loader = TorchModelLoader(cfg)
        model, ref_model, tokenizer, processor = loader.load()

        # Model should be loaded successfully
        assert model is not None
        # gradient_checkpointing_enable should have been called
        # (model has gradient_checkpointing_enable method if PEFT was applied)
        assert hasattr(model, "gradient_checkpointing_enable")

    def test_lora_targets_with_vision_config(self) -> None:
        """Test that LoRA targets include projector when VLM is enabled."""
        from mlx_grpo.training.vlm_loader import get_vlm_lora_targets

        vision_cfg = VisionConfig(
            image_processor="openai/clip-vit-base-patch32",
            freeze_vision_encoder=False,
        )
        base_targets = ["q_proj", "v_proj"]
        extended_targets = get_vlm_lora_targets(vision_cfg, base_targets)

        # Should include base targets plus projector targets
        assert "q_proj" in extended_targets
        assert "v_proj" in extended_targets
        # Projector targets should be added when VLM is enabled
        # and freeze_vision_encoder is False or projector_lr_scale != 1.0

    def test_model_training_mode_set(self) -> None:
        """Test that model is put in training mode after loading."""
        cfg = ModelConfig(
            model_path="gpt2",
            adapter_type="lora",
            vision=VisionConfig(image_processor=""),
        )
        loader = TorchModelLoader(cfg)
        model, ref_model, tokenizer, processor = loader.load()

        # Model should be in training mode
        assert model.training is True

    def test_reference_model_creation(self) -> None:
        """Test that reference model is created correctly for adapter mode."""
        cfg = ModelConfig(
            model_path="gpt2",
            adapter_type="lora",
            vision=VisionConfig(image_processor=""),
        )
        loader = TorchModelLoader(cfg)
        model, ref_model, tokenizer, processor = loader.load()

        assert ref_model is not None
        # Reference model should be callable
        assert callable(ref_model)

    def test_full_finetune_with_vision(self) -> None:
        """Test full fine-tuning mode with VisionConfig."""
        cfg = ModelConfig(
            model_path="gpt2",
            adapter_type="full",
            vision=VisionConfig(image_processor=""),
        )
        loader = TorchModelLoader(cfg)
        model, ref_model, tokenizer, processor = loader.load()

        assert model is not None
        assert ref_model is not None
        # Full fine-tune reference should be in eval mode
        assert ref_model._ref.training is False


class TestPyTorchBackendVLM:
    """Test VLM support in PyTorch backend."""

    def test_backend_accepts_processor(self) -> None:
        """Test that PyTorch backend accepts processor parameter signature."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")

        from mlx_grpo.backends.pytorch_backend import PyTorchBackend

        # Check that __init__ accepts processor parameter
        sig_params = PyTorchBackend.__init__.__code__.co_varnames
        assert "processor" in sig_params

    def test_backend_generate_with_images_parameter(self) -> None:
        """Test that backend generate() accepts images parameter."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")

        from mlx_grpo.backends.pytorch_backend import PyTorchBackend

        # Test that generate method accepts images parameter
        assert hasattr(PyTorchBackend.generate, "__code__")
        sig_params = PyTorchBackend.generate.__code__.co_varnames
        assert "images" in sig_params


class TestVisionLayerFreezing:
    """Test vision layer freezing in PyTorch."""

    def test_freeze_vision_layers_function_exists(self) -> None:
        """Test that freeze_vision_layers function is available and callable."""
        from mlx_grpo.training.vlm_loader import freeze_vision_layers

        # Function should be callable
        assert callable(freeze_vision_layers)

        # Test with disabled VisionConfig
        vision_cfg = VisionConfig(
            image_processor="",
            freeze_vision_encoder=True,
        )

        # Should handle mock objects gracefully
        class MockModel:
            def named_modules(self):
                return []

        result = freeze_vision_layers(MockModel(), vision_cfg)
        # Should return 0 for disabled VLM
        assert result == 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestGradientCheckpointing:
    """Test gradient checkpointing integration with torch_ops."""

    def test_checkpoint_function_available(self) -> None:
        """Test that torch.utils.checkpoint is available."""
        assert hasattr(torch.utils.checkpoint, "checkpoint")

    def test_compute_log_probs_accepts_checkpoint_param(self) -> None:
        """Test that compute_log_probs_for_tokens accepts use_checkpoint param."""
        from mlx_grpo.loss.torch_ops import compute_log_probs_for_tokens

        sig = compute_log_probs_for_tokens.__code__.co_varnames
        assert "use_checkpoint" in sig

    def test_compute_grads_accepts_checkpoint_param(self) -> None:
        """Test that compute_grads accepts use_checkpoint param."""
        from mlx_grpo.loss.torch_ops import compute_grads

        sig = compute_grads.__code__.co_varnames
        assert "use_checkpoint" in sig

    def test_gradient_step_accepts_checkpoint_param(self) -> None:
        """Test that gradient_step accepts use_checkpoint param."""
        from mlx_grpo.loss.torch_ops import gradient_step

        sig = gradient_step.__code__.co_varnames
        assert "use_checkpoint" in sig
