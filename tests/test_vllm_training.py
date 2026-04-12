"""Tests for vLLM backend training validation.

vLLM is an inference-only backend and should raise NotImplementedError
when used for GRPO training (which requires gradient computation).
"""

import json
import tempfile

import pytest
from conftest import MockBackend

from mlx_grpo.backends.base import BackendRegistry
from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.engine.result import RolloutResult
from mlx_grpo.training import Trainer


class TestVLLMTrainingValidation:
    """Verify vLLM backend raises NotImplementedError during training."""

    def test_vllm_backend_name(self):
        """Verify vLLM backend is registered and has correct name."""
        try:
            backend = BackendRegistry.get("vllm", model_path="dummy")
        except ImportError:
            pytest.skip("vLLM not installed")

        assert backend.name == "vllm"

    def test_do_gradient_step_rejects_vllm(self):
        """Verify _do_gradient_step raises NotImplementedError with vLLM backend."""

        # Create a mock vLLM-like backend
        class VLLMMockBackend:
            @property
            def name(self) -> str:
                return "vllm"

            def generate(self, prefix_tokens, params):
                raise NotImplementedError

        config = TrainingConfig()
        # Create minimal dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Q", "answer": "A", "type": "mcq"}) + "\n")
            path = f.name
        dataset = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        dataset.load()

        trainer = Trainer(
            config=config,
            dataset=dataset,
            backend=VLLMMockBackend(),
        )

        # Create minimal rollout data
        rollout = RolloutResult(segments=[])

        # Verify NotImplementedError is raised with helpful message
        with pytest.raises(NotImplementedError, match="vLLM backend is inference-only"):
            trainer._do_gradient_step([rollout], [0.0])

    def test_training_with_pytorch_backend_allowed(self):
        """Verify PyTorch backend is allowed (even if not fully functional in test)."""
        config = TrainingConfig()
        # Create minimal dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Q", "answer": "A", "type": "mcq"}) + "\n")
            path = f.name
        dataset = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        dataset.load()

        # Use MockBackend which simulates pytorch (no vllm rejection)
        trainer = Trainer(
            config=config,
            dataset=dataset,
            backend=MockBackend(),
        )

        rollout = RolloutResult(segments=[])

        # Should not raise NotImplementedError (may raise other errors, but not the vllm one)
        try:
            trainer._do_gradient_step([rollout], [0.0])
        except NotImplementedError as e:
            # Only fail if it's the vllm-specific error
            assert "vLLM" not in str(e), f"vLLM error raised unexpectedly: {e}"

    def test_none_backend_allowed(self):
        """Verify None backend is allowed (evaluation/rollout-only mode)."""
        config = TrainingConfig()
        # Create minimal dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Q", "answer": "A", "type": "mcq"}) + "\n")
            path = f.name
        dataset = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        dataset.load()

        trainer = Trainer(
            config=config,
            dataset=dataset,
            backend=None,
        )

        rollout = RolloutResult(segments=[])

        # None backend should go through reward-only path
        loss, kl, grad_norm = trainer._do_gradient_step([rollout], [0.0])
        assert loss == 0.0
        assert kl == 0.0
        assert grad_norm == 0.0

    def test_vllm_error_message_helpful(self):
        """Verify error message suggests alternatives to user."""

        class VLLMMockBackend:
            @property
            def name(self) -> str:
                return "vllm"

        config = TrainingConfig()
        # Create minimal dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Q", "answer": "A", "type": "mcq"}) + "\n")
            path = f.name
        dataset = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        dataset.load()

        trainer = Trainer(
            config=config,
            dataset=dataset,
            backend=VLLMMockBackend(),
        )

        rollout = RolloutResult(segments=[])

        with pytest.raises(NotImplementedError) as exc_info:
            trainer._do_gradient_step([rollout], [0.0])

        error_msg = str(exc_info.value)
        # Verify helpful suggestions in error message
        assert "pytorch" in error_msg.lower() or "mlx" in error_msg.lower()
        assert "inference" in error_msg.lower() or "eval" in error_msg.lower()

    def test_vllm_allowed_for_rollout_only(self):
        """Verify vLLM works for rollout-only evaluation (no gradient step)."""
        # This test verifies the happy path where vLLM is only used for
        # generation, not for training. The NotImplementedError is only
        # raised if _do_gradient_step is actually called.

        class VLLMMockBackend:
            @property
            def name(self) -> str:
                return "vllm"

            def generate(self, prefix_tokens, params):
                from mlx_grpo.backends.base import GenerateOutput

                return GenerateOutput(
                    text="result",
                    tokens=[1, 2, 3],
                )

            def encode(self, text: str):
                return [1, 2, 3]

            def decode(self, tokens):
                return "text"

            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]

        config = TrainingConfig()
        # Create minimal dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Q", "answer": "A", "type": "mcq"}) + "\n")
            path = f.name
        dataset = DatasetLoader(DatasetConfig(path=path, shuffle=False))
        dataset.load()

        # Trainer with vLLM backend, but no model/optimizer means
        # _do_gradient_step won't be called (evaluation-only)
        trainer = Trainer(
            config=config,
            dataset=dataset,
            backend=VLLMMockBackend(),
        )

        # Verify executor was created (for generation)
        assert trainer.executor is not None
        assert trainer.backend.name == "vllm"

        # Verify model is None (evaluation-only)
        assert trainer.model is None
        assert trainer.optimizer is None
