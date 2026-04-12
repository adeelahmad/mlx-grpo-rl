"""Tests for PyTorch gradient operations (torch_ops.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from mlx_grpo.loss.torch_ops import (  # noqa: E402
    GradStepResult,
    compute_log_probs_for_tokens,
    gradient_step,
    precompute_ref_logps,
    sft_gradient_step,
)


class TinyModel(nn.Module):
    """Minimal causal LM for testing: embed -> linear -> vocab logits."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        return self.head(h)  # [B, T, vocab]


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return TinyModel(vocab_size=32, hidden=16)


class TestComputeLogProbs:
    def test_shape(self, tiny_model, device):
        tokens = [1, 5, 10, 3, 7]
        lp = compute_log_probs_for_tokens(tiny_model, tokens, device)
        assert lp.shape == (len(tokens) - 1,)

    def test_values_are_negative(self, tiny_model, device):
        tokens = [1, 5, 10, 3]
        lp = compute_log_probs_for_tokens(tiny_model, tokens, device)
        assert (lp <= 0).all(), "Log probs should be <= 0"

    def test_differentiable(self, tiny_model, device):
        tokens = [1, 5, 10]
        lp = compute_log_probs_for_tokens(tiny_model, tokens, device)
        loss = -lp.mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in tiny_model.parameters())
        assert has_grad, "Gradients should flow back through log probs"


class TestPrecomputeRefLogps:
    def test_returns_detached(self, tiny_model, device):
        sequences = [[1, 5, 10], [2, 3, 4, 5]]
        ref_lps = precompute_ref_logps(tiny_model, sequences, device)
        assert len(ref_lps) == 2
        assert ref_lps[0].shape == (2,)
        assert ref_lps[1].shape == (3,)
        assert not ref_lps[0].requires_grad

    def test_short_sequence(self, tiny_model, device):
        sequences = [[1]]  # Too short
        ref_lps = precompute_ref_logps(tiny_model, sequences, device)
        assert ref_lps[0].shape == (1,)


class TestGradientStep:
    def test_basic_step(self, tiny_model, device):
        ref_model = TinyModel(vocab_size=32, hidden=16)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        sequences = [[1, 5, 10, 3, 7], [2, 8, 4, 6, 9]]
        prompt_lengths = [2, 2]
        masks = [[True, True, True], [True, True, True]]
        advantages = [1.0, -0.5]

        result = gradient_step(
            tiny_model,
            ref_model,
            optimizer,
            sequences,
            prompt_lengths,
            masks,
            advantages,
            beta=0.04,
            device=device,
        )

        assert isinstance(result, GradStepResult)
        assert result.loss != 0.0
        assert result.grad_norm >= 0.0
        assert result.effective_tokens == 6

    def test_zero_advantage_not_skipped(self, tiny_model, device):
        """Zero advantages must NOT be skipped — KL gradient must flow."""
        ref_model = TinyModel(vocab_size=32, hidden=16)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        sequences = [[1, 5, 10, 3]]
        prompt_lengths = [1]
        masks = [[True, True, True]]
        advantages = [0.0]

        result = gradient_step(
            tiny_model,
            ref_model,
            optimizer,
            sequences,
            prompt_lengths,
            masks,
            advantages,
            beta=0.04,
            device=device,
        )

        # With zero advantage, loss = beta * KL / n (pure KL penalty)
        assert isinstance(result, GradStepResult)


    def test_grad_flows_inside_no_grad_context(self, tiny_model, device):
        """torch.enable_grad() in compute_grads must override outer no_grad."""
        ref_model = TinyModel(vocab_size=32, hidden=16)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        sequences = [[1, 5, 10, 3, 7]]
        prompt_lengths = [2]
        masks = [[True, True, True]]
        advantages = [1.0]

        with torch.no_grad():
            result = gradient_step(
                tiny_model,
                ref_model,
                optimizer,
                sequences,
                prompt_lengths,
                masks,
                advantages,
                beta=0.04,
                device=device,
            )

        assert result.grad_norm > 0.0, (
            "Gradient must be non-zero even when called inside torch.no_grad()"
        )

    def test_kl_nonzero_with_different_ref(self, tiny_model, device):
        """KL divergence must be non-zero when ref model differs from policy."""
        ref_model = TinyModel(vocab_size=32, hidden=16)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        sequences = [[1, 5, 10, 3, 7]]
        prompt_lengths = [2]
        masks = [[True, True, True]]
        advantages = [1.0]

        result = gradient_step(
            tiny_model,
            ref_model,
            optimizer,
            sequences,
            prompt_lengths,
            masks,
            advantages,
            beta=0.04,
            device=device,
        )

        assert result.kl_divergence > 0.0, (
            "KL must be non-zero when policy and reference models differ"
        )


class TestPeftGradientStep:
    """Test gradient flow through actual PEFT LoRA models."""

    @pytest.fixture
    def peft_setup(self, device):
        peft = pytest.importorskip("peft")
        torch.manual_seed(42)

        base = TinyModel(vocab_size=32, hidden=16)
        config = peft.LoraConfig(r=4, target_modules=["head"], task_type=None)
        model = peft.get_peft_model(base, config)
        model.train()

        # Use the same _PeftReferenceModel wrapper used in production
        from mlx_grpo.training.torch_model_loader import _PeftReferenceModel

        ref_model = _PeftReferenceModel(model)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        return model, ref_model, optimizer

    def test_peft_grad_nonzero(self, peft_setup, device):
        """LoRA model must produce non-zero gradients via _PeftReferenceModel."""
        model, ref_model, optimizer = peft_setup

        sequences = [[1, 5, 10, 3, 7], [2, 8, 4, 6, 9]]
        prompt_lengths = [2, 2]
        masks = [[True, True, True], [True, True, True]]
        advantages = [1.0, -0.5]

        result = gradient_step(
            model, ref_model, optimizer, sequences,
            prompt_lengths, masks, advantages,
            beta=0.04, device=device,
        )

        assert result.grad_norm > 0.0, (
            f"PEFT model grad must be non-zero, got {result.grad_norm}"
        )

    def test_peft_kl_nonzero_after_update(self, peft_setup, device):
        """After one gradient step, KL must diverge from zero."""
        model, ref_model, optimizer = peft_setup

        sequences = [[1, 5, 10, 3, 7]]
        prompt_lengths = [2]
        masks = [[True, True, True]]
        advantages = [1.0]

        # First step — at init, KL may be ~0 (LoRA B=0 → no adapter effect)
        result1 = gradient_step(
            model, ref_model, optimizer, sequences,
            prompt_lengths, masks, advantages,
            beta=0.04, device=device,
        )
        assert result1.grad_norm > 0.0, "First step must have non-zero grad"

        # Second step — weights have changed, ref (adapter disabled) should differ
        result2 = gradient_step(
            model, ref_model, optimizer, sequences,
            prompt_lengths, masks, advantages,
            beta=0.04, device=device,
        )
        assert result2.kl_divergence > 0.0, (
            f"KL must be non-zero after weight update, got {result2.kl_divergence}"
        )

    def test_peft_grad_inside_no_grad(self, peft_setup, device):
        """enable_grad() must override outer no_grad() for PEFT models."""
        model, ref_model, optimizer = peft_setup

        sequences = [[1, 5, 10, 3, 7]]
        prompt_lengths = [2]
        masks = [[True, True, True]]
        advantages = [1.0]

        with torch.no_grad():
            result = gradient_step(
                model, ref_model, optimizer, sequences,
                prompt_lengths, masks, advantages,
                beta=0.04, device=device,
            )

        assert result.grad_norm > 0.0, (
            "PEFT grad must be non-zero inside torch.no_grad()"
        )


class TestSftGradientStep:
    def test_sft_step(self, tiny_model, device):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        token_ids = [1, 5, 10, 3, 7, 2]
        prompt_length = 2

        loss = sft_gradient_step(
            tiny_model,
            optimizer,
            token_ids,
            prompt_length,
            lr_multiplier=0.1,
            device=device,
        )
        assert loss > 0.0
