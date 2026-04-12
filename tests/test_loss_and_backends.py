"""Tests for loss functions and backend registration."""

import pytest

from mlx_grpo.backends.base import BackendRegistry
from mlx_grpo.loss import (
    LossResult,
    bnpo_loss,
    compute_advantages,
    dr_grpo_loss,
    grpo_loss,
    sft_loss,
)

# ── Advantage computation ──


class TestAdvantages:
    def test_uniform_rewards(self):
        advs = compute_advantages([1.0, 1.0, 1.0, 1.0])
        assert all(abs(a) < 1e-6 for a in advs)

    def test_zero_centered(self):
        advs = compute_advantages([0.0, 0.5, 1.0])
        assert abs(sum(advs)) < 1e-6  # Should be zero-centered

    def test_ordering_preserved(self):
        advs = compute_advantages([0.0, 0.5, 1.0])
        assert advs[0] < advs[1] < advs[2]

    def test_clipping(self):
        advs = compute_advantages([0.0, 100.0], clip=1.5)
        assert all(-1.5 <= a <= 1.5 for a in advs)


# ── GRPO Loss ──


class TestGRPOLoss:
    def _make_data(self, g=4, t=10):
        import random

        rng = random.Random(123)
        lp = [[rng.gauss(-2.0, 0.3) for _ in range(t)] for _ in range(g)]
        rlp = [[rng.gauss(-2.0, 0.3) for _ in range(t)] for _ in range(g)]
        masks = [[True] * t for _ in range(g)]
        return lp, rlp, masks

    def test_basic(self):
        lp, rlp, masks = self._make_data()
        advs = [1.0, 0.5, -0.5, -1.0]
        result = grpo_loss(lp, rlp, advs, masks, beta=0.04)
        assert isinstance(result, LossResult)
        assert result.effective_tokens == 40

    def test_zero_beta(self):
        lp, rlp, masks = self._make_data()
        advs = [1.0, -1.0, 0.5, -0.5]
        result = grpo_loss(lp, rlp, advs, masks, beta=0.0)
        assert result.kl_penalty == 0.0

    def test_all_masked(self):
        lp, rlp, _ = self._make_data()
        masks = [[False] * 10 for _ in range(4)]
        result = grpo_loss(lp, rlp, [1.0] * 4, masks, beta=0.04)
        assert result.loss == 0.0
        assert result.effective_tokens == 0

    def test_same_policy_zero_kl(self):
        lp = [[float(x) for x in range(10)] for _ in range(2)]
        masks = [[True] * 10 for _ in range(2)]
        result = grpo_loss(lp, lp, [1.0, -1.0], masks, beta=0.04)
        # Same policy → log_ratio = 0 → KL = 0
        assert result.kl_divergence == pytest.approx(0.0, abs=1e-10)

    def test_ppo_clip(self):
        lp, rlp, masks = self._make_data()
        advs = [1.0, -1.0, 0.5, -0.5]
        result = grpo_loss(lp, rlp, advs, masks, beta=0.04, epsilon_clip=0.2)
        assert result.effective_tokens > 0


# ── BNPO Loss ──


class TestBNPOLoss:
    def test_separates_pos_neg(self):
        import random

        rng = random.Random(42)
        lp = [[rng.gauss(-2.0, 0.3) for _ in range(10)] for _ in range(4)]
        rlp = [[rng.gauss(-2.0, 0.3) for _ in range(10)] for _ in range(4)]
        masks = [[True] * 10 for _ in range(4)]
        advs = [1.0, 0.5, -0.5, -1.0]
        result = bnpo_loss(lp, rlp, advs, masks, beta=0.1, tau=0.1)
        assert result.effective_tokens == 40

    def test_empty(self):
        result = bnpo_loss([], [], [], [], beta=0.1)
        assert result.loss == 0.0


# ── DR-GRPO Loss ──


class TestDRGRPOLoss:
    def test_with_ref_and_kl(self):
        import random

        rng = random.Random(42)
        lp = [[rng.gauss(-2.0, 0.3) for _ in range(10)] for _ in range(4)]
        rlp = [[rng.gauss(-2.0, 0.3) for _ in range(10)] for _ in range(4)]
        masks = [[True] * 10 for _ in range(4)]
        advs = [1.0, 0.5, -0.5, -1.0]
        result = dr_grpo_loss(lp, rlp, advs, masks, beta=0.04)
        assert result.effective_tokens == 40
        assert result.kl_divergence >= 0.0


# ── SFT Loss ──


class TestSFTLoss:
    def test_basic(self):
        lp = [-1.0, -2.0, -3.0, -4.0, -5.0]
        mask = [True, True, True, False, False]
        loss = sft_loss(lp, mask)
        assert loss == pytest.approx(2.0)  # -(-1 + -2 + -3) / 3 = 2.0

    def test_all_masked(self):
        loss = sft_loss([-1.0, -2.0], [False, False])
        assert loss == 0.0


# ── Backend registration ──


class TestBackendRegistration:
    def test_mlx_registered(self):
        assert "mlx" in BackendRegistry.available()

    def test_all_backends_registered(self):
        available = BackendRegistry.available()
        # At minimum MLX should be registered; others depend on deps
        assert "mlx" in available
        # These register even without deps installed
        assert "pytorch" in available
        assert "vllm" in available
        assert "llamacpp" in available

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown backend"):
            BackendRegistry.get("nonexistent")
