"""Unit tests for LoRA-MoE layer classes.

Requires MLX. Tests use small tensors and verify math correctness.
"""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core", reason="MLX not installed")
nn = pytest.importorskip("mlx.nn", reason="MLX not installed")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn_mlx  # noqa: E402

from mlx_grpo.model.lora_moe import (  # noqa: E402
    GatingNetwork,
    LoRAExpert,
    LoRAMoELinear,
    pool_embeddings,
)

# ── LoRAExpert ────────────────────────────────────────────────────────


class TestLoRAExpert:
    def test_zero_init_output_zero(self):
        """lora_b=zeros → delta is zero at init."""
        expert = LoRAExpert(in_features=8, out_features=4, rank=2, alpha=4.0)
        x = mx.ones((1, 8))
        delta = expert(x)
        mx.eval(delta)
        assert delta.shape == (1, 4)
        assert mx.allclose(delta, mx.zeros((1, 4))).item()

    def test_scale_computation(self):
        """scale = alpha / rank."""
        expert = LoRAExpert(in_features=8, out_features=4, rank=4, alpha=16.0)
        assert expert.scale == 4.0

    def test_nonzero_after_lora_b_set(self):
        """Manual lora_b → nonzero delta."""
        expert = LoRAExpert(in_features=4, out_features=2, rank=2, alpha=2.0)
        expert.lora_b = mx.ones((2, 2))
        expert.lora_a = mx.ones((2, 4))
        x = mx.ones((1, 4))
        delta = expert(x)
        mx.eval(delta)
        # delta = (x @ lora_a.T @ lora_b.T) * scale = (ones @ ones @ ones) * 1.0
        # x @ lora_a.T = [1,2] (batch dot with 2-row lora_a)
        # Actually x[1,4] @ lora_a.T[4,2] = [1,2] (sum across 4 dims = 4 each)
        # Result[1,2] @ lora_b.T[2,2] = [2,2] → then multiply by scale=1.0
        assert delta.shape == (1, 2)
        assert (delta > 0).all().item()

    def test_batch_dims(self):
        """Handles [T, in_features] input."""
        expert = LoRAExpert(in_features=8, out_features=4, rank=2, alpha=4.0)
        x = mx.ones((5, 8))  # T=5 tokens
        delta = expert(x)
        mx.eval(delta)
        assert delta.shape == (5, 4)


# ── GatingNetwork ─────────────────────────────────────────────────────


class TestGatingNetwork:
    def test_output_shape(self):
        gate = GatingNetwork(embed_dim=16, hidden_dim=8, num_experts=3)
        pooled = mx.ones((16,))
        weights, logits = gate(pooled)
        mx.eval(weights, logits)
        assert weights.shape == (3,)
        assert logits.shape == (3,)

    def test_weights_sum_to_one(self):
        gate = GatingNetwork(embed_dim=16, hidden_dim=8, num_experts=4)
        pooled = mx.random.normal((16,))
        weights, _ = gate(pooled, temperature=1.0)
        mx.eval(weights)
        total = mx.sum(weights).item()
        assert abs(total - 1.0) < 1e-5

    def test_high_temperature_uniform(self):
        """Very high temperature → weights approach 1/N."""
        gate = GatingNetwork(embed_dim=8, hidden_dim=4, num_experts=4)
        pooled = mx.zeros((8,))
        weights, _ = gate(pooled, temperature=1000.0)
        mx.eval(weights)
        for w in weights.tolist():
            assert abs(w - 0.25) < 0.05  # allow some deviation from bias

    def test_noise_changes_output(self):
        """noise_std > 0 → different weights (stochastic)."""
        gate = GatingNetwork(embed_dim=8, hidden_dim=4, num_experts=3)
        pooled = mx.ones((8,))
        mx.random.seed(42)
        w1, _ = gate(pooled, noise_std=0.0)
        mx.random.seed(99)
        w2, _ = gate(pooled, noise_std=5.0)
        mx.eval(w1, w2)
        # With large noise, outputs differ
        assert not mx.allclose(w1, w2).item()


# ── pool_embeddings ───────────────────────────────────────────────────


class TestPoolEmbeddings:
    """Test embedding pooling methods."""

    def setup_method(self):
        # [1, 4, 8] — batch=1, T=4 tokens, D=8 dims
        self.embeds = mx.array([[[float(i)] * 8 for i in range(4)]])

    def test_mean_pool(self):
        result = pool_embeddings(self.embeds, "mean")
        mx.eval(result)
        assert result.shape == (8,)
        # mean of [0,1,2,3] = 1.5
        assert abs(result[0].item() - 1.5) < 1e-5

    def test_first_pool(self):
        result = pool_embeddings(self.embeds, "first")
        mx.eval(result)
        assert mx.allclose(result, mx.zeros((8,))).item()

    def test_last_pool(self):
        result = pool_embeddings(self.embeds, "last")
        mx.eval(result)
        assert abs(result[0].item() - 3.0) < 1e-5

    def test_max_pool(self):
        result = pool_embeddings(self.embeds, "max")
        mx.eval(result)
        assert abs(result[0].item() - 3.0) < 1e-5

    def test_default_is_mean(self):
        r1 = pool_embeddings(self.embeds)
        r2 = pool_embeddings(self.embeds, "mean")
        mx.eval(r1, r2)
        assert mx.allclose(r1, r2).item()


# ── LoRAMoELinear ─────────────────────────────────────────────────────


class TestLoRAMoELinear:
    def _make_layer(self, in_f=8, out_f=4, num_experts=2, rank=2):
        weight = mx.random.normal((out_f, in_f))
        bias = mx.zeros((out_f,))
        experts = [LoRAExpert(in_f, out_f, rank, alpha=float(rank)) for _ in range(num_experts)]
        return LoRAMoELinear(weight, bias, experts)

    def test_scale_zero_ignores_experts(self):
        """scale=0 → pure base output."""
        layer = self._make_layer()
        layer.scale = 0.0
        # Manually set non-zero gate weights to verify they're ignored
        layer._gate_weights = mx.array([0.5, 0.5])
        # Set lora_b to something nonzero
        for e in layer.experts:
            e.lora_b = mx.ones_like(e.lora_b)

        x = mx.ones((1, 8))
        out = layer(x)
        expected = x @ layer.weight.T + layer.bias
        mx.eval(out, expected)
        assert mx.allclose(out, expected).item()

    def test_no_gate_weights_gives_base(self):
        """_gate_weights=None → base output only."""
        layer = self._make_layer()
        layer._gate_weights = None
        x = mx.ones((1, 8))
        out = layer(x)
        expected = x @ layer.weight.T + layer.bias
        mx.eval(out, expected)
        assert mx.allclose(out, expected).item()

    def test_gate_weights_add_delta(self):
        """Non-zero gate_weights add expert deltas to base output."""
        layer = self._make_layer(num_experts=2)
        # Set lora_b to all ones so delta is nonzero
        for e in layer.experts:
            e.lora_b = mx.ones_like(e.lora_b)
            e.lora_a = mx.ones_like(e.lora_a)
        layer._gate_weights = mx.array([0.5, 0.5])
        layer.scale = 1.0

        x = mx.ones((1, 8))
        base = x @ layer.weight.T + layer.bias
        out = layer(x)
        mx.eval(base, out)
        # output != base (expert deltas added)
        assert not mx.allclose(out, base).item()

    def test_from_linear(self):
        """from_linear copies weight and bias correctly."""
        linear = nn_mlx.Linear(8, 4)
        mx.eval(linear.parameters())
        experts = [LoRAExpert(8, 4, 2, 4.0) for _ in range(2)]
        layer = LoRAMoELinear.from_linear(linear, experts)
        mx.eval(layer.weight)
        assert mx.allclose(layer.weight, linear.weight).item()

    def test_output_shape(self):
        layer = self._make_layer(in_f=16, out_f=8, num_experts=3)
        layer._gate_weights = mx.array([0.3, 0.4, 0.3])
        x = mx.ones((2, 16))  # batch=2
        out = layer(x)
        mx.eval(out)
        assert out.shape == (2, 8)


# ── AdapterType enum ─────────────────────────────────────────────────


def test_adapter_type_lora_moe():
    from mlx_grpo.training.model_loader import AdapterType

    at = AdapterType.from_str("lora_moe")
    assert at == AdapterType.LORA_MOE
    assert at.value == "lora_moe"
