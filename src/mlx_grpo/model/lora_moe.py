"""LoRA Mixture-of-Experts layers.

LoRAExpert, GatingNetwork, pool_embeddings, LoRAMoELinear,
apply_moe_lora_to_model.

Gradient note: gate_weights set on LoRAMoELinear inside LoRAMoEModel.__call__
stores a lazy MLX array — gradients flow gate → expert combination correctly.
"""

from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


if HAS_MLX:

    class LoRAExpert(nn.Module):
        """Single LoRA expert — additive delta only (no base weight copy)."""

        def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
            super().__init__()
            self.scale = alpha / max(rank, 1)
            # lora_a: [rank, in_features], lora_b: [out_features, rank]
            self.lora_a = mx.random.normal((rank, in_features)) * (0.01 / max(rank, 1) ** 0.5)
            self.lora_b = mx.zeros((out_features, rank))

        def __call__(self, x: mx.array) -> mx.array:
            """Return additive delta: (lora_b @ lora_a @ x.T).T * scale."""
            # x: [..., in_features] — handle arbitrary leading dims
            return (x @ self.lora_a.T @ self.lora_b.T) * self.scale

    class GatingNetwork(nn.Module):
        """Two-layer MLP that routes input embeddings to expert weights."""

        def __init__(self, embed_dim: int, hidden_dim: int, num_experts: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_experts)

        def __call__(
            self,
            pooled: mx.array,
            temperature: float = 1.0,
            noise_std: float = 0.0,
        ) -> tuple[mx.array, mx.array]:
            """Return (weights [N], logits [N])."""
            h = nn.relu(self.fc1(pooled))
            logits = self.fc2(h)  # [num_experts]
            if noise_std > 0.0:
                noise = mx.random.normal(logits.shape) * noise_std
                logits = logits + noise
            weights = mx.softmax(logits / max(temperature, 1e-6), axis=-1)
            return weights, logits

    def pool_embeddings(embeds: mx.array, method: str = "mean") -> mx.array:
        """Reduce [1, T, D] → [D] embedding vector.

        Args:
            embeds: Shape [1, T, D] from embed_tokens
            method: mean | first | last | max
        """
        e = embeds[0]  # [T, D]
        if method == "first":
            return e[0]
        elif method == "last":
            return e[-1]
        elif method == "max":
            return mx.max(e, axis=0)
        else:  # mean
            return mx.mean(e, axis=0)

    class LoRAMoELinear(nn.Module):
        """Linear + N LoRA experts with dynamic gating.

        scale=0.0 → base output only (reference model mode).
        _gate_weights=None → base output only (pre-routing).
        """

        def __init__(
            self,
            weight: mx.array,
            bias: mx.array | None,
            experts: list[LoRAExpert],
        ) -> None:
            super().__init__()
            self.weight = weight
            if bias is not None:
                self.bias = bias
            self._has_bias = bias is not None
            self.experts = experts
            self.scale: float = 1.0
            self._gate_weights: mx.array | None = None

        @classmethod
        def from_linear(
            cls,
            linear: Any,
            experts: list[LoRAExpert],
        ) -> LoRAMoELinear:
            """Create MoE layer from frozen nn.Linear, copying weight/bias."""
            bias = getattr(linear, "bias", None)
            return cls(linear.weight, bias, experts)

        def __call__(self, x: mx.array) -> mx.array:
            """Forward: base_out + scale * Σᵢ gate_weights[i] * expert_i(x)."""
            base_out = x @ self.weight.T
            if self._has_bias:
                base_out = base_out + self.bias

            if self.scale == 0.0 or self._gate_weights is None:
                return base_out

            # Weighted sum of expert deltas
            delta = mx.zeros_like(base_out)
            for i, expert in enumerate(self.experts):
                delta = delta + self._gate_weights[i] * expert(x)

            return base_out + self.scale * delta

    def apply_moe_lora_to_model(
        model: Any,
        num_experts: int,
        target_modules: tuple[str, ...],
        num_layers: int | None,
        expert_ranks: list[int],
        expert_alphas: list[float],
        dropout: float = 0.0,
    ) -> None:
        """Replace target nn.Linear modules with LoRAMoELinear in-place."""
        layers = getattr(getattr(model, "model", None), "layers", [])
        if not layers:
            return

        if num_layers is not None and num_layers > 0:
            target_layers = layers[-num_layers:]
        else:
            target_layers = layers

        for layer in target_layers:
            for path in target_modules:
                parts = path.split(".")
                # Navigate to parent object
                parent = layer
                for part in parts[:-1]:
                    parent = getattr(parent, part, None)
                    if parent is None:
                        break
                if parent is None:
                    continue

                attr = parts[-1]
                linear = getattr(parent, attr, None)
                if linear is None or not hasattr(linear, "weight"):
                    continue

                w = linear.weight  # [out, in]
                in_f, out_f = w.shape[1], w.shape[0]
                experts = [
                    LoRAExpert(in_f, out_f, expert_ranks[i], expert_alphas[i])
                    for i in range(num_experts)
                ]
                moe_linear = LoRAMoELinear.from_linear(linear, experts)
                setattr(parent, attr, moe_linear)

else:
    # Stubs for non-MLX environments (allow import for config/testing)

    class LoRAExpert:  # type: ignore[no-redef]
        """Requires MLX."""

    class GatingNetwork:  # type: ignore[no-redef]
        """Requires MLX."""

    class LoRAMoELinear:  # type: ignore[no-redef]
        """Requires MLX."""

    def pool_embeddings(embeds: Any, method: str = "mean") -> Any:  # type: ignore[misc]
        raise ImportError("pool_embeddings requires MLX")

    def apply_moe_lora_to_model(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        raise ImportError("apply_moe_lora_to_model requires MLX")


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: mlx (required for actual use; stubs provided for import)
# Usage: apply_moe_lora_to_model(model, num_experts=3, ...)
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
