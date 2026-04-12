"""LoRAMoEModel — wraps a base MLX model with MoE gating.

Architecture:
  1. pool input embeddings (single base forward avoided by using embed_tokens)
  2. gate produces weights via GatingNetwork
  3. weights injected as Python attrs on all LoRAMoELinear layers
  4. base model forward runs (using pre-set gate weights in each layer)

Schedule helpers return plain Python floats — safe inside value_and_grad
traced scope because they never call .item() or mx.eval().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.loader import MoEConfig

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


if HAS_MLX:
    from .lora_moe import GatingNetwork, LoRAMoELinear, pool_embeddings

    class LoRAMoEModel(nn.Module):
        """Base model + gating network for LoRA-MoE training.

        Gate routes on mean-pooled input embeddings (single forward pass,
        fully differentiable, no hooks needed).

        Usage:
            moe = LoRAMoEModel(base_model, moe_cfg)
            moe.freeze_base()       # freeze base, unfreeze experts + gate
            logits = moe(input_ids) # traced by nn.value_and_grad
        """

        def __init__(self, base: Any, moe_cfg: MoEConfig) -> None:
            super().__init__()
            self._base = base
            self._moe_cfg = moe_cfg
            self.current_step: int = 0
            self.current_task_type: str | None = None
            self._last_gate_logits: mx.array | None = None

            # Build type → expert index lookup
            self._type_to_expert: dict[str, int] = {
                et.type_name: et.expert_index for et in moe_cfg.expert_types if et.type_name
            }

            # Collect all MoE layers (set during apply_moe_lora_to_model)
            self._moe_layers: list[LoRAMoELinear] = []
            for _, m in base.named_modules():
                if isinstance(m, LoRAMoELinear):
                    self._moe_layers.append(m)

            # Determine embedding dimension from embed_tokens
            embed_dim = self._get_embed_dim()
            self._gate = GatingNetwork(
                embed_dim=embed_dim,
                hidden_dim=moe_cfg.gate.hidden_dim,
                num_experts=moe_cfg.num_experts,
            )

        def _get_embed_dim(self) -> int:
            """Infer embedding dim from the model's embed_tokens layer."""
            embed_tokens = getattr(getattr(self._base, "model", self._base), "embed_tokens", None)
            if embed_tokens is not None and hasattr(embed_tokens, "weight"):
                return embed_tokens.weight.shape[1]
            # Fallback: check lm_head
            lm_head = getattr(self._base, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                return lm_head.weight.shape[1]
            return 2048  # Reasonable default

        def __call__(self, input_ids: mx.array, **kwargs: Any) -> mx.array:
            """Forward pass: gate on embeddings → inject weights → base forward."""
            # Step 1: pool input embeddings for routing signal
            base_model = getattr(self._base, "model", self._base)
            embed_tokens = getattr(base_model, "embed_tokens", None)
            if embed_tokens is not None:
                embeds = embed_tokens(input_ids)  # [1, T, D]
                pooled = pool_embeddings(embeds, self._moe_cfg.gate.pool_method)  # [D]
            else:
                # Fallback: zero pooled (gate learns from bias only)
                pooled = mx.zeros((self._get_embed_dim(),))

            # Step 2: compute gate weights
            temp = self.get_temperature()
            noise = self.get_noise_std()
            gate_weights, gate_logits = self._gate(pooled, temp, noise)

            # Step 3: supervised blend during warmup
            sup_w = self.get_supervised_weight()
            if sup_w > 0.0 and self.current_task_type in self._type_to_expert:
                target_idx = self._type_to_expert[self.current_task_type]
                # Build one-hot: compare range [0..N) to target index
                n = self._moe_cfg.num_experts
                one_hot = (mx.arange(n) == target_idx).astype(mx.float32)
                gate_weights = (1.0 - sup_w) * gate_weights + sup_w * one_hot

            # Step 4: inject gate weights into all MoE layers (lazy graph node)
            for layer in self._moe_layers:
                layer._gate_weights = gate_weights

            # Step 5: store logits for balance loss (outside v&g scope)
            self._last_gate_logits = gate_logits

            # Step 6: run base model (uses injected gate weights in each layer)
            return self._base(input_ids, **kwargs)

        # ── Schedule helpers — Python floats, safe inside traced scope ──

        def get_temperature(self) -> float:
            """Linear decay from temperature → temperature_min."""
            g = self._moe_cfg.gate
            if self.current_step >= g.temperature_decay_steps:
                return g.temperature_min
            frac = self.current_step / max(g.temperature_decay_steps, 1)
            return g.temperature - frac * (g.temperature - g.temperature_min)

        def get_noise_std(self) -> float:
            """Linear decay from noise_std → 0."""
            g = self._moe_cfg.gate
            if self.current_step >= g.noise_decay_steps or g.noise_std <= 0.0:
                return 0.0
            frac = self.current_step / max(g.noise_decay_steps, 1)
            return g.noise_std * (1.0 - frac)

        def get_supervised_weight(self) -> float:
            """1.0 → linear → 0.0 over supervised/transition steps."""
            r = self._moe_cfg.routing
            s = self.current_step
            if s < r.supervised_steps:
                return 1.0
            if s >= r.supervised_steps + r.transition_steps:
                return 0.0
            frac = (s - r.supervised_steps) / max(r.transition_steps, 1)
            return 1.0 - frac

        def get_routing_mode(self) -> str:
            """Return current routing phase as string."""
            r = self._moe_cfg.routing
            s = self.current_step
            if s < r.supervised_steps:
                return "supervised"
            if s < r.supervised_steps + r.transition_steps:
                return "transition"
            return "learned"

        # ── State management ──

        def set_scale(self, s: float) -> None:
            """Set LoRA scale on all MoE layers (0.0 = reference, 1.0 = policy)."""
            for layer in self._moe_layers:
                layer.scale = s

        def step(self) -> None:
            """Advance schedule counter and clear per-step transient state."""
            self.current_step += 1
            self._last_gate_logits = None
            for layer in self._moe_layers:
                layer._gate_weights = None

        def freeze_base(self) -> None:
            """Freeze base model; unfreeze LoRA experts and gate network."""
            self._base.freeze()
            # Unfreeze all expert parameters
            for layer in self._moe_layers:
                for expert in layer.experts:
                    expert.unfreeze()
            # Unfreeze gate
            self._gate.unfreeze()

else:

    class LoRAMoEModel:  # type: ignore[no-redef]
        """Requires MLX."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("LoRAMoEModel requires MLX: pip install mlx")


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: mlx (required)
# Usage: moe = LoRAMoEModel(base_model, moe_cfg); moe.freeze_base()
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
