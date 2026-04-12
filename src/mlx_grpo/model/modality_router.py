"""Modality-aware MoE routing — routes tokens to text vs vision experts.

Provides deterministic token routing for Mixture-of-Experts training
on multimodal (text + vision) inputs. Image-placeholder tokens are
routed to vision experts; all other tokens to text experts.

Two routing strategies:
  - ``hard``:  Each token → exactly one expert (binary mask).
  - ``soft``:  Returns a continuous weight per expert per token
               (useful for gated routing where gradients flow to both).

Usage:
    config = ModalityRouterConfig(image_token_id=32000, num_experts=2)
    router = TextVisualRouter(config)
    expert_ids = router.route(token_ids)      # [0, 0, 1, 1, 0, ...]
    weights   = router.soft_weights(token_ids)  # [[1,0],[1,0],[0,1],...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModalityRouterConfig:
    """Configuration for text/vision token routing.

    Attributes:
        image_token_id:    Token ID used as image placeholder in the
                           input sequence. -1 = use model default
                           (must be overridden before use).
        num_experts:       Total number of LoRA experts. Must be ≥ 2.
        text_expert_idx:   Expert index routed to for text tokens.
        vision_expert_idx: Expert index routed to for image tokens.
        strategy:          'hard' (argmax) or 'soft' (continuous weights).
        image_token_ids:   Additional image-related token IDs (e.g. BOI/EOI
                           markers). Union with image_token_id at route time.
    """

    image_token_id: int = -1
    num_experts: int = 2
    text_expert_idx: int = 0
    vision_expert_idx: int = 1
    strategy: str = "hard"  # hard | soft
    image_token_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.num_experts < 2:
            raise ValueError(f"num_experts must be ≥ 2, got {self.num_experts}")
        if self.text_expert_idx == self.vision_expert_idx:
            raise ValueError("text_expert_idx and vision_expert_idx must differ")
        if self.strategy not in ("hard", "soft"):
            raise ValueError(f"strategy must be 'hard' or 'soft', got {self.strategy!r}")

    def all_image_ids(self) -> set[int]:
        """Union of image_token_id and any additional image token IDs."""
        ids: set[int] = set(self.image_token_ids)
        if self.image_token_id != -1:
            ids.add(self.image_token_id)
        return ids


class TextVisualRouter:
    """Routes token IDs to text or vision expert indices.

    Stateless: safe to share across forward passes.
    """

    def __init__(self, config: ModalityRouterConfig) -> None:
        self.config = config
        self._image_ids = config.all_image_ids()

    def route(self, token_ids: list[int]) -> list[int]:
        """Return expert index for each token.

        Args:
            token_ids: Flat list of integer token IDs.

        Returns:
            List of expert indices (same length as token_ids).
        """
        vis = self.config.vision_expert_idx
        txt = self.config.text_expert_idx
        return [vis if t in self._image_ids else txt for t in token_ids]

    def soft_weights(self, token_ids: list[int]) -> list[list[float]]:
        """Return per-expert probability weights for each token.

        For 'hard' strategy, returns one-hot vectors.
        For 'soft' strategy, returns uniform weights over all experts
        (intended as a starting point for learned gating).

        Args:
            token_ids: Flat list of integer token IDs.

        Returns:
            List of length len(token_ids), each element a list of
            ``num_experts`` floats summing to 1.0.
        """
        n = self.config.num_experts
        vis = self.config.vision_expert_idx
        txt = self.config.text_expert_idx

        result = []
        for t in token_ids:
            if self.config.strategy == "soft":
                result.append([1.0 / n] * n)
            else:
                weights = [0.0] * n
                weights[vis if t in self._image_ids else txt] = 1.0
                result.append(weights)
        return result

    def image_mask(self, token_ids: list[int]) -> list[bool]:
        """Return True for each token that is an image placeholder."""
        return [t in self._image_ids for t in token_ids]

    def as_mlx_arrays(self, token_ids: list[int]) -> Any:
        """Convert route() output to MLX arrays if MLX is available.

        Returns (expert_ids_array, weights_array) or (None, None) if
        MLX is not installed.
        """
        try:
            import mlx.core as mx  # type: ignore[import]
        except ImportError:
            return None, None

        expert_ids = mx.array(self.route(token_ids))
        weights_list = self.soft_weights(token_ids)
        weights = mx.array(weights_list)
        return expert_ids, weights


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None (stdlib only; MLX optional in as_mlx_arrays)
# Usage: router = TextVisualRouter(ModalityRouterConfig(image_token_id=32000))
#        expert_ids = router.route(token_ids)
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
