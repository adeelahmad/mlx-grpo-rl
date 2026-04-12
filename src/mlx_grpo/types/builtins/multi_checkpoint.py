"""Multi-checkpoint task type — generate at multiple points in a reasoning chain.

Different group members generate at different checkpoints:
- Member 0: generate only final checkpoint (easy -- full context provided)
- Member 1: generate last 2 checkpoints
- Member 2: generate first + last checkpoints
- Member 3: generate all checkpoints (hardest)

Degenerate detection short-circuits remaining checkpoints.

Requires sample metadata:
    sample.metadata["checkpoints"] = [
        {"text": "step 1 text", "label": "setup"},
        {"text": "step 2 text", "label": "compute"},
        {"text": "step 3 text", "label": "conclusion"},
    ]
"""

from __future__ import annotations

from typing import Any

from ...engine.context import RolloutContext
from ...engine.degenerate import DegenerateDetector
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType


def _build_generation_mask(
    group_index: int,
    group_size: int,
    n_checkpoints: int,
) -> list[bool]:
    """Determine which checkpoints this group member generates.

    Returns list of bools: True = generate, False = inject (prefill).

    Strategy (for group_size=4, n_checkpoints=3):
        Member 0: [F, F, T]  -- only final
        Member 1: [F, T, T]  -- last 2
        Member 2: [T, F, T]  -- first + last
        Member 3: [T, T, T]  -- all (hardest)
    """
    if n_checkpoints == 0:
        return []

    # Difficulty level: 0.0 (easiest) to 1.0 (hardest)
    difficulty = group_index / max(group_size - 1, 1)

    # Number of checkpoints to generate (at least 1, at most all)
    n_generate = max(1, int(round(difficulty * n_checkpoints)))

    # Place generated checkpoints: always include last, spread rest
    mask = [False] * n_checkpoints
    mask[-1] = True  # Always generate final checkpoint
    n_generate -= 1

    if n_generate > 0 and n_checkpoints > 1:
        # Spread remaining from front
        step = max(1, (n_checkpoints - 1) // max(n_generate, 1))
        idx = 0
        for _ in range(n_generate):
            if idx < n_checkpoints - 1:  # Don't re-mark last
                mask[idx] = True
            idx += step

    return mask


class MultiCheckpointType(TaskType, name="multi_checkpoint"):
    """Multi-checkpoint reasoning with per-member difficulty variation."""

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.7,
            "checkpoint_max_tokens": 256,
            "rep_window": 6,
            "rep_threshold": 3,
            "confidence_threshold": -5.0,
            "reward": "multiplicative",
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt="You are a step-by-step reasoning assistant.",
            enable_thinking=True,
        )

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config
        checkpoints = ctx.sample.get("checkpoints", [])

        if not checkpoints:
            # Fallback: single generate if no checkpoints provided
            ctx.generate(
                max_tokens=cfg.get("checkpoint_max_tokens", 256),
                temperature=cfg.get("temperature", 0.7),
                tag="answer",
            )
            return

        # Build generation mask for this group member
        mask = _build_generation_mask(
            ctx.batch.group_index,
            ctx.batch.group_size,
            len(checkpoints),
        )
        ctx.set_meta("generation_mask", mask)

        # Create degenerate detector
        detector = DegenerateDetector(
            rep_window=cfg.get("rep_window", 6),
            rep_threshold=cfg.get("rep_threshold", 3),
            confidence_threshold=cfg.get("confidence_threshold", -5.0),
        )

        # Execute each checkpoint
        for i, (checkpoint, should_generate) in enumerate(
            zip(checkpoints, mask, strict=True),
        ):
            cp_text = checkpoint["text"] if isinstance(checkpoint, dict) else str(checkpoint)

            if ctx.short_circuited:
                ctx.set_meta(f"checkpoint_{i}_status", "skipped")
                continue

            if should_generate:
                # Model must generate this checkpoint
                result = ctx.generate(
                    max_tokens=cfg.get("checkpoint_max_tokens", 256),
                    temperature=cfg.get("temperature", 0.7),
                    tag=f"checkpoint_{i}",
                    on_token=detector.check,
                )
                ctx.capture(f"checkpoint_{i}", result)
                ctx.set_meta(f"checkpoint_{i}_status", result.stop_reason)

                # Check for degenerate output
                if result.stop_reason in ("repetition", "low_confidence"):
                    ctx.short_circuited = True
                    ctx.set_meta("short_circuit_at", i)
                    ctx.set_meta("short_circuit_reason", result.stop_reason)
            else:
                # Inject ground truth (prefill) -- masked from loss
                ctx.inject(cp_text, masked=True, tag=f"prefill_{i}")
                ctx.set_meta(f"checkpoint_{i}_status", "prefilled")

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        """Multiplicative scoring across generated checkpoints."""
        if result.metadata.get("short_circuited"):
            return 0.0

        from ...rewards.base import BaseReward

        reward_name = self.config().get("reward", "multiplicative")
        return BaseReward.get(reward_name).compute(result, answer)
