"""Variable prefill task type — different group members get different scaffolding.

Training strategy: within each GRPO group, members receive varying amounts
of ground-truth prefill. This creates a natural difficulty gradient:
- High prefill -> easier task -> likely higher reward -> establishes baseline
- Low prefill  -> harder task -> model must reason independently

The ratio decays over training progress (curriculum scaffolding).

Usage in config:
    types:
      variable_prefill:
        reward: accuracy
        ratios: [0.5, 0.3, 0.1, 0.0]
        decay_rate: 1.5
"""

from __future__ import annotations

from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType

# Default ratios for group_size=4; extended cyclically for other sizes
DEFAULT_RATIOS = [0.50, 0.30, 0.10, 0.0]


class VariablePrefillType(TaskType, name="variable_prefill"):
    """Variable prefill across group members with curriculum decay."""

    def config(self) -> dict[str, Any]:
        return {
            "ratios": DEFAULT_RATIOS,
            "decay_rate": 1.5,
            "temperature": 0.7,
            "thinking_budget": 1024,
            "answer_max_tokens": 128,
            "stop_sequences": ["</think>"],
            "recovery_ratio": 0.8,
            "reward": "accuracy",
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=(
                "You are a reasoning assistant. Think step by step in <think> tags, "
                "then provide your answer."
            ),
            enable_thinking=True,
        )

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config
        ratios = cfg.get("ratios", DEFAULT_RATIOS)
        decay = cfg.get("decay_rate", 1.5)

        # Select ratio for this group member (cyclic if group > len(ratios))
        base_ratio = ratios[ctx.batch.group_index % len(ratios)]

        # Curriculum decay: reduce prefill over training
        progress = ctx.batch.progress
        ratio = base_ratio * max(0.0, 1.0 - progress * decay)

        # Inject prefill if ratio > 0 and answer available
        if ratio > 0.05 and ctx.answer:
            prefix = ctx.callbacks.slice_answer(ctx.answer, ratio, mode="prefix")
            if prefix:
                ctx.inject(prefix, masked=True, tag="scaffold")
                ctx.set_meta("prefill_ratio", ratio)
                ctx.set_meta("prefill_chars", len(prefix))

        # Phase 1: Thinking — reserve tokens for answer phase
        thinking_budget = cfg.get("thinking_budget", 1024)
        answer_reserve = cfg.get("answer_max_tokens", 128)
        if ctx.max_completion_tokens > 0:
            thinking_budget = min(
                thinking_budget,
                max(1, ctx.max_completion_tokens - answer_reserve),
            )
        thinking = ctx.generate(
            max_tokens=thinking_budget,
            stop=cfg.get("stop_sequences", ["</think>"]),
            temperature=cfg.get("temperature", 0.7),
            tag="thinking",
        )
        ctx.capture("thinking", thinking)

        # Recovery: force close thinking if it hit max_tokens
        if thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
            if ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.8)):
                ctx.inject("</think>\n", masked=True)
            else:
                return  # Abandon this rollout

        # Phase 2: Answer
        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 128),
            temperature=0.3,
            tag="answer",
        )
        ctx.capture("answer", answer)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        """Delegate to configured reward function."""
        if answer is None:
            return 0.0
        from ...rewards.base import BaseReward

        reward_name = self.config().get("reward", "accuracy")
        return BaseReward.get(reward_name).compute(result, answer)
