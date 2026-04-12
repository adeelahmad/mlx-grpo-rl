"""MLX-GRPO: Template-driven GRPO training framework.

A thin, extensible framework for reinforcement learning from human feedback.
Define task types as Python classes or YAML configs. Extend with custom
rewards, backends, and callbacks.

    from mlx_grpo import TaskType, RolloutContext, BaseReward

Quick start::

    class MyType(TaskType, name="my_type"):
        def rollout(self, ctx):
            result = ctx.generate(max_tokens=512)
            ctx.capture("output", result)

        def reward(self, result, answer=None):
            return 1.0 if answer in result.completion_text else 0.0
"""

__version__ = "0.1.0"

from .backends.base import BackendRegistry, GenerationBackend
from .engine import (
    BatchContext,
    CallbackRegistry,
    GenerateResult,
    RolloutContext,
    RolloutExecutor,
    RolloutResult,
    Segment,
    SegmentSource,
)
from .rewards.base import BaseReward, reward_fn
from .types.base import ChatParams, TaskType

__all__ = [
    "BackendRegistry",
    "BaseReward",
    "BatchContext",
    "CallbackRegistry",
    "ChatParams",
    "GenerateResult",
    "GenerationBackend",
    "RolloutContext",
    "RolloutExecutor",
    "RolloutResult",
    "Segment",
    "SegmentSource",
    "TaskType",
    "reward_fn",
]

# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: pip install pyyaml (required), pip install mlx mlx-lm (for MLX backend)
# Run: python -m mlx_grpo train --config config/default.yml
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
