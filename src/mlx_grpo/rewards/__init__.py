"""Reward system — pluggable reward functions."""

from .accuracy import AccuracyReward, FormatReward  # noqa: F401 — triggers registration
from .base import BaseReward, reward_fn
from .efficiency import EfficiencyReward, ThinkingEfficiencyReward  # noqa: F401
from .exam import ExamReward  # noqa: F401 — triggers registration
from .hierarchical import HierarchicalReward, RewardBreakdown  # noqa: F401
from .identity import IdentityReward  # noqa: F401 — triggers registration
from .llm_judge import LLMJudgeReward  # noqa: F401 — triggers registration
from .lotto import LottoReward  # noqa: F401 — triggers registration
from .math import MathReward  # noqa: F401 — triggers registration
from .multiplicative import MultiplicativeReward  # noqa: F401
from .reasoning import ReasoningReward  # noqa: F401
from .vision import (  # noqa: F401 — triggers VLM reward registration
    VisionBleuReward,
    VisionExactMatchReward,
    VisionRougeLReward,
    VisionTokenF1Reward,
)

__all__ = [
    "BaseReward",
    "reward_fn",
    "AccuracyReward",
    "FormatReward",
    "ExamReward",
    "IdentityReward",
    "ReasoningReward",
    "EfficiencyReward",
    "ThinkingEfficiencyReward",
    "HierarchicalReward",
    "RewardBreakdown",
    "MultiplicativeReward",
    "LLMJudgeReward",
    "LottoReward",
    "MathReward",
    "VisionExactMatchReward",
    "VisionTokenF1Reward",
    "VisionBleuReward",
    "VisionRougeLReward",
]
