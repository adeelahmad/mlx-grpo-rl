"""Reward system — metaprogramming-driven reward registry.

Every reward function is a subclass of ``BaseReward``.
Auto-registered via ``__init_subclass__``::

    class AccuracyReward(BaseReward, name="accuracy"):
        def compute(self, result, answer): ...

    # Now available:
    BaseReward.get("accuracy").compute(result, answer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..engine.result import RolloutResult


class BaseReward(ABC):
    """Base class for reward functions with auto-registration.

    Subclass with ``name=`` to register::

        class MyReward(BaseReward, name="my_reward"):
            def compute(self, result, answer=None):
                return 1.0 if "correct" in result.completion_text else 0.0
    """

    _registry: ClassVar[dict[str, type[BaseReward]]] = {}

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls._registry[name] = cls

    @classmethod
    def get(cls, name: str) -> BaseReward:
        """Instantiate a registered reward by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown reward {name!r}. Available: {available}")
        return cls._registry[name]()

    @classmethod
    def get_with_kwargs(cls, name: str, **kwargs: Any) -> BaseReward:
        """Instantiate a registered reward with constructor kwargs."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown reward {name!r}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_rewards(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def register(cls, name: str, reward_cls: type[BaseReward]) -> None:
        cls._registry[name] = reward_cls

    @abstractmethod
    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        """Compute reward for a rollout result.

        Args:
            result: The complete rollout result
            answer: Ground truth answer (if available)

        Returns:
            Reward value (typically 0.0 to 1.0)
        """
        ...


# ── Convenience: reward from a plain function ──


def reward_fn(name: str):
    """Decorator to register a plain function as a reward::

    @reward_fn("my_reward")
    def my_reward(result, answer=None):
        return 1.0
    """

    def decorator(fn):
        cls_dict = {"compute": lambda self, result, answer=None: fn(result, answer)}
        cls = type(f"{name.title()}Reward", (BaseReward,), cls_dict)
        BaseReward.register(name, cls)
        return fn

    return decorator


# ── Convenience: reward from a lambda in YAML ──


def register_lambda_reward(name: str, fn) -> None:
    """Register a lambda/callable as a reward function."""
    cls_dict = {"compute": lambda self, result, answer=None: fn(result, answer)}
    cls = type(f"{name.title()}LambdaReward", (BaseReward,), cls_dict)
    BaseReward.register(name, cls)
