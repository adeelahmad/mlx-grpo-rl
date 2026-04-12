"""Task type system — metaprogramming-driven type registry.

Every task type (math, mcq, tool_call, etc.) is a subclass of ``TaskType``.
Registration is automatic via ``__init_subclass__``::

    class MCQType(TaskType, name="mcq"):
        def rollout(self, ctx): ...
        def reward(self, result): ...

    # Now available:
    TaskType.get("mcq")
    TaskType.list_types()

Types can also be created from YAML via ``TaskType.from_yaml()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..engine.context import RolloutContext
from ..engine.result import RolloutResult


@dataclass(slots=True)
class ChatParams:
    """Chat template parameters passed to tokenizer.apply_chat_template()."""

    add_generation_prompt: bool = True
    enable_thinking: bool = True
    thinking_budget: int | None = None
    tools: list[dict[str, Any]] | None = None
    system_prompt: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "add_generation_prompt": self.add_generation_prompt,
        }
        if self.enable_thinking:
            kw["enable_thinking"] = True
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.tools is not None:
            kw["tools"] = self.tools
        kw.update(self.extra)
        return kw


class TaskType(ABC):
    """Base class for all task types. Subclass to register automatically.

    ``__init_subclass__`` handles registration — no decorators or manual
    registration needed::

        class MathType(TaskType, name="math"):
            ...

    The ``name=`` kwarg is optional; defaults to class name lowercased
    with 'Type' suffix stripped.
    """

    _registry: ClassVar[dict[str, type[TaskType]]] = {}
    _aliases: ClassVar[dict[str, str]] = {
        # Maps from original project's type system (exam variants → mcq)
        "exam": "mcq",
        "exam_math": "math",
        "exam_aime": "math",
        "exam_olympiad": "math",
        "general": "general_qna",
        "code": "python_code",
        "code_verified": "python_verified",
        "tattslotto": "lotto",
        "lottery": "lotto",
        "tatts": "lotto",
    }

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls._registry[name] = cls
        elif not getattr(cls, "__abstractmethods__", None):
            # Auto-name from class: MCQType → mcq, GeneralQnAType → general_qna
            auto = cls.__name__.removesuffix("Type")
            auto = "".join(f"_{c.lower()}" if c.isupper() else c for c in auto).lstrip("_")
            cls._registry[auto] = cls

    # ── Registry class methods ──

    @classmethod
    def get(cls, name: str) -> TaskType:
        """Instantiate a registered type by name. Supports aliases."""
        resolved = cls._aliases.get(name, name)
        if resolved not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown type {name!r}. Available: {available}")
        return cls._registry[resolved]()

    @classmethod
    def list_types(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def register(cls, name: str, type_cls: type[TaskType]) -> None:
        """Manually register a type class."""
        cls._registry[name] = type_cls

    @classmethod
    def add_alias(cls, alias: str, target: str) -> None:
        """Register a type alias (e.g., 'exam' → 'mcq')."""
        cls._aliases[alias] = target

    # ── Abstract interface ──

    @abstractmethod
    def rollout(self, ctx: RolloutContext) -> None:
        """Execute the rollout: generate, inject, build completion."""
        ...

    @abstractmethod
    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        """Compute reward for a completed rollout."""
        ...

    # ── Optional overrides ──

    def chat_params(self) -> ChatParams:
        """Chat template parameters."""
        return ChatParams()

    def callbacks(self) -> dict[str, Callable[..., Any]]:
        """Extra callbacks to register for this type."""
        return {}

    def config(self) -> dict[str, Any]:
        """Type-specific configuration (temperature, max_tokens, etc.).

        Values are accessible in the rollout as ``ctx.type_config``.
        """
        return {}

    def chat_messages(self, prompt: str) -> list[dict[str, str]]:
        """Build chat messages from raw prompt."""
        messages = []
        cp = self.chat_params()
        if cp.system_prompt:
            messages.append({"role": "system", "content": cp.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
