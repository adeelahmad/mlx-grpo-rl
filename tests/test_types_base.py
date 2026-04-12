"""Tests for types/base.py — ChatParams, TaskType registry."""

from __future__ import annotations

import pytest

from mlx_grpo.types.base import ChatParams, TaskType

# ── ChatParams ──────────────────────────────────────────────────────────────


class TestChatParamsToKwargs:
    def test_defaults(self):
        cp = ChatParams()
        kw = cp.to_kwargs()
        assert kw["add_generation_prompt"] is True
        assert kw.get("enable_thinking") is True

    def test_enable_thinking_false(self):
        cp = ChatParams(enable_thinking=False)
        kw = cp.to_kwargs()
        assert "enable_thinking" not in kw

    def test_thinking_budget_included(self):
        cp = ChatParams(thinking_budget=512)
        kw = cp.to_kwargs()
        assert kw["thinking_budget"] == 512

    def test_thinking_budget_none_not_included(self):
        cp = ChatParams(thinking_budget=None)
        kw = cp.to_kwargs()
        assert "thinking_budget" not in kw

    def test_tools_included(self):
        tools = [{"name": "calc"}]
        cp = ChatParams(tools=tools)
        kw = cp.to_kwargs()
        assert kw["tools"] == tools

    def test_tools_none_not_included(self):
        cp = ChatParams(tools=None)
        kw = cp.to_kwargs()
        assert "tools" not in kw

    def test_extra_merged(self):
        cp = ChatParams(extra={"custom_param": 42})
        kw = cp.to_kwargs()
        assert kw["custom_param"] == 42

    def test_add_generation_prompt_false(self):
        cp = ChatParams(add_generation_prompt=False)
        kw = cp.to_kwargs()
        assert kw["add_generation_prompt"] is False

    def test_all_fields(self):
        cp = ChatParams(
            add_generation_prompt=True,
            enable_thinking=True,
            thinking_budget=1024,
            tools=[{"name": "search"}],
            extra={"top_p": 0.9},
        )
        kw = cp.to_kwargs()
        assert kw["add_generation_prompt"] is True
        assert kw["enable_thinking"] is True
        assert kw["thinking_budget"] == 1024
        assert kw["tools"] == [{"name": "search"}]
        assert kw["top_p"] == 0.9


# ── TaskType registry ────────────────────────────────────────────────────────


class TestTaskTypeRegistry:
    def test_get_registered_type(self):
        """Built-in types are registered."""
        import mlx_grpo.types.builtins  # noqa: F401 — triggers registration

        t = TaskType.get("math")
        assert t is not None

    def test_alias_resolution(self):
        """Alias 'exam' resolves to 'mcq'."""
        import mlx_grpo.types.builtins  # noqa: F401

        t = TaskType.get("exam")
        t2 = TaskType.get("mcq")
        assert isinstance(t2, type(t))

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown type"):
            TaskType.get("nonexistent_type_xyz")

    def test_list_types(self):
        import mlx_grpo.types.builtins  # noqa: F401

        types = TaskType.list_types()
        assert "math" in types
        assert "mcq" in types

    def test_register_custom_type(self):
        """TaskType.register() adds a new type manually."""

        class _FakeType(TaskType):
            def rollout(self, ctx):
                pass

            def reward(self, result, answer=None):
                return 0.0

        TaskType.register("_test_fake_register_xyz", _FakeType)
        t = TaskType.get("_test_fake_register_xyz")
        assert isinstance(t, _FakeType)
        # Clean up
        del TaskType._registry["_test_fake_register_xyz"]

    def test_add_alias(self):
        """TaskType.add_alias() adds a new alias."""
        import mlx_grpo.types.builtins  # noqa: F401

        TaskType.add_alias("_test_alias_xyz", "math")
        t = TaskType.get("_test_alias_xyz")
        assert t is not None
        # Clean up
        del TaskType._aliases["_test_alias_xyz"]

    def test_chat_params_default(self):
        """chat_params() returns ChatParams instance."""
        import mlx_grpo.types.builtins  # noqa: F401

        t = TaskType.get("math")
        cp = t.chat_params()
        assert isinstance(cp, ChatParams)

    def test_chat_messages_no_system(self):
        """chat_messages builds user message without system prompt."""
        import mlx_grpo.types.builtins  # noqa: F401

        # Use general_qna which has a system prompt
        tg = TaskType.get("general_qna")
        msgs = tg.chat_messages("What is 2+2?")
        assert len(msgs) >= 1
        # Last message is user
        assert msgs[-1]["role"] == "user"
        assert "2+2" in msgs[-1]["content"]

    def test_chat_messages_with_system(self):
        """When system_prompt is None, only user message is returned."""
        import mlx_grpo.types.builtins  # noqa: F401

        t = TaskType.get("general_qna")
        msgs = t.chat_messages("hello")
        # general_qna has no system prompt — just user message
        assert msgs[-1]["role"] == "user"

    def test_chat_messages_no_system_prompt(self):
        """chat_messages without system_prompt returns just user message."""
        import mlx_grpo.types.builtins  # noqa: F401

        class _NoSystemType(TaskType, name="_no_sys_xyz"):
            def rollout(self, ctx):
                pass

            def reward(self, r, a=None):
                return 0.0

            def chat_params(self):
                return ChatParams(system_prompt=None)

        t = _NoSystemType()
        msgs = t.chat_messages("hello world")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Clean up
        del TaskType._registry["_no_sys_xyz"]
