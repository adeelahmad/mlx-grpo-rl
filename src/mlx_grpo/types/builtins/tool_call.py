"""Tool calling task type — think → <tool_call>JSON</tool_call>.

For training models to produce structured tool/function calls
with validated JSON arguments.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType


class ToolCallType(TaskType, name="tool_call"):
    """Tool/function calling type."""

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.6,
            "thinking_budget": 512,
            "call_max_tokens": 256,
            "recovery_ratio": 0.8,
            "require_thinking": True,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=(
                "You are a helpful assistant with tool calling capabilities. "
                "Analyze the request, then call the appropriate tool with correct arguments."
            ),
            enable_thinking=True,
        )

    def callbacks(self) -> dict[str, Callable[..., Any]]:
        return {
            "validate_tool_call": _validate_tool_call,
            "extract_tool_name": _extract_tool_name,
            "extract_tool_args": _extract_tool_args,
        }

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config

        # Optional thinking phase — reserve tokens for tool call
        if cfg.get("require_thinking", True):
            thinking_budget = cfg.get("thinking_budget", 512)
            call_reserve = cfg.get("call_max_tokens", 256)
            if ctx.max_completion_tokens > 0:
                thinking_budget = min(
                    thinking_budget,
                    max(1, ctx.max_completion_tokens - call_reserve),
                )
            thinking = ctx.generate(
                max_tokens=thinking_budget,
                stop=["</think>"],
                temperature=cfg.get("temperature", 0.6),
                tag="thinking",
            )
            ctx.capture("thinking", thinking)

            if thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
                if ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.8)):
                    ctx.inject("</think>\n", masked=True)
                else:
                    return

        # Tool call generation
        ctx.inject("<tool_call>\n", masked=True, tag="tool_prefix")
        call = ctx.generate(
            max_tokens=cfg.get("call_max_tokens", 256),
            stop=["</tool_call>"],
            temperature=0.3,  # Low temp for structured output
            tag="tool_call",
        )
        ctx.capture("tool_call", call)

        # Ensure closing tag
        if "</tool_call>" not in call.text:
            ctx.inject("</tool_call>", masked=True)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        completion = result.completion_text
        call_var = result.variables.get("tool_call")
        call_text = call_var.text if call_var else completion

        score = 0.0

        # Format reward: valid JSON in tool_call tags
        is_valid_format = _validate_tool_call(call_text)
        if is_valid_format:
            score += 0.3

        # Try to parse the call
        try:
            parsed = json.loads(call_text.strip())
            score += 0.2  # Valid JSON
        except (json.JSONDecodeError, TypeError):
            return score

        if answer is None:
            return score

        # Compare with ground truth
        try:
            gt = json.loads(answer) if isinstance(answer, str) else answer
        except (json.JSONDecodeError, TypeError):
            return score

        # Tool name match
        pred_name = parsed.get("name") or parsed.get("function", {}).get("name", "")
        gt_name = gt.get("name") or gt.get("function", {}).get("name", "")
        if pred_name == gt_name:
            score += 0.2

        # Arguments match
        pred_args = parsed.get("arguments") or parsed.get("parameters", {})
        gt_args = gt.get("arguments") or gt.get("parameters", {})
        if isinstance(pred_args, str):
            try:
                pred_args = json.loads(pred_args)
            except json.JSONDecodeError:
                pred_args = {}
        if isinstance(gt_args, str):
            try:
                gt_args = json.loads(gt_args)
            except json.JSONDecodeError:
                gt_args = {}

        if pred_args == gt_args:
            score += 0.3
        elif isinstance(pred_args, dict) and isinstance(gt_args, dict):
            # Partial credit for matching keys
            common = set(pred_args.keys()) & set(gt_args.keys())
            if common:
                matching = sum(1 for k in common if pred_args[k] == gt_args[k])
                score += 0.3 * matching / max(len(gt_args), 1)

        return min(1.0, score)


def _validate_tool_call(text: str) -> bool:
    """Check if text contains valid JSON tool call."""
    text = str(text).strip()
    try:
        parsed = json.loads(text)
        return isinstance(parsed, dict) and ("name" in parsed or "function" in parsed)
    except (json.JSONDecodeError, TypeError):
        return False


def _extract_tool_name(text: str) -> str | None:
    """Extract tool name from JSON call."""
    try:
        parsed = json.loads(text.strip())
        return parsed.get("name") or parsed.get("function", {}).get("name")
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_tool_args(text: str) -> dict[str, Any] | None:
    """Extract tool arguments from JSON call."""
    try:
        parsed = json.loads(text.strip())
        args = parsed.get("arguments") or parsed.get("parameters", {})
        if isinstance(args, str):
            args = json.loads(args)
        return args if isinstance(args, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None
