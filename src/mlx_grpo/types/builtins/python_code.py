"""Python code generation task type — think → ```python...```

For training models to produce valid, executable Python code
with optional test case validation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType


class PythonCodeType(TaskType, name="python_code"):
    """Python code generation type."""

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.7,
            "thinking_budget": 768,
            "code_max_tokens": 512,
            "recovery_ratio": 0.7,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=(
                "You are a Python programming expert. Think through the problem, "
                "then provide your solution as a complete Python function."
            ),
            enable_thinking=True,
        )

    def callbacks(self) -> dict[str, Callable[..., Any]]:
        return {
            "extract_python": _extract_python_code,
            "check_syntax": _check_syntax,
        }

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config

        # Phase 1: Thinking — reserve tokens for code phase
        thinking_budget = cfg.get("thinking_budget", 768)
        code_reserve = cfg.get("code_max_tokens", 512)
        if ctx.max_completion_tokens > 0:
            thinking_budget = min(
                thinking_budget,
                max(1, ctx.max_completion_tokens - code_reserve),
            )
        thinking = ctx.generate(
            max_tokens=thinking_budget,
            stop=["</think>"],
            temperature=cfg.get("temperature", 0.7),
            tag="thinking",
        )
        ctx.capture("thinking", thinking)

        if thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
            if ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.7)):
                ctx.inject("</think>\n", masked=True)
            else:
                return

        # Phase 2: Code block
        ctx.inject("\n```python\n", masked=True, tag="code_prefix")
        code = ctx.generate(
            max_tokens=cfg.get("code_max_tokens", 512),
            stop=["```"],
            temperature=0.4,
            tag="code",
        )
        ctx.capture("code", code)

        # Ensure closing fence
        if "```" not in code.text:
            ctx.inject("\n```", masked=True)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        code_var = result.variables.get("code")
        code_text = code_var.text if code_var else ""

        if not code_text.strip():
            return 0.0

        score = 0.0

        # Syntactically valid Python
        if _check_syntax(code_text):
            score += 0.3

        # Has function definition
        if re.search(r"def\s+\w+\s*\(", code_text):
            score += 0.1

        # Has return statement
        if "return " in code_text:
            score += 0.1

        if answer is None:
            return min(1.0, score)

        # Compare with ground truth
        gt_code = _extract_python_code(answer) or answer

        # Function name match
        pred_name = _extract_function_name(code_text)
        gt_name = _extract_function_name(gt_code)
        if pred_name and gt_name and pred_name == gt_name:
            score += 0.1

        # Keyword overlap (rough semantic similarity)
        pred_tokens = set(_code_tokens(code_text))
        gt_tokens = set(_code_tokens(gt_code))
        if gt_tokens:
            overlap = len(pred_tokens & gt_tokens) / max(len(gt_tokens), 1)
            score += 0.4 * min(1.0, overlap * 1.5)

        return min(1.0, score)


def _extract_python_code(text: str) -> str | None:
    """Extract Python code from markdown code blocks."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _check_syntax(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        compile(code.strip(), "<reward>", "exec")
        return True
    except SyntaxError:
        return False


def _extract_function_name(code: str) -> str | None:
    """Extract the first function name from code."""
    m = re.search(r"def\s+(\w+)\s*\(", code)
    return m.group(1) if m else None


def _code_tokens(code: str) -> list[str]:
    """Extract meaningful code tokens for overlap comparison."""
    keywords = {
        "def",
        "return",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "in",
        "import",
        "from",
        "class",
        "try",
        "except",
        "with",
        "as",
        "and",
        "or",
        "not",
        "is",
        "None",
        "True",
        "False",
        "self",
        "pass",
        "break",
        "continue",
        "yield",
        "lambda",
    }
    tokens = re.findall(r"\b\w+\b", code)
    return [t for t in tokens if t not in keywords and len(t) > 1]
