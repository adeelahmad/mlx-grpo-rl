"""Python verified — think → code → run → expected output → explanation.

Execution-based self-verification: model writes code, the bash command
to run it (saved as solution.py), predicts output, then explains.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_grpo.sandbox import Sandbox

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType

_RE_THINK = re.compile(r"(<think>[\s\S]*?</think>)")
_RE_CODE = re.compile(r"```python\n([\s\S]*?)```")
_RE_FUNC = re.compile(r"def\s+(\w+)\s*\(")

_sandbox: Sandbox | None = None


def _get_sandbox(timeout: int = 10) -> Sandbox:
    global _sandbox  # noqa: PLW0603
    if _sandbox is None:
        from mlx_grpo.sandbox import Sandbox

        _sandbox = Sandbox(timeout=timeout)
    return _sandbox


class PythonVerifiedType(TaskType, name="python_verified"):
    """Python code with execution-based self-verification."""

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.7,
            "thinking_budget": 192,
            "code_max_tokens": 384,
            "run_max_tokens": 64,
            "output_max_tokens": 64,
            "explanation_max_tokens": 192,
            "recovery_ratio": 0.8,
            "exec_timeout": 10,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=(
                "You are a Python expert. Think briefly, write the code, "
                "provide the exact bash command to run it (the code is saved "
                "as solution.py), predict the expected output, then explain."
            ),
            enable_thinking=True,
        )

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config
        frac = ctx.batch.group_index / max(ctx.batch.group_size - 1, 1)
        tier = 0 if frac < 0.34 else (1 if frac < 0.67 else 2)
        ctx.set_meta("scaffold_tier", tier)

        # Phase 1: Thinking
        if tier <= 1 and ctx.answer:
            m = _RE_THINK.search(ctx.answer)
            if m:
                ctx.inject(m.group(1), masked=True, tag="scaffold")
        else:
            self._gen_thinking(ctx, cfg)

        # Phase 2: Code
        if tier == 0 and ctx.answer:
            half = _extract_half_code(ctx.answer)
            ctx.inject("\n\n```python\n" + (half or ""), masked=True, tag="scaffold")
        else:
            ctx.inject("\n\n```python\n", masked=True, tag="code_prefix")
        code = ctx.generate(
            max_tokens=cfg.get("code_max_tokens", 384),
            stop=["```"],
            temperature=0.4,
            tag="code",
        )
        ctx.capture("code", code)
        if "```" not in code.text:
            ctx.inject("\n```", masked=True)

        # Phase 3: Run command
        ctx.inject("\n\n## Run\n```bash\n", masked=True, tag="run_prefix")
        run = ctx.generate(
            max_tokens=cfg.get("run_max_tokens", 64),
            stop=["```", "\n\n"],
            temperature=0.2,
            tag="run",
        )
        ctx.capture("run", run)
        if "```" not in run.text:
            ctx.inject("\n```", masked=True)

        # Phase 4: Expected output
        ctx.inject("\n\n## Expected Output\n```\n", masked=True, tag="output_prefix")
        exp = ctx.generate(
            max_tokens=cfg.get("output_max_tokens", 64),
            stop=["```"],
            temperature=0.2,
            tag="expected_output",
        )
        ctx.capture("expected_output", exp)
        if "```" not in exp.text:
            ctx.inject("\n```", masked=True)

        # Phase 5: Explanation
        ctx.inject("\n\n## Explanation\n", masked=True, tag="explanation_prefix")
        ctx.capture(
            "explanation",
            ctx.generate(
                max_tokens=cfg.get("explanation_max_tokens", 192),
                temperature=0.5,
                tag="explanation",
            ),
        )

    def _gen_thinking(self, ctx: RolloutContext, cfg: dict) -> None:
        budget = cfg.get("thinking_budget", 192)
        if ctx.max_completion_tokens > 0:
            budget = min(budget, max(1, ctx.max_completion_tokens - 700))
        t = ctx.generate(
            max_tokens=budget,
            stop=["</think>"],
            temperature=cfg.get("temperature", 0.7),
            tag="thinking",
        )
        ctx.capture("thinking", t)
        if (
            t.stop_reason == "max_tokens"
            and "</think>" not in t.text
            and ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.8))
        ):
            ctx.inject("</think>", masked=True)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        code = _vt(result, "code")
        if not code:
            return 0.0
        run_cmd, expected = _vt(result, "run"), _vt(result, "expected_output")
        expl = _vt(result, "explanation")
        s = 0.0
        # Syntax + structure (0.20)
        if _check_syntax(code):
            s += 0.10
        if _RE_FUNC.search(code):
            s += 0.05
        if "return " in code:
            s += 0.05
        # Execution (0.55) — core signal
        if run_cmd and expected:
            rc, stdout, _ = _get_sandbox(self.config()["exec_timeout"]).execute(
                code,
                run_cmd,
                timeout=self.config()["exec_timeout"],
            )
            if rc == 0:
                s += 0.25
                if _output_matches(stdout, expected):
                    s += 0.30
            result.metadata["exec_rc"] = rc
            result.metadata["exec_stdout"] = stdout[:200]
        # Explanation (0.15)
        if expl and len(expl) > 20:
            s += 0.10
            names = set(_RE_FUNC.findall(code))
            if names and any(n in expl for n in names):
                s += 0.05
        # Ground truth (0.10)
        if answer:
            gt = (_RE_CODE.search(answer) or type("", (), {"group": lambda *_: answer})).group(1)
            s += 0.10 * _kw_overlap(code, gt.strip())
        return min(1.0, s)


def _vt(result: RolloutResult, name: str) -> str:
    v = result.variables.get(name)
    return v.text.strip() if v else ""


def _extract_half_code(text: str) -> str | None:
    m = _RE_CODE.search(text)
    if not m:
        return None
    lines = m.group(1).splitlines(keepends=True)
    return "".join(lines[: max(1, len(lines) // 2)])


def _check_syntax(code: str) -> bool:
    try:
        compile(code.strip(), "<reward>", "exec")
        return True
    except SyntaxError:
        return False


def _output_matches(actual: str, expected: str) -> bool:
    a, e = actual.strip(), expected.strip()
    if a == e:
        return True
    if [ln.rstrip() for ln in a.splitlines()] == [ln.rstrip() for ln in e.splitlines()]:
        return True
    try:
        return abs(float(a) - float(e)) < 1e-6
    except (ValueError, OverflowError):
        return False


def _kw_overlap(pred: str, gt: str) -> float:
    skip = {
        "def",
        "return",
        "if",
        "else",
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
    }
    pt = {t for t in re.findall(r"\b\w+\b", pred) if t not in skip and len(t) > 1}
    gt_t = {t for t in re.findall(r"\b\w+\b", gt) if t not in skip and len(t) > 1}
    return min(1.0, len(pt & gt_t) / max(len(gt_t), 1) * 1.5) if gt_t else 0.0
