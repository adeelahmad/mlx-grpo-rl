"""Template compiler — converts Jinja2 templates to executable rollout functions.

The compiler bridges the declarative template world with the imperative
RolloutContext API. It works in two stages:

1. **Parse** — Jinja2 renders the template with custom tags into an
   instruction list (a list of TemplateOp dataclasses).
2. **Execute** — The executor walks the op list and calls ctx.generate(),
   ctx.inject(), etc.

Templates are cached after first compilation. The Jinja2 rendering
happens at parse time (once); execution is a fast linear walk.

    compiler = TemplateCompiler()
    rollout_fn = compiler.compile("math.j2")
    rollout_fn(ctx)  # calls ctx.generate/inject/etc.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .context import RolloutContext

# ── Instruction Set ──


class OpCode(StrEnum):
    GENERATE = "generate"
    GENERATE_UNTIL = "generate_until"
    GENERATE_IF = "generate_if"
    INJECT = "inject"
    SCAFFOLD = "scaffold"
    CAPTURE = "capture"
    SET_META = "set_meta"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    ABORT = "abort"


@dataclass(slots=True)
class TemplateOp:
    """Single instruction in a compiled template."""

    op: OpCode
    params: dict[str, Any] = field(default_factory=dict)
    children: list[TemplateOp] = field(default_factory=list)
    else_children: list[TemplateOp] = field(default_factory=list)


# ── Template DSL Parser ──

# Directive syntax: @generate, @inject, @scaffold, etc.
# These are embedded in rendered template output as pseudo-instructions.
_DIRECTIVE_RE = re.compile(
    r"@(generate|generate_until|generate_if|inject|scaffold|capture|set_meta|abort|if|elif|else|endif|for|endfor)"
    r"(?:\s+(.*))?$",
    re.MULTILINE,
)


def parse_template_string(template_text: str) -> list[TemplateOp]:
    """Parse a rendered template string into a list of TemplateOps.

    The template uses a simple directive language:

        @generate max_tokens=512 stop=</think> tag=thinking as=thinking
        @inject text="</think>\\n" masked=true
        @scaffold ratio=0.5 mode=prefix
        @if condition=thinking.stop_reason=="max_tokens"
        @endif
        @abort

    This is intentionally simpler than full Jinja2 — the Jinja2 rendering
    has already happened, so we're just parsing the output directives.
    """
    ops: list[TemplateOp] = []
    stack: list[tuple[str, list[TemplateOp], list[TemplateOp] | None]] = []

    for line in template_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = _DIRECTIVE_RE.match(line)
        if not m:
            # Plain text becomes inject(masked=false) — template literal
            if line:
                target = stack[-1][1] if stack else ops
                target.append(
                    TemplateOp(
                        op=OpCode.INJECT,
                        params={"text": line, "masked": False},
                    )
                )
            continue

        directive, args_str = m.group(1), m.group(2) or ""
        params = _parse_args(args_str)

        if directive == "generate":
            op = TemplateOp(op=OpCode.GENERATE, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "generate_until":
            op = TemplateOp(op=OpCode.GENERATE_UNTIL, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "generate_if":
            op = TemplateOp(op=OpCode.GENERATE_IF, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "inject":
            op = TemplateOp(op=OpCode.INJECT, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "scaffold":
            op = TemplateOp(op=OpCode.SCAFFOLD, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "capture":
            op = TemplateOp(op=OpCode.CAPTURE, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "set_meta":
            op = TemplateOp(op=OpCode.SET_META, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "abort":
            op = TemplateOp(op=OpCode.ABORT, params=params)
            target = stack[-1][1] if stack else ops
            target.append(op)

        elif directive == "if":
            cond_op = TemplateOp(op=OpCode.CONDITIONAL, params=params)
            target = stack[-1][1] if stack else ops
            target.append(cond_op)
            stack.append(("if", cond_op.children, cond_op.else_children))

        elif directive == "elif":
            if stack and stack[-1][0] == "if":
                # Switch to else branch with nested conditional
                stack[-1] = ("elif", stack[-1][2], None)  # type: ignore

        elif directive == "else":
            if stack and stack[-1][0] in ("if", "elif"):
                stack[-1] = ("else", stack[-1][2], None)  # type: ignore

        elif directive == "endif":
            if stack:
                stack.pop()

        elif directive == "for":
            loop_op = TemplateOp(op=OpCode.LOOP, params=params)
            target = stack[-1][1] if stack else ops
            target.append(loop_op)
            stack.append(("for", loop_op.children, None))

        elif directive == "endfor":
            if stack:
                stack.pop()

    return ops


def _parse_args(args_str: str) -> dict[str, Any]:
    """Parse key=value argument string into a dict.

    Handles:
        max_tokens=512
        stop=</think>
        masked=true
        text="hello world"
        tag=thinking
        as=thinking
        condition=x>0
    """
    if not args_str:
        return {}

    result: dict[str, Any] = {}
    # Match key=value pairs, where value can be quoted or unquoted
    for m in re.finditer(r'(\w+)=(?:"([^"]*)"|([\S]+))', args_str):
        key = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)

        # Type coercion
        if value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        elif re.match(r"^-?\d+$", value):
            result[key] = int(value)
        elif re.match(r"^-?\d+\.\d+$", value):
            result[key] = float(value)
        else:
            result[key] = value

    return result


# ── Executor ──


class TemplateExecutor:
    """Execute compiled template ops against a RolloutContext.

    This is the runtime engine. It walks the op list and translates
    each op into RolloutContext method calls.
    """

    def execute(self, ops: list[TemplateOp], ctx: RolloutContext) -> None:
        """Execute a list of ops in order."""
        for op in ops:
            if not self._execute_one(op, ctx):
                break  # abort

    def _execute_one(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        """Execute a single op. Returns False to abort."""
        if op.op == OpCode.GENERATE:
            return self._do_generate(op, ctx)
        elif op.op == OpCode.GENERATE_UNTIL:
            return self._do_generate_until(op, ctx)
        elif op.op == OpCode.GENERATE_IF:
            return self._do_generate_if(op, ctx)
        elif op.op == OpCode.INJECT:
            return self._do_inject(op, ctx)
        elif op.op == OpCode.SCAFFOLD:
            return self._do_scaffold(op, ctx)
        elif op.op == OpCode.CAPTURE:
            return self._do_capture(op, ctx)
        elif op.op == OpCode.SET_META:
            return self._do_set_meta(op, ctx)
        elif op.op == OpCode.CONDITIONAL:
            return self._do_conditional(op, ctx)
        elif op.op == OpCode.LOOP:
            return self._do_loop(op, ctx)
        elif op.op == OpCode.ABORT:
            return False
        return True

    def _do_generate(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        p = op.params
        result = ctx.generate(
            max_tokens=p.get("max_tokens", 512),
            stop=(
                p.get("stop", "").split(",")
                if isinstance(p.get("stop"), str)
                else p.get("stop", [])
            ),
            temperature=p.get("temperature", 0.7),
            top_p=p.get("top_p", 0.95),
            repetition_penalty=p.get("repetition_penalty", 1.0),
            repetition_context_size=p.get("repetition_context_size", 20),
            tag=p.get("tag", ""),
        )
        if "as" in p:
            ctx.capture(p["as"], result)
        return True

    def _do_generate_until(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        p = op.params
        result = ctx.generate_until(
            endswith=p.get("endswith", ""),
            max_tokens=p.get("max_tokens", 256),
            temperature=p.get("temperature", 0.7),
            tag=p.get("tag", ""),
        )
        if "as" in p:
            ctx.capture(p["as"], result)
        return True

    def _do_generate_if(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        p = op.params
        validator_name = p.get("validator", "")
        validator = getattr(ctx.callbacks, validator_name, None) if validator_name else None
        if validator is None:
            validator = lambda x: True  # noqa: E731

        result = ctx.generate_if(
            validator=validator,
            max_retries=p.get("max_retries", 3),
            max_tokens=p.get("max_tokens", 128),
            temperature=p.get("temperature", 0.7),
            tag=p.get("tag", ""),
        )
        if "as" in p:
            ctx.capture(p["as"], result)
        return True

    def _do_inject(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        p = op.params
        text = p.get("text", "")
        # Unescape common sequences
        text = text.replace("\\n", "\n").replace("\\t", "\t")
        ctx.inject(text, masked=p.get("masked", True), tag=p.get("tag", ""))
        return True

    def _do_scaffold(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        p = op.params
        if not ctx.answer:
            return True

        sc = ctx.batch.scaffold
        mode = p.get("mode", sc.mode)
        tag = p.get("tag", "scaffold")
        gi = ctx.batch.group_index
        max_tok = p.get("max_tokens")  # template-level override

        tokens = ctx.encode(ctx.answer)
        if max_tok is not None:
            budget = max(0, int(max_tok * sc.effective_ratio(ctx.batch.progress, gi)))
        else:
            ratio = p.get("ratio", 0.0)
            if isinstance(ratio, str):
                ratio = self._eval_ratio(ratio, ctx)
            if sc.max_scaffolded_tokens is not None:
                budget = max(0, int(sc.max_scaffolded_tokens * ratio))
            else:
                budget = max(0, int(len(tokens) * ratio))

        if budget > 0:
            sliced = ctx.callbacks.slice_answer_tokens(tokens, budget, mode)
            if sliced:
                ctx.inject(ctx.decode(sliced), masked=True, tag=tag)
        return True

    def _do_capture(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        # Capture is handled inline by generate ops via 'as' param
        return True

    def _do_set_meta(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        for k, v in op.params.items():
            ctx.set_meta(k, v)
        return True

    def _do_conditional(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        condition = op.params.get("condition", "true")
        if self._eval_condition(condition, ctx):
            self.execute(op.children, ctx)
        elif op.else_children:
            self.execute(op.else_children, ctx)
        return True

    def _do_loop(self, op: TemplateOp, ctx: RolloutContext) -> bool:
        count = op.params.get("count", 1)
        for i in range(int(count)):
            ctx.set_meta("_loop_index", i)
            self.execute(op.children, ctx)
        return True

    # ── Expression evaluation ──

    def _eval_condition(self, expr: str, ctx: RolloutContext) -> bool:
        """Evaluate a simple condition expression."""
        expr = expr.strip()
        if expr.lower() in ("true", "1", "yes"):
            return True
        if expr.lower() in ("false", "0", "no"):
            return False

        # Variable access: thinking.stop_reason=="max_tokens"
        ns = self._build_namespace(ctx)
        try:
            return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception:
            return False

    def _eval_ratio(self, expr: str, ctx: RolloutContext) -> float:
        """Evaluate a ratio expression."""
        ns = self._build_namespace(ctx)
        try:
            return float(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception:
            return 0.0

    def _build_namespace(self, ctx: RolloutContext) -> dict[str, Any]:
        """Build namespace for expression evaluation."""
        ns: dict[str, Any] = {
            "batch": ctx.batch,
            "progress": ctx.batch.progress,
            "answer": ctx.answer or "",
            "prompt": ctx.prompt,
            "callbacks": ctx.callbacks,
        }
        # Add captured variables
        for name, var in ctx._variables.items():
            ns[name] = var
        return ns


# ── Compiler ──


class TemplateCompiler:
    """Compiles template strings into reusable rollout functions.

    Templates are parsed once and the resulting op list is cached.
    The compiled function is a standard ``(ctx: RolloutContext) -> None``.

        compiler = TemplateCompiler()
        fn = compiler.compile(template_string)
        fn(ctx)
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[TemplateOp]] = {}
        self._executor = TemplateExecutor()

    def compile(self, template: str) -> Callable[[RolloutContext], None]:
        """Compile a template string into a rollout function."""
        cache_key = template
        if cache_key not in self._cache:
            self._cache[cache_key] = parse_template_string(template)

        ops = self._cache[cache_key]

        def rollout_fn(ctx: RolloutContext) -> None:
            self._executor.execute(ops, ctx)

        return rollout_fn

    def compile_file(self, path: str) -> Callable[[RolloutContext], None]:
        """Compile a template file into a rollout function."""
        with open(path) as f:
            return self.compile(f.read())

    def parse_only(self, template: str) -> list[TemplateOp]:
        """Parse without compiling (useful for debugging)."""
        return parse_template_string(template)

    def clear_cache(self) -> None:
        self._cache.clear()
