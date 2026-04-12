"""YAML type loader — create TaskType subclasses from config with inline lambdas.

Uses ``type()`` metaclass call for dynamic class creation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..engine.context import RolloutContext
from ..engine.result import RolloutResult
from .base import ChatParams, TaskType

# ── Safe lambda parsing ──

_SAFE_BUILTINS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "sorted": sorted,
    "range": range,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "True": True,
    "False": False,
    "None": None,
}

try:
    import re as _re

    _SAFE_BUILTINS["re"] = _re
except ImportError:
    pass  # re is stdlib — this guard is defensive only


def parse_lambda(expr: str) -> Callable[..., Any]:
    """Parse an inline lambda from YAML. No statements, no imports."""
    expr = expr.strip()
    if not expr.startswith("lambda"):
        raise ValueError(f"Expected 'lambda' expression, got: {expr!r}")
    if any(kw in expr for kw in ("import", "__", "exec", "eval", "open", "compile")):
        raise ValueError(f"Forbidden keyword in lambda: {expr!r}")
    return eval(expr, {"__builtins__": _SAFE_BUILTINS})  # noqa: S307


# ── Step compiler: YAML rollout steps → Python function ──


def _compile_rollout(steps: list[dict[str, Any]]) -> Callable[[RolloutContext], None]:
    """Compile a YAML rollout step list into a rollout function."""
    compiled = [_compile_step(step) for step in steps]

    def rollout_fn(ctx: RolloutContext) -> None:
        variables: dict[str, Any] = {}
        for step in compiled:
            step(ctx, variables)

    return rollout_fn


def _compile_step(step: dict[str, Any]) -> Callable:
    """Compile a single YAML step into a callable."""
    if "generate" in step:
        params = dict(step["generate"]) if isinstance(step["generate"], dict) else {}
        tag = params.pop("tag", "")
        return lambda ctx, vs, p=params, t=tag: _do_generate(ctx, vs, p, t)

    if "inject" in step:
        p = step["inject"] if isinstance(step["inject"], dict) else {"text": step["inject"]}
        return lambda ctx, vs, p=p: ctx.inject(
            _resolve_text(p.get("text", ""), vs, ctx),
            masked=p.get("masked", True),
            tag=p.get("tag", ""),
        )

    if "if" in step:
        condition = step["if"]
        then_steps = []
        if "inject" in step:
            then_steps.append(_compile_step({"inject": step["inject"]}))
        if "generate" in step:
            then_steps.append(_compile_step({"generate": step["generate"]}))
        if "then" in step:
            then_steps = [_compile_step(s) for s in step["then"]]
        else_steps = [_compile_step(s) for s in step.get("else", [])]

        return lambda ctx, vs, c=condition, t=then_steps, e=else_steps: (
            [s(ctx, vs) for s in t] if _eval_condition(c, vs, ctx) else [s(ctx, vs) for s in e]
        )

    if "for" in step:
        var_name, iterable_expr = step["for"].split(" in ", 1)
        body_steps = [_compile_step(s) for s in step.get("body", [])]
        vn = var_name.strip()
        ie = iterable_expr.strip()
        return lambda ctx, vs, v=vn, it=ie, b=body_steps: (_do_for_loop(ctx, vs, v, it, b))

    raise ValueError(f"Unknown step type: {step}")


def _do_generate(ctx: RolloutContext, variables: dict, params: dict, tag: str) -> None:
    result = ctx.generate(tag=tag, **params)
    if tag:
        variables[tag] = result
        ctx.capture(tag, result)


def _do_for_loop(ctx, variables, var_name, iterable_expr, body_steps):
    iterable = _eval_expr(iterable_expr, variables, ctx)
    for item in iterable:
        variables[var_name] = item
        for step in body_steps:
            step(ctx, variables)


def _resolve_text(text: str, variables: dict, ctx: RolloutContext | None = None) -> str:
    """Resolve inject text — literal by default, evaluated if it looks like code."""
    expr_markers = ("callbacks.", "ctx.", "batch.", "str(", 'f"', "variables[")
    if any(m in text for m in expr_markers):
        return str(_eval_expr(text, variables, ctx))
    return text


def _eval_expr(expr: str, variables: dict, ctx: RolloutContext | None = None) -> Any:
    """Evaluate a simple expression in the variable context."""
    ns = {**_SAFE_BUILTINS, **variables}
    if ctx:
        ns["ctx"] = ctx
        ns["batch"] = ctx.batch
        ns["callbacks"] = ctx.callbacks
    return eval(expr, {"__builtins__": _SAFE_BUILTINS}, ns)  # noqa: S307


def _eval_condition(condition: str, variables: dict, ctx: RolloutContext) -> bool:
    return bool(_eval_expr(condition, variables, ctx))


# ── YAML TaskType factory ──


def create_yaml_type(name: str, cfg: dict[str, Any]) -> type[TaskType]:
    """Dynamically create a TaskType subclass from YAML configuration.

    Uses ``type()`` metaclass call — the purest form of metaprogramming.
    """
    rollout_steps = cfg.get("rollout", [])
    rollout_fn = _compile_rollout(rollout_steps) if rollout_steps else _noop_rollout
    reward_name = cfg.get("reward", "accuracy")
    raw_callbacks = {k: parse_lambda(v) for k, v in cfg.get("callbacks", {}).items()}
    skip = ("rollout", "reward", "callbacks", "system_prompt")
    type_config = {k: v for k, v in cfg.items() if k not in skip}

    chat = ChatParams(
        system_prompt=cfg.get("system_prompt"),
        enable_thinking=cfg.get("enable_thinking", True),
        thinking_budget=cfg.get("thinking_budget"),
    )

    # Build class dict
    cls_dict = {
        "_rollout_fn": staticmethod(rollout_fn),
        "_reward_name": reward_name,
        "_type_config": type_config,
        "_chat_params": chat,
        "_callbacks_dict": raw_callbacks,
        "rollout": lambda self, ctx: self._rollout_fn(ctx),
        "reward": lambda self, result, answer=None: _yaml_reward(self._reward_name, result, answer),
        "config": lambda self: self._type_config,
        "chat_params": lambda self: self._chat_params,
        "callbacks": lambda self: self._callbacks_dict,
    }

    # type() call — dynamic class creation
    cls = type(f"{name.title()}YAMLType", (TaskType,), cls_dict)
    TaskType.register(name, cls)
    return cls


def _noop_rollout(ctx: RolloutContext) -> None:
    ctx.generate(max_tokens=ctx.type_config.get("max_tokens", 512))


def _yaml_reward(reward_name: str, result: RolloutResult, answer: str | None) -> float:
    from ..rewards.base import BaseReward

    reward_cls = BaseReward.get(reward_name)
    return reward_cls.compute(result, answer)
