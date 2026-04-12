"""Pluggable rollout intervention for the escalation engine.

An intervention handler is called before a target group member's rollout
and receives all completed group members (with rewards) so far. It returns
text to inject into the thinking context before </think>, or None to skip.

Three loading modes (checked in order):
  lambda_expr — inline lambda string (same sandbox as yaml_type.parse_lambda)
  script      — path to a .py file with a handle() function
  callable    — "my.module:function_name" (dotted module path + colon + name)

Handler signature::

    def handle(
        prompt: str,
        answer: str | None,
        ground_truth: str | None,
        group_progress: list[GroupMemberRecord],
    ) -> str | None:
        ...  # return text to inject, or None to skip
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.escalation import InterventionConfig
    from ..engine.result import RolloutResult


@dataclass
class GroupMemberRecord:
    """Snapshot of one completed group member — passed to intervention handlers."""

    rollout: RolloutResult
    reward: float
    member_idx: int
    is_naked: bool


# Module-level cache so scripts/modules are loaded once per process.
_handler_cache: dict[str, Callable[..., Any]] = {}


def _load_handler(cfg: InterventionConfig) -> Callable[..., Any] | None:
    """Load and cache the callable for cfg. Returns None when nothing configured."""
    if not cfg.enabled:
        return None

    if cfg.lambda_expr:
        from ..types.yaml_type import parse_lambda  # noqa: PLC0415

        return parse_lambda(cfg.lambda_expr)

    if cfg.script:
        if cfg.script in _handler_cache:
            return _handler_cache[cfg.script]
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location("_esc_intervention", cfg.script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load intervention script: {cfg.script!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        fn = getattr(mod, "handle", None)
        if fn is None:
            raise AttributeError(f"Script {cfg.script!r} has no handle() function")
        _handler_cache[cfg.script] = fn
        return fn

    if cfg.callable_path:
        if cfg.callable_path in _handler_cache:
            return _handler_cache[cfg.callable_path]
        import importlib  # noqa: PLC0415

        sep = ":" if ":" in cfg.callable_path else "."
        mod_path, fn_name = cfg.callable_path.rsplit(sep, 1)
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, fn_name)
        _handler_cache[cfg.callable_path] = fn
        return fn

    return None


def call_intervention(
    cfg: InterventionConfig,
    prompt: str,
    answer: str | None,
    ground_truth: str | None,
    group_progress: list[GroupMemberRecord],
) -> str | None:
    """Call the configured handler. Swallows exceptions (prints + returns None)."""
    handler = _load_handler(cfg)
    if handler is None:
        return None
    try:
        return handler(prompt, answer, ground_truth, group_progress)
    except Exception as e:  # noqa: BLE001
        print(f"[intervention] handler raised {type(e).__name__}: {e} — skipping")
        return None


def build_group_progress(group_rollouts: list[RolloutResult]) -> list[GroupMemberRecord]:
    """Build GroupMemberRecord list from completed rollouts (reads _esc_* metadata)."""
    return [
        GroupMemberRecord(
            rollout=r,
            reward=float(r.metadata.get("_esc_reward", 0.0)),
            member_idx=int(r.metadata.get("_esc_member_idx", 0)),
            is_naked=bool(r.metadata.get("_view_is_naked", False)),
        )
        for r in group_rollouts
    ]
