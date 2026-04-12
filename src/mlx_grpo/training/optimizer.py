"""Configurable optimizer & LR schedule builder for MLX training.

Reads ``optimizer`` and ``lr_schedule`` dicts from TrainingConfig and
constructs the corresponding MLX objects. Empty dicts → Adam with flat LR
(backward-compatible default).

    optimizer = build_optimizer(config.training)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.optimizers as optim

if TYPE_CHECKING:
    from ..config.loader import TrainingConfig

# ── Registry maps ────────────────────────────────────────────────────

_OPTIMIZERS: dict[str, type] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "lion": optim.Lion,
    "adagrad": optim.Adagrad,
    "adafactor": optim.Adafactor,
    "adamax": optim.Adamax,
}

_SCHEDULES: dict[str, Any] = {
    "cosine_decay": optim.cosine_decay,
    "linear": optim.linear_schedule,
    "step_decay": optim.step_decay,
    "exponential_decay": optim.exponential_decay,
}

# User-friendly aliases → MLX parameter names
_SCHEDULE_ALIASES: dict[str, str] = {
    "end_lr": "end",
    "steps": "steps",
}


def build_optimizer(training_cfg: TrainingConfig) -> optim.Optimizer:
    """Build an MLX optimizer with optional LR schedule from config."""
    lr = training_cfg.learning_rate
    opt_cfg = dict(training_cfg.optimizer)  # shallow copy
    sched_cfg = dict(training_cfg.lr_schedule)

    # Build learning rate (schedule or constant)
    learning_rate: float | Any = _build_schedule(lr, sched_cfg) if sched_cfg else lr

    # Build optimizer
    if not opt_cfg:
        return optim.Adam(learning_rate=learning_rate)

    name = opt_cfg.pop("name", "adam").lower()
    if name not in _OPTIMIZERS:
        avail = ", ".join(sorted(_OPTIMIZERS))
        msg = f"Unknown optimizer '{name}'. Available: {avail}"
        raise ValueError(msg)

    cls = _OPTIMIZERS[name]
    return cls(learning_rate=learning_rate, **opt_cfg)


def _build_schedule(lr: float, sched_cfg: dict[str, Any]) -> Any:
    """Build an MLX LR schedule, optionally with linear warmup."""
    if not sched_cfg:
        return lr

    cfg = dict(sched_cfg)  # shallow copy
    warmup_steps = cfg.pop("warmup_steps", 0)
    name = cfg.pop("name", "")
    if not name:
        return lr

    decay = _build_decay(lr, name, cfg)

    if warmup_steps and warmup_steps > 0:
        warmup = optim.linear_schedule(0.0, lr, warmup_steps)
        return optim.join_schedules([warmup, decay], [warmup_steps])

    return decay


def _build_decay(lr: float, name: str, kwargs: dict[str, Any]) -> Any:
    """Map a schedule name to the corresponding MLX schedule function."""
    if name not in _SCHEDULES:
        avail = ", ".join(sorted(_SCHEDULES))
        msg = f"Unknown LR schedule '{name}'. Available: {avail}"
        raise ValueError(msg)

    # Remap user-friendly keys to MLX parameter names
    for alias, real in _SCHEDULE_ALIASES.items():
        if alias in kwargs:
            kwargs[real] = kwargs.pop(alias)

    fn = _SCHEDULES[name]
    return fn(init=lr, **kwargs)


def describe_optimizer(optimizer: optim.Optimizer, training_cfg: TrainingConfig) -> str:
    """Human-readable one-line summary for logging."""
    opt_cfg = training_cfg.optimizer
    sched_cfg = training_cfg.lr_schedule
    opt_name = opt_cfg.get("name", "adam") if opt_cfg else "adam"
    lr = training_cfg.learning_rate

    parts = [f"Optimizer: {opt_name}(lr={lr}"]

    # Include notable kwargs
    for k, v in (opt_cfg or {}).items():
        if k != "name":
            parts[0] += f", {k}={v}"
    parts[0] += ")"

    if sched_cfg and sched_cfg.get("name"):
        sched_name = sched_cfg["name"]
        warmup = sched_cfg.get("warmup_steps", 0)
        sched_parts = [f"schedule={sched_name}"]
        if warmup:
            sched_parts.append(f"warmup={warmup}")
        for k, v in sched_cfg.items():
            if k not in ("name", "warmup_steps"):
                sched_parts.append(f"{k}={v}")
        parts.append(", ".join(sched_parts))

    return " | ".join(parts)
