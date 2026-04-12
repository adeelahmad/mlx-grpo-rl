"""InspectConfig — frozen dataclass for the inspect subcommand.

Build via build_inspect_config(raw_dict) — ignores unknown keys, coerces list→tuple.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class InspectConfig:
    """Configuration for mechanistic interpretability analysis."""

    model_path: str = ""
    model_b_path: str | None = None
    prompt: str = ""
    prompts_file: str | None = None  # JSON [{prompt, label}]
    control_prompt: str | None = None  # SHAMAN H4
    save_dir: str = "./inspect_out"
    capture_sublayers: bool = False
    trace_generation: bool = False
    max_new_tokens: int = 30
    top_k_logit_lens: int = 5
    layers_to_plot: tuple[int, ...] = ()  # empty = all
    circuit_layers: tuple[int, ...] = ()  # empty = all
    proj_dim: int = 128
    sketch_seed: int = 12345
    # feature values: compare|checkpoint|reward_strat|lora|fingerprint|scaffold|probe_save
    feature: str = "compare"
    checkpoints_dir: str | None = None  # Feature 1
    rollout_log: str | None = None  # Feature 2
    lora_path: str | None = None  # Feature 3/4
    probe_base: str | None = None  # Feature 6
    probe_wandb_key: str = "inspect/cka_drift"


_TUPLE_INT_FIELDS = frozenset({"layers_to_plot", "circuit_layers"})

_VALID_FEATURES = frozenset(
    {"compare", "checkpoint", "reward_strat", "lora", "fingerprint", "scaffold", "probe_save"}
)


def build_inspect_config(raw: dict[str, Any]) -> InspectConfig:
    """Build InspectConfig from a raw dict, ignoring unknown keys."""
    fields = InspectConfig.__dataclass_fields__
    filtered: dict[str, Any] = {}

    for k, v in raw.items():
        if k not in fields:
            continue

        # list → tuple for tuple[int, ...] fields
        if k in _TUPLE_INT_FIELDS:
            v = tuple(int(x) for x in v) if isinstance(v, list | tuple) else ()
            filtered[k] = v
            continue

        # String coercion for numeric fields
        if isinstance(v, str):
            field_type = fields[k].type
            if field_type in ("float", float) or "float" in str(field_type):
                with contextlib.suppress(ValueError):
                    v = float(v)
            elif field_type in ("int", int) or "int" in str(field_type):
                with contextlib.suppress(ValueError):
                    v = int(v)

        filtered[k] = v

    cfg = InspectConfig(**filtered)

    if cfg.feature not in _VALID_FEATURES:
        raise ValueError(
            f"Invalid feature {cfg.feature!r}. Valid values: {sorted(_VALID_FEATURES)}"
        )

    return cfg
