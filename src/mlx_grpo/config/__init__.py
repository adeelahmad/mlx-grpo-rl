"""Configuration — YAML loading with dynamic type creation."""

from .escalation import EscalationConfig, build_escalation_config
from .loader import Config, load_config, merge_configs
from .metal_budget import SAFE_GRAD_TOKENS, check_metal_budget, recommended_max_grad_seq_len
from .vision import VisionConfig

__all__ = [
    "Config",
    "EscalationConfig",
    "SAFE_GRAD_TOKENS",
    "VisionConfig",
    "build_escalation_config",
    "check_metal_budget",
    "load_config",
    "merge_configs",
    "recommended_max_grad_seq_len",
]
