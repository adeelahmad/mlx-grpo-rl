"""mlx_grpo.inspect — mechanistic interpretability analysis subpackage.

Public API::

    from mlx_grpo.inspect import InspectConfig, build_inspect_config, run_inspect, CKAProbe
"""

from __future__ import annotations

from .cka_probe import CKAProbe, CKAProbeResult
from .config import InspectConfig, build_inspect_config
from .runner import run_inspect

__all__ = [
    "CKAProbe",
    "CKAProbeResult",
    "InspectConfig",
    "build_inspect_config",
    "run_inspect",
]
