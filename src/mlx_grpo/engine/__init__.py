"""Template-driven rollout engine.

Public API::

    from mlx_grpo.engine import RolloutContext, RolloutExecutor, RolloutResult
    from mlx_grpo.engine import BatchContext, CallbackRegistry
    from mlx_grpo.engine import Segment, GenerateResult, SegmentSource
    from mlx_grpo.engine import TemplateCompiler
"""

from .callbacks import CallbackRegistry
from .compiler import TemplateCompiler, TemplateExecutor, TemplateOp
from .context import BatchContext, RolloutContext
from .degenerate import DegenerateDetector
from .executor import RolloutExecutor, RolloutFn
from .result import GenerateResult, RolloutResult, Segment, SegmentSource

__all__ = [
    "BatchContext",
    "CallbackRegistry",
    "DegenerateDetector",
    "GenerateResult",
    "RolloutContext",
    "RolloutExecutor",
    "RolloutFn",
    "RolloutResult",
    "Segment",
    "SegmentSource",
    "TemplateCompiler",
    "TemplateExecutor",
    "TemplateOp",
]
