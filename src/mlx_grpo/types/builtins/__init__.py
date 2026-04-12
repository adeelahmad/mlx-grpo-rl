"""Built-in task types — auto-registered via import."""

from .general_qna import GeneralQnAType  # noqa: F401
from .identity import IdentityType  # noqa: F401
from .image_captioning import ImageCaptioningType  # noqa: F401
from .lotto import LottoType  # noqa: F401
from .math import MathType  # noqa: F401
from .mcq import ExamType, MCQType  # noqa: F401
from .multi_checkpoint import MultiCheckpointType  # noqa: F401
from .python_code import PythonCodeType  # noqa: F401
from .python_verified import PythonVerifiedType  # noqa: F401
from .tool_call import ToolCallType  # noqa: F401
from .variable_prefill import VariablePrefillType  # noqa: F401
from .vqa import VQAType  # noqa: F401

__all__ = [
    "GeneralQnAType",
    "IdentityType",
    "LottoType",
    "MathType",
    "ExamType",
    "MCQType",
    "MultiCheckpointType",
    "PythonCodeType",
    "PythonVerifiedType",
    "ToolCallType",
    "VariablePrefillType",
    "VQAType",
    "ImageCaptioningType",
]
