"""Generation backends — pluggable model inference."""

import logging

from .base import (
    BackendRegistry,
    GenerateOutput,
    GenerateParams,
    GenerationBackend,
    generate_batch_serial,
)

# Auto-register all backends (suppress ImportError for optional deps)
from .mlx_backend import MLXBackend  # noqa: F401
from .mlx_batch_generate_backend import MLXBatchGenerateBackend  # noqa: F401

_log = logging.getLogger(__name__)

try:
    from .pytorch_backend import PyTorchBackend  # noqa: F401
except ImportError:
    _log.debug("PyTorch backend unavailable (install torch + transformers)")

try:
    from .vllm_backend import VLLMBackend  # noqa: F401
except ImportError:
    _log.debug("vLLM backend unavailable (install vllm)")

try:
    from .llamacpp_backend import LlamaCppBackend  # noqa: F401
except ImportError:
    _log.debug("llama.cpp backend unavailable (install llama-cpp-python)")

__all__ = [
    "BackendRegistry",
    "GenerateOutput",
    "GenerateParams",
    "GenerationBackend",
    "generate_batch_serial",
]
