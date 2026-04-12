"""Generation backend protocol — the abstraction boundary for model inference.

Every backend implements one method: generate tokens from a prefix.
The engine never touches model internals directly.

Current: MLX
Future:  PyTorch, vLLM, llama.cpp, Ollama, OpenAI-compatible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class GenerateParams:
    """Parameters for a single generation call."""

    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.3
    repetition_context_size: int = 40
    logit_bias: dict[int, float] | None = None
    # XTC (eXclusion-based Token Culling) sampler params
    xtc_probability: float = 0.0  # 0.0 = disabled
    xtc_threshold: float = 0.1
    xtc_special_tokens: set[int] | None = None
    min_tokens_to_keep: int = 1
    # Reproducibility
    seed: int | None = None
    # Additional EOS tokens beyond model default
    extra_eos_tokens: list[int] = field(default_factory=list)
    # Pre-populated KV cache (from prefill_prompt_cache); backends that
    # support it will skip redundant prompt prefill.
    prompt_cache: Any = None
    # VLM: images passed as local paths (str) or raw bytes.
    # None = text-only mode (backward compatible).
    images: list[str | bytes] | None = None
    # KV cache quantization
    kv_bits: int | None = None
    kv_group_size: int = 64


@dataclass(slots=True)
class GenerateOutput:
    """Raw output from a generation backend."""

    text: str
    tokens: list[int]
    stop_reason: str = "eos"  # "stop_sequence" | "max_tokens" | "eos"
    stop_sequence: str | None = None
    logprobs: Any = None  # Optional: per-token log probs


@runtime_checkable
class GenerationBackend(Protocol):
    """Protocol for generation backends.

    Implementations register via ``__init_subclass__`` on ``BackendRegistry``.

    Optional: backends may implement generate_batch() for efficient batched generation.
    If not present, use generate_batch_serial() helper (falls back to serial generate calls).
    """

    @property
    def name(self) -> str: ...

    def generate(self, prefix_tokens: list[int], params: GenerateParams) -> GenerateOutput:
        """Generate continuation from prefix tokens."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> list[int]:
        """Apply chat template and return token IDs."""
        ...

    def generate_batch(
        self,
        prefix_tokens_batch: list[list[int]],
        params: GenerateParams,
    ) -> list[GenerateOutput]:
        """Optional: Generate from multiple prefixes with identical params.

        If implemented, called when all sequences use same GenerateParams.
        If not implemented, use generate_batch_serial() fallback.

        Args:
            prefix_tokens_batch: List of token ID lists (multiple sequences)
            params: Generation parameters (applied uniformly to all)

        Returns:
            List of GenerateOutput, one per input sequence
        """
        ...


def generate_batch_serial(
    backend: GenerationBackend,
    prefix_tokens_batch: list[list[int]],
    params: GenerateParams,
) -> list[GenerateOutput]:
    """Fallback: serial generation when backend doesn't implement generate_batch.

    Non-breaking: safely handles backends that don't support batching.

    Args:
        backend: GenerationBackend instance
        prefix_tokens_batch: List of token ID lists
        params: Generation parameters (uniform across batch)

    Returns:
        List of GenerateOutput results
    """
    results: list[GenerateOutput] = []
    for prefix_tokens in prefix_tokens_batch:
        result = backend.generate(prefix_tokens, params)
        results.append(result)
    return results


class BackendRegistry:
    """Auto-registration for generation backends.

    Subclass to register::

        class MLXBackend(BackendRegistry, name="mlx"):
            ...
    """

    _registry: dict[str, type[BackendRegistry]] = {}

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls._registry[name] = cls

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> GenerationBackend:
        """Instantiate a backend by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown backend {name!r}. Available: {available}")
        return cls._registry[name](**kwargs)  # type: ignore[return-value]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None (protocol definition)
# Usage: Implement GenerationBackend protocol for custom backends
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
