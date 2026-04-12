"""Shared test fixtures — mock backend for testing without MLX hardware."""

import pytest

import mlx_grpo.rewards.accuracy  # noqa: F401
import mlx_grpo.rewards.efficiency  # noqa: F401
import mlx_grpo.rewards.hierarchical  # noqa: F401
import mlx_grpo.rewards.reasoning  # noqa: F401

# Ensure builtins are always registered for tests
import mlx_grpo.types.builtins  # noqa: F401
from mlx_grpo.backends.base import GenerateOutput, GenerateParams


class MockBackend:
    """Deterministic mock backend for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._call_count = 0
        self.last_images: list[str | bytes] | None = None

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prefix_tokens: list[int], params: GenerateParams) -> GenerateOutput:
        self._call_count += 1
        self.last_images = params.images  # capture images for test assertions
        text = self._responses.get(f"call_{self._call_count}", f"mock_{self._call_count}")

        stop_reason, stop_sequence = "eos", None
        for stop in params.stop:
            if stop in text:
                text = text[: text.index(stop)]
                stop_reason, stop_sequence = "stop_sequence", stop
                break

        tokens = self.encode(text)
        if len(tokens) > params.max_tokens:
            tokens = tokens[: params.max_tokens]
            text = self.decode(tokens)
            stop_reason = "max_tokens"

        return GenerateOutput(
            text=text, tokens=tokens, stop_reason=stop_reason, stop_sequence=stop_sequence
        )

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)

    @property
    def eos_token_id(self) -> int:
        return 0  # Mock EOS token

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("".join(m["content"] for m in messages))


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def mock_backend_factory():
    def factory(**responses):
        return MockBackend(responses=responses)

    return factory
