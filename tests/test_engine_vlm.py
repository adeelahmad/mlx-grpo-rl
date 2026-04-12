"""Tests for VLM image propagation through RolloutContext and RolloutExecutor
(Phase 1, Task 3.3)."""

from __future__ import annotations

import base64

from mlx_grpo.data import Sample
from mlx_grpo.engine.context import RolloutContext
from mlx_grpo.engine.executor import RolloutExecutor

# ── RolloutContext image tests ────────────────────────────────────────────────


def test_rollout_context_no_images(mock_backend):
    """ctx created with no images → generate() calls backend with images=None."""
    ctx = RolloutContext(
        backend=mock_backend,
        prompt="Hello",
        prompt_tokens=[72, 101, 108, 108, 111],
    )
    assert ctx._images is None
    ctx.generate(max_tokens=8)
    assert mock_backend.last_images is None


def test_rollout_context_ctx_level_images(mock_backend):
    """ctx created with images → generate() passes them to the backend."""
    ctx = RolloutContext(
        backend=mock_backend,
        prompt="Describe this image",
        prompt_tokens=[68],
        images=["https://example.com/cat.jpg"],
    )
    assert ctx._images == ["https://example.com/cat.jpg"]
    ctx.generate(max_tokens=8)
    assert mock_backend.last_images == ["https://example.com/cat.jpg"]


def test_rollout_context_call_level_override(mock_backend):
    """ctx has images A; generate(images=B) → backend gets B (call-level wins)."""
    ctx = RolloutContext(
        backend=mock_backend,
        prompt="Multi-turn",
        prompt_tokens=[77],
        images=["https://example.com/A.jpg"],
    )
    ctx.generate(max_tokens=8, images=["https://example.com/B.jpg"])
    assert mock_backend.last_images == ["https://example.com/B.jpg"]


def test_rollout_context_call_none_uses_ctx_images(mock_backend):
    """When generate(images=None) is called, the ctx-level images are used."""
    ctx = RolloutContext(
        backend=mock_backend,
        prompt="Text",
        prompt_tokens=[84],
        images=["https://example.com/ctx.jpg"],
    )
    ctx.generate(max_tokens=8, images=None)
    assert mock_backend.last_images == ["https://example.com/ctx.jpg"]


# ── RolloutExecutor image tests ───────────────────────────────────────────────


def test_executor_text_sample_no_images(mock_backend):
    """Sample with no image fields → executor passes images=None to ctx."""
    executor = RolloutExecutor(mock_backend)
    sample = Sample(prompt="Q", answer="A")

    captured_ctx_images: list | None = []

    def rollout(ctx: RolloutContext) -> None:
        captured_ctx_images.clear()
        if ctx._images is not None:
            captured_ctx_images.extend(ctx._images)
        ctx.generate(max_tokens=4)

    executor.execute(rollout_fn=rollout, prompt="Q", sample_obj=sample)
    assert len(captured_ctx_images) == 0
    assert mock_backend.last_images is None


def test_executor_image_path_sample(mock_backend):
    """Sample(image_path='x.jpg') → ctx._images == ['x.jpg']."""
    executor = RolloutExecutor(mock_backend)
    sample = Sample(prompt="Describe", answer="A cat", image_path="https://example.com/x.jpg")

    result_images: list[str | bytes] | None = None

    def rollout(ctx: RolloutContext) -> None:
        nonlocal result_images
        result_images = ctx._images
        ctx.generate(max_tokens=4)

    executor.execute(rollout_fn=rollout, prompt="Describe", sample_obj=sample)
    assert result_images == ["https://example.com/x.jpg"]


def test_executor_image_b64_sample(mock_backend):
    """Sample(image_b64=...) → ctx._images contains decoded bytes."""
    executor = RolloutExecutor(mock_backend)
    raw_bytes = b"\x89PNG\r\n\x1a\n"
    b64_str = base64.b64encode(raw_bytes).decode()
    sample = Sample(prompt="What colour?", answer="Red", image_b64=b64_str)

    result_images: list[str | bytes] | None = None

    def rollout(ctx: RolloutContext) -> None:
        nonlocal result_images
        result_images = ctx._images
        ctx.generate(max_tokens=4)

    executor.execute(rollout_fn=rollout, prompt="What colour?", sample_obj=sample)
    assert result_images is not None
    assert len(result_images) == 1
    assert result_images[0] == raw_bytes


def test_executor_image_path_priority(mock_backend):
    """When both image_path and image_b64 are set, image_path wins."""
    executor = RolloutExecutor(mock_backend)
    sample = Sample(
        prompt="Q",
        answer="A",
        image_path="https://example.com/priority.jpg",
        image_b64="ZGF0YQ==",
    )

    result_images: list[str | bytes] | None = None

    def rollout(ctx: RolloutContext) -> None:
        nonlocal result_images
        result_images = ctx._images
        ctx.generate(max_tokens=4)

    executor.execute(rollout_fn=rollout, prompt="Q", sample_obj=sample)
    assert result_images == ["https://example.com/priority.jpg"]
