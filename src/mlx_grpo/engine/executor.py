"""Rollout executor — runs a rollout function to produce a RolloutResult.

The executor is the bridge between the training loop and the template engine.
It builds the context, runs the rollout, and returns the result.

    executor = RolloutExecutor(backend)
    result = executor.execute(type_def, sample, batch_ctx)
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

from ..backends.base import GenerationBackend
from ..data import Sample
from .callbacks import CallbackRegistry
from .context import BatchContext, RolloutContext
from .result import RolloutResult

# Type alias for rollout functions
RolloutFn = Callable[[RolloutContext], None]


class RolloutExecutor:
    """Executes rollout functions against a generation backend.

    Stateless — safe to share across iterations. All mutable state
    lives in the per-execution ``RolloutContext``.
    """

    def __init__(self, backend: GenerationBackend, gen_defaults: dict[str, Any] | None = None):
        self.backend = backend
        self._gen_defaults: dict[str, Any] = gen_defaults or {}

    @staticmethod
    def _images_from_sample(sample_obj: Sample | None) -> list[str | bytes] | None:
        """Resolve VLM images from a Sample dataclass.

        Priority: image_paths > image_b64s > image_path > image_b64
        image_b64 strings are decoded to bytes here so the backend always
        receives ``bytes`` rather than a base64 string.
        """
        if sample_obj is None:
            return None
        # Multi-image: image_paths (list of paths/URLs)
        if sample_obj.image_paths is not None:
            return list(sample_obj.image_paths)
        # Multi-image: image_b64s (list of base64 strings → decode to bytes)
        if sample_obj.image_b64s is not None:
            return [base64.b64decode(b64) for b64 in sample_obj.image_b64s]
        # Single-image: image_path
        if sample_obj.image_path is not None:
            return [sample_obj.image_path]
        # Single-image: image_b64
        if sample_obj.image_b64 is not None:
            return [base64.b64decode(sample_obj.image_b64)]
        return None

    def execute(
        self,
        rollout_fn: RolloutFn,
        prompt: str,
        prompt_tokens: list[int] | None = None,
        answer: str | None = None,
        type_name: str = "",
        type_config: dict[str, Any] | None = None,
        batch: BatchContext | None = None,
        callbacks: CallbackRegistry | None = None,
        sample: dict[str, Any] | None = None,
        chat_messages: list[dict[str, str]] | None = None,
        chat_kwargs: dict[str, Any] | None = None,
        sample_obj: Sample | None = None,
        **kwargs: Any,
    ) -> RolloutResult:
        """Execute a rollout function and return the result.

        Args:
            rollout_fn: The rollout template function ``(ctx) -> None``
            prompt: Raw prompt text
            prompt_tokens: Pre-tokenized prompt (computed if not provided)
            answer: Ground truth answer (if available)
            type_name: Task type name for tagging
            type_config: Type-specific config (temperature, etc.)
            batch: Training loop state
            callbacks: Python callback registry
            chat_messages: Chat messages for template application
            chat_kwargs: Extra args for apply_chat_template
            sample_obj: Optional typed Sample dataclass — used to resolve VLM images.
        """
        # Build prompt tokens
        if prompt_tokens is None:
            if chat_messages:
                prompt_tokens = self.backend.apply_chat_template(
                    chat_messages,
                    **(chat_kwargs or {}),
                )
            else:
                prompt_tokens = self.backend.encode(prompt)

        # Resolve VLM images from typed Sample (image_path > image_b64)
        images = self._images_from_sample(sample_obj)

        # Add prompt segment
        ctx = RolloutContext(
            backend=self.backend,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            answer=answer,
            type_name=type_name,
            type_config=type_config,
            batch=batch,
            callbacks=callbacks,
            sample=sample,
            max_completion_tokens=kwargs.get("max_completion_tokens", 0),
            prompt_cache=kwargs.get("prompt_cache"),
            gen_defaults=kwargs.get("gen_defaults") or self._gen_defaults,
            images=images,
        )

        # Attach batch collector for generate_batch() fusion (optional)
        _collector = kwargs.get("batch_collector")
        if _collector is not None:
            ctx._batch_collector = _collector

        # Pre-inject prefix hint before rollout (e.g. first N lines of thinking
        # for middle/postfix views — gives model a warm start for proximity loss)
        # Tag as "scaffold_prefix" so proximity decay recognizes it as a scaffold injection
        pre_inject = kwargs.get("pre_inject_text")
        if pre_inject:
            ctx.inject(pre_inject, masked=True, tag="scaffold_prefix")

        # Execute the rollout function
        try:
            rollout_fn(ctx)
        finally:
            # Mark done so other members waiting in the collector don't deadlock
            if _collector is not None:
                _collector.mark_done(batch.group_index if batch else 0)

        return ctx.build_result()

    def execute_batch(
        self,
        rollout_fn: RolloutFn,
        samples: list[dict[str, Any]],
        batch: BatchContext,
        callbacks: CallbackRegistry | None = None,
    ) -> list[RolloutResult]:
        """Execute rollout for a batch of samples.

        Each sample dict should have 'prompt' and optionally 'answer',
        'type', 'type_config', 'chat_messages', 'chat_kwargs'.
        """
        results = []
        for i, sample in enumerate(samples):
            sample_batch = BatchContext(
                iteration=batch.iteration,
                epoch=batch.epoch,
                total_iters=batch.total_iters,
                global_step=batch.global_step,
                group_index=batch.group_index,
                group_size=batch.group_size,
                batch_index=i,
                dataset_name=batch.dataset_name,
                custom=batch.custom,
            )
            result = self.execute(
                rollout_fn=rollout_fn,
                prompt=sample["prompt"],
                answer=sample.get("answer"),
                type_name=sample.get("type", ""),
                type_config=sample.get("type_config"),
                batch=sample_batch,
                callbacks=callbacks,
                sample=sample,
                chat_messages=sample.get("chat_messages"),
                chat_kwargs=sample.get("chat_kwargs"),
            )
            results.append(result)
        return results


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None (uses GenerationBackend protocol)
# Usage: executor = RolloutExecutor(backend)
#        result = executor.execute(rollout_fn, prompt="...", answer="...")
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
