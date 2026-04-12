"""Prompt KV cache utilities for GRPO group rollouts.

Within a GRPO group, all G rollouts share the same prompt. Instead of
redundantly prefilling the prompt G times, we prefill once and clone
the cache for each group member — reducing prefill cost from O(G*P) to O(P).

Uses mlx-lm's ``make_prompt_cache`` and ``generate_step`` with
``max_tokens=0`` to build the prompt cache without generating tokens.

We cache all-but-last prompt token (P-1). This guarantees that every
``generate_step`` call has at least 1 new token to process (the last
prompt token), avoiding the empty-prompt edge case while still saving
the bulk of the prefill work.
"""

from __future__ import annotations

from typing import Any


def prefill_prompt_cache(
    model: Any,
    prompt_tokens: list[int],
    *,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    max_kv_size: int | None = None,
    quantized_kv_start: int = 0,
    prefill_step_size: int = 512,
) -> list[Any] | None:
    """Prefill a prompt cache with all-but-last prompt token.

    Caching P-1 tokens means every subsequent ``generate_step`` call
    gets at least the last prompt token as input, which is required
    for the model to produce logits for the first generated position.

    Returns:
        List of KVCache objects with offset = P-1, or None if prompt
        is too short to benefit from caching (len <= 1).
    """
    if len(prompt_tokens) <= 1:
        return None

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    try:
        from mlx_lm.generate import generate_step
    except ImportError:
        from mlx_lm.utils import generate_step

    cache = make_prompt_cache(model, max_kv_size)

    step_kw: dict[str, Any] = {"prompt_cache": cache, "max_tokens": 0}
    if kv_bits is not None:
        step_kw["kv_bits"] = kv_bits
        step_kw["kv_group_size"] = kv_group_size
        if quantized_kv_start > 0:
            step_kw["quantized_kv_start"] = quantized_kv_start
    if prefill_step_size != 512:
        step_kw["prefill_step_size"] = prefill_step_size

    # Cache all-but-last token; last token is always re-processed
    prompt = mx.array(prompt_tokens[:-1])
    for _ in generate_step(prompt, model, **step_kw):
        pass

    # Materialize KV state so clones don't carry lazy graph references
    mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])
    return cache


def clone_prompt_cache(cache: list[Any]) -> list[Any]:
    """Clone a prefilled prompt cache for independent generation.

    Each clone gets its own KVCache objects with copied state. After
    cloning, ``offset == keys.shape[2]``, so any new token in
    ``update_and_fetch`` triggers the growth path (``mx.concatenate``)
    which creates new arrays — the original cache is never mutated.

    Returns:
        Fresh list of KVCache objects with identical KV state.
    """
    from mlx_lm.models.cache import KVCache

    cloned: list[Any] = []
    for layer_cache in cache:
        c = KVCache()
        c.state = layer_cache.state
        cloned.append(c)
    return cloned
