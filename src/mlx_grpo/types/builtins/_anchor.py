"""ReasonAnchor injection helper — shared by all built-in task types.

Reads four keys from ctx.type_config:
  reason_anchor: bool                       — enable/disable (default False)
  reason_anchor_masked: bool                — exclude anchor tokens from loss (default True)
  reason_anchor_weight: float               — loss multiplier when masked=False (default 1.0)
      0.0 → same effect as masked=True (no gradient)
      1.0 → include in loss at normal weight
      1.1 → anchor tokens count 10% more in the loss
  reason_anchor_max_words: int              — cap on anchor length in words (default 80)
  reason_anchor_opposite_tokens: bool       — append high-contrast token pairs (default False)
  reason_anchor_opposite_k: int            — number of contrast pairs to append (default 3)
  reason_anchor_similarity_threshold: float — cosine sim ceiling for "opposite" (default 0.10)

CoT source priority: ctx.sample["cot"] → ctx.sample["thinking"] → ctx.answer.
The CoT is sanitized before passing to generate_anchor: <think>/</think> tags
and \\boxed{...} spans are stripped so the anchor contains only the reasoning
skeleton, not raw completion tokens.

When reason_anchor_opposite_tokens=True, the most semantically distant token
pairs from the CoT are discovered via the model's embedding layer (a single cheap
embedding lookup, no full forward pass) and appended to the anchor as:
  "Contrasts: {a} / {b}, {c} / {d}"
This gives the model explicit semantic poles to reason about during generation.

Gracefully no-ops if reason_anchor=False or the reason_anchor library is unavailable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...engine.context import RolloutContext

# ── One-time import with src/ path fallback ───────────────────────────────────
_generate_anchor: Any = None
try:
    from reason_anchor import generate_anchor as _generate_anchor  # type: ignore[import]
except ImportError:
    try:
        _src_dir = str(Path(__file__).parents[4])  # …/src/mlx_grpo/types/builtins → src/
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)
        from reason_anchor import generate_anchor as _generate_anchor  # type: ignore[import]
    except ImportError:
        pass  # Library absent — inject_reason_anchor becomes a no-op

# Pre-compiled patterns for CoT sanitization
_RE_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
_RE_BOXED = re.compile(r"\\boxed\{[^}]*\}")
_RE_EXCESS_NL = re.compile(r"\n{3,}")


def _sanitize_cot(text: str) -> str:
    """Strip thinking tags and boxed answers from a CoT string."""
    text = _RE_THINK_TAG.sub("", text)
    text = _RE_BOXED.sub("", text)
    text = _RE_EXCESS_NL.sub("\n\n", text)
    return text.strip()


def _find_opposite_tokens(
    backend: Any,
    text: str,
    k: int = 3,
    threshold: float = 0.10,
) -> list[tuple[str, str]]:
    """Find k token pairs with the lowest cosine similarity in text.

    Uses the model's embedding layer — a single cheap lookup with no
    autoregressive forward pass. Falls back to [] on any failure so
    it never affects rollout stability.
    """
    try:
        import mlx.core as mx
        import numpy as np

        model = getattr(backend, "model", None)
        embed_layer = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embed_layer is None:
            return []

        token_ids = backend.encode(text)
        if len(token_ids) < 4:
            return []

        # Embedding lookup only — materialize immediately to release MLX graph
        emb = np.array(embed_layer(mx.array(token_ids)))  # (n, hidden_dim)
        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        normed = emb / (norms + 1e-8)
        sim = normed @ normed.T  # (n, n) pairwise cosine similarity

        # Lower-triangle indices give each unique pair exactly once
        n = len(token_ids)
        tri_i, tri_j = np.tril_indices(n, k=-1)
        sims = sim[tri_i, tri_j]
        mask = sims < threshold
        if not mask.any():
            return []

        order = np.argsort(sims[mask])
        idx_a = tri_i[mask][order]
        idx_b = tri_j[mask][order]

        seen: set[frozenset[str]] = set()
        result: list[tuple[str, str]] = []
        for i, j in zip(idx_a, idx_b, strict=False):
            ta = backend.decode([token_ids[int(i)]]).strip()
            tb = backend.decode([token_ids[int(j)]]).strip()
            if not ta or not tb or ta == tb:
                continue
            key: frozenset[str] = frozenset({ta, tb})
            if key in seen:
                continue
            seen.add(key)
            result.append((ta, tb))
            if len(result) >= k:
                break
        return result
    except Exception:
        return []


def inject_reason_anchor(ctx: RolloutContext) -> None:
    """Inject a ReasonAnchor after the scaffold if configured.

    Call this immediately after the scaffold inject block in each type's
    rollout(), before the first ctx.generate() call.
    """
    if _generate_anchor is None:
        return

    cfg = ctx.type_config
    if not cfg.get("reason_anchor", False):
        return

    masked = bool(cfg.get("reason_anchor_masked", True))
    weight = float(cfg.get("reason_anchor_weight", 1.0))
    max_words = int(cfg.get("reason_anchor_max_words", 80))

    _cot_raw = ctx.sample.get("cot") or ctx.sample.get("thinking") or ctx.answer or ""
    _cot = _sanitize_cot(_cot_raw)

    try:
        result = _generate_anchor(ctx.prompt or "", _cot, ctx.answer or "", max_words=max_words)
        anchor_text = result.anchor

        if cfg.get("reason_anchor_opposite_tokens", False) and anchor_text:
            k = int(cfg.get("reason_anchor_opposite_k", 3))
            thresh = float(cfg.get("reason_anchor_similarity_threshold", 0.10))
            # Search across anchor+prompt+answer space — the full semantic context
            # the model must reason within, not just the raw CoT.
            _opp_text = anchor_text + " " + (ctx.prompt or "") + " " + (ctx.answer or "")
            pairs = _find_opposite_tokens(ctx._backend, _opp_text, k=k, threshold=thresh)
            if pairs:
                contrasts = ", ".join(f"{a} / {b}" for a, b in pairs)
                anchor_text = f"{anchor_text}\nContrasts: {contrasts}"

        ctx.inject(
            anchor_text,
            masked=masked,
            tag="reason_anchor",
            loss_weight=max(weight, 0.0),
        )
    except Exception:
        pass  # Never let anchor failure crash a rollout
