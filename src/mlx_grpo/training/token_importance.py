"""Token-level importance sampling weight computation for GRPO.

Addresses verbosity: gives higher gradient weight to shorter completions
and/or earlier tokens, discouraging the model from padding with redundant text.

Two modes (set via TokenISConfig.mode):
  token   — positional decay + length scale. Later tokens and longer completions
             get lower gradient weight: decay_base^(t/n) * (ref_len/n)^exponent
  segment — tag-based weights. Thinking/reasoning gets lower weight than answer.

Pure Python, no MLX — called before nn.value_and_grad to produce weights
that are then passed as segment_weights to grpo_loss_fn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_grpo.config.loader import TokenISConfig


def compute_length_scale(n: int, cfg: TokenISConfig) -> float:
    """Length scale: shorter completions get higher gradient weight.

    ls = (ref_length / n) ^ length_exponent

    With length_exponent=0.5 and ref_length=256:
      n=64   → ls=2.0  (4x shorter → 2x weight)
      n=256  → ls=1.0  (at reference length)
      n=1024 → ls=0.5  (4x longer → 0.5x weight)
    """
    if cfg.length_exponent == 0.0 or cfg.ref_length <= 0 or n <= 0:
        return 1.0
    return (cfg.ref_length / n) ** cfg.length_exponent


def compute_position_weights(n: int, cfg: TokenISConfig) -> list[float]:
    """Positional decay weights: w(t) = decay_base^(t/n).

    With decay_base=0.5:
      t=0   → w=1.0   (first token: full weight)
      t=n/2 → w≈0.71  (middle: 71% weight)
      t=n-1 → w≈0.5   (last token: 50% weight)

    decay_base=1.0 → uniform weights (no positional decay).
    """
    if n == 0:
        return []
    if cfg.decay_base >= 1.0:
        return [1.0] * n
    return [cfg.decay_base ** (t / n) for t in range(n)]


def compute_segment_weights(
    n: int,
    cfg: TokenISConfig,
    answer_mask: list[bool] | None,
    thinking_mask: list[bool] | None,
) -> list[float]:
    """Per-token weights from segment tags.

    thinking tokens → cfg.thinking_weight (e.g. 0.5, reduces thinking gradient)
    answer tokens   → cfg.answer_weight   (e.g. 2.0, amplifies answer gradient)
    other tokens    → 1.0
    """
    weights = [1.0] * n
    if thinking_mask is not None:
        for i, is_think in enumerate(thinking_mask[:n]):
            if is_think:
                weights[i] = cfg.thinking_weight
    if answer_mask is not None:
        for i, is_ans in enumerate(answer_mask[:n]):
            if is_ans:
                weights[i] = cfg.answer_weight
    return weights


def compute_scaffold_proximity_weights(
    n: int,
    cfg: TokenISConfig,
    scaffold_mask: list[bool],
) -> list[float]:
    """Distance-from-scaffold weights.

    Two modes controlled by cfg.invert_proximity:

    invert_proximity=False (default — near gets more):
      w(t) = proximity_decay ^ distance   →  near scaffold = high weight

    invert_proximity=True (ICL mode — far gets more):
      w(t) = proximity_decay + (1 - proximity_decay) * (dist / max_dist)
      proximity_decay is the minimum weight assigned to scaffold-adjacent tokens.
      Tokens at maximum distance receive weight 1.0.
      Rationale: ICL handles tokens near scaffold automatically; tokens the model
      generates far from any scaffold represent independent capability and deserve
      a stronger gradient signal.

    O(n) two-pass algorithm: forward then backward min-distance.

    Args:
        n: Completion token count.
        cfg: TokenISConfig with proximity_decay and invert_proximity fields.
        scaffold_mask: Per-token bool; True = injected scaffold token.
    """
    if n == 0:
        return []
    scale = cfg.proximity_scale
    if not any(scaffold_mask[:n]):
        # No scaffold → naked member. Apply scale only if naked_apply_scale=True.
        naked_w = scale if getattr(cfg, "naked_apply_scale", True) else 1.0
        return [naked_w] * n if naked_w != 1.0 else [1.0] * n

    # Forward-only pass: distance from most recent scaffold injection.
    # d is primed to -1 on scaffold tokens so the FIRST generated token after
    # injection gets dist=0 → weight = decay^0 * scale = scale (e.g. 1.3).
    # No backward pass: tokens before the next scaffold are NOT boosted.
    dist = [n] * n
    d = n
    for t in range(n):
        if t < len(scaffold_mask) and scaffold_mask[t]:
            d = -1  # prime: next token starts at dist=0
            dist[t] = 0  # scaffold token itself (masked in loss anyway)
        else:
            d = min(d + 1, n)
            dist[t] = d

    pcs = getattr(cfg, "proximity_chunk_split", 1.0)
    if pcs < 1.0:
        # Per-chunk direction: identify scaffold blocks and assign each chunk near or far.
        # Chunk k < n_near → near-proximity (decay^dist).
        # Chunk k >= n_near → far-proximity (lo + (1-lo)*(dist/chunk_max_dist)).
        # Ignores cfg.invert_proximity — use proximity_chunk_split OR invert_proximity, not both.
        chunk = [-1] * n
        ck, prev = -1, False
        for t in range(n):
            cur = bool(t < len(scaffold_mask) and scaffold_mask[t])
            if cur and not prev:
                ck += 1
            chunk[t] = ck
            prev = cur
        n_ck = ck + 1
        n_near = round(n_ck * pcs) if n_ck else 0
        ck_max = [0] * max(n_ck, 1)
        for t in range(n):
            if chunk[t] >= 0 and not (t < len(scaffold_mask) and scaffold_mask[t]):
                ck_max[chunk[t]] = max(ck_max[chunk[t]], dist[t])
        lo, pd = cfg.proximity_decay, cfg.proximity_decay
        out = []
        for t in range(n):
            c = chunk[t]
            is_scaf = bool(t < len(scaffold_mask) and scaffold_mask[t])
            if c < 0 or is_scaf:
                out.append(scale)
                continue
            w = lo + (1.0 - lo) * (dist[t] / max(ck_max[c], 1)) if c >= n_near else pd ** dist[t]
            out.append(w * scale if scale != 1.0 else w)
        return out

    if cfg.invert_proximity:
        # Far tokens get MORE gradient. Normalize distance to [0, 1] then
        # scale to [proximity_decay, 1.0] so scaffold-adjacent tokens are
        # downweighted rather than zeroed.
        max_d = max(max(dist), 1)
        lo = cfg.proximity_decay
        weights = [lo + (1.0 - lo) * (dist[t] / max_d) for t in range(n)]
        return [w * scale for w in weights] if scale != 1.0 else weights

    if cfg.proximity_decay >= 1.0:
        return [scale] * n if scale != 1.0 else [1.0] * n
    weights = [cfg.proximity_decay ** dist[t] for t in range(n)]
    return [w * scale for w in weights] if scale != 1.0 else weights


def compute_is_weights(
    n: int,
    cfg: TokenISConfig,
    answer_mask: list[bool] | None = None,
    thinking_mask: list[bool] | None = None,
    scaffold_mask: list[bool] | None = None,
) -> list[float] | None:
    """Compute combined IS weights for one completion of n tokens.

    Returns None for mode='none' (avoids allocation, fully backward compat).

    Args:
        n: Completion token count (post-truncation).
        cfg: TokenISConfig with mode, decay params, segment weights.
        answer_mask: Per-token bool; True = answer segment (segment mode).
        thinking_mask: Per-token bool; True = thinking segment (segment mode).
        scaffold_mask: Per-token bool; True = injected scaffold (scaffold_proximity mode).

    Returns:
        list[float] of length n, or None when IS is disabled.
    """
    if cfg.mode == "none" or n == 0:
        return None

    if cfg.mode == "segment":
        return compute_segment_weights(n, cfg, answer_mask, thinking_mask)

    if cfg.mode == "scaffold_proximity":
        return compute_scaffold_proximity_weights(n, cfg, scaffold_mask or [])

    # mode == "token": positional decay combined with length scale
    ls = compute_length_scale(n, cfg)
    pos_w = compute_position_weights(n, cfg)
    return [ls * w for w in pos_w]


def build_is_config(
    base: TokenISConfig | None,
    overrides: dict[str, Any],
) -> TokenISConfig:
    """Build a TokenISConfig by merging overrides onto a base.

    base=None uses TokenISConfig() defaults. Unknown keys are ignored.
    """
    from dataclasses import fields, replace

    from mlx_grpo.config.loader import TokenISConfig

    cfg = base if base is not None else TokenISConfig()
    valid = {f.name for f in fields(cfg)}
    filtered = {k: v for k, v in overrides.items() if k in valid}
    return replace(cfg, **filtered) if filtered else cfg
