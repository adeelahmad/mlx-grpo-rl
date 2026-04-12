"""Escalation planning helpers — scaffold boundary detection, two-phase generation,
tri-factor weight building, and group-level reward shaping.

Pure-Python, no MLX dependencies. Safe to call anywhere including tests.

Public API (imported by escalation.py):
  _count_n_eff_max              — max achievable n_eff from expert trace (no backend)
  _plan_adaptive_scaffold       — boundary-snapped scaffold splitting with token fallback
  _two_phase_generate           — Phase A (free) + Phase B (natural-stop) generation
  _build_tri_factor_weights     — per-token tri-factor importance weights
  _apply_group_shaping          — bidirectional group-level reward shaping (in-place)
  _jaccard_anchor_from_naked    — targeted anchor via naked-output × expert-trace intersection
  _strip_answer_close           — strip trailing close-brace from model answer capture
  merge_esc_cfg                 — merge per-type escalation overrides into global EscalationConfig
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..engine.result import SegmentSource

if TYPE_CHECKING:
    from ..config.escalation import RewardShapingConfig, TriFactorConfig
    from ..engine.context import RolloutContext
    from ..engine.result import RolloutResult

# Phase B natural-stop sequences (sentence boundaries + think close)
_NAT_STOPS: list[str] = [".\n", "?\n", "!\n", "\n\n", "</think>"]
_BOUNDARIES: frozenset[str] = frozenset({".\n", "?\n", "!\n", "\n\n"})


def _find_boundaries(t: str) -> list[int]:
    """Non-overlapping scan for sentence boundaries (.\n ?.\n !\n \n\n) in text.

    Returns char positions (AFTER each 2-char boundary pair). Using i += 2
    on match prevents double-counting overlapping pairs like '.\n\n' → only
    counts the '.\n' boundary, not the subsequent '\n\n' at i+1.
    """
    bpos: list[int] = []
    i = 0
    while i < len(t) - 1:
        if t[i : i + 2] in _BOUNDARIES:
            bpos.append(i + 2)
            i += 2
        else:
            i += 1
    return bpos


def _find_boundaries_line(t: str) -> list[int]:
    """Scan for line boundaries (\\n) — suited for structured step-format traces."""
    return [i + 1 for i, c in enumerate(t) if c == "\n"]


def _find_boundaries_regex(t: str, patterns: tuple[str, ...]) -> list[int]:
    """Find split positions using regex patterns, returns sorted unique positions.

    Each match's end() position is used as the split point, so the matched
    boundary is included in the preceding chunk.  Invalid patterns are silently
    skipped.  Example patterns: ["\\n\\n", "\\n", "\\. (?=[A-Z])"].
    """
    all_pos: list[int] = []
    for p in patterns:
        try:
            all_pos.extend(m.end() for m in re.finditer(p, t))
        except re.error:
            continue
    return sorted(set(all_pos))


def _get_bpos(t: str, boundary_mode: str, split_boundaries: tuple[str, ...] = ()) -> list[int]:
    """Dispatch boundary detection: regex patterns > boundary_mode fallback."""
    if split_boundaries:
        return _find_boundaries_regex(t, split_boundaries)
    return _find_boundaries_line(t) if boundary_mode == "line" else _find_boundaries(t)


# ── Scaffold planning ──────────────────────────────────────────────────────────


def _count_n_eff_max(
    text: str,
    boundary_mode: str = "sentence",
    split_boundaries: tuple[str, ...] = (),
) -> int:
    """Count maximum achievable n_eff from expert trace (no backend required).

    Uses the same boundary detection logic as _plan_adaptive_scaffold.
    Returns max(1, boundary_count) for non-empty traces, 0 for empty/whitespace.
    boundary_mode: 'sentence' (.\n ?.\n !\n \n\n) | 'line' (\n).
    split_boundaries: regex patterns; non-empty overrides boundary_mode.
    """
    t = text.lstrip()
    if t.startswith("<think>"):
        t = t[len("<think>") :]
    end = t.find("</think>")
    t = t[:end] if end != -1 else t
    if not t.strip():
        return 0
    bpos = _get_bpos(t, boundary_mode, split_boundaries)
    return max(1, len(bpos))


def _plan_adaptive_scaffold(
    text: str,
    c_k: int,
    backend: Any,
    max_tokens: int,
    boundary_mode: str = "sentence",
    split_boundaries: tuple[str, ...] = (),
    region_offset_chars: int = 0,
    region_length_chars: int = 0,
) -> tuple[list[str], int, int]:
    """Boundary-snapped scaffold planning from an expert trace.

    Args:
        text:             Raw sample.answer (may contain <think>...</think>).
        c_k:              Allocated chunk count for this member.
        backend:          GenerationBackend for encode/decode.
        max_tokens:       Token cap per returned segment.
        boundary_mode:    'sentence' (.\n ?.\n !\n \n\n) | 'line' (\n).
        split_boundaries: Regex patterns; non-empty overrides boundary_mode.
        region_offset_chars: Start position in thinking block (0 = from start).
        region_length_chars: Length to scaffold (0 = rest of block from offset).

    Returns:
        (segments, n_eff, b_eff)
        segments : list[str] — actual scaffold chunks (len <= c_k)
        n_eff    : int       — effective cycle count
        b_eff    : int       — ideal Phase A budget per cycle (tokens)
    """
    if c_k == 0:
        return [], 0, 0
    t = text.lstrip()
    if t.startswith("<think>"):
        t = t[len("<think>") :]
    end = t.find("</think>")
    t = t[:end] if end != -1 else t
    # Apply region window: slice thinking block to the assigned repeat region
    if region_offset_chars > 0 or region_length_chars > 0:
        start = min(region_offset_chars, len(t))
        end_pos = (start + region_length_chars) if region_length_chars > 0 else len(t)
        t = t[start:end_pos]
    # Boundary positions — regex patterns take priority over boundary_mode
    bpos = _get_bpos(t, boundary_mode, split_boundaries)
    toks = backend.encode(t)
    if not toks:
        return [], 0, 0
    # Token-based fallback: when boundaries < c_k, split token array into equal slices
    if len(bpos) < c_k:
        chunk_size = max(1, len(toks) // c_k)
        fb_segs: list[str] = []
        pos = 0
        for _ in range(c_k):
            chunk = toks[pos : pos + chunk_size][:max_tokens]
            if chunk:
                fb_segs.append(backend.decode(chunk))
            pos += chunk_size
            if pos >= len(toks):
                break
        actual = [s for s in fb_segs if s.strip()]
        n_eff = len(actual)
        return actual, n_eff, max(32, len(toks) // max(n_eff, 1))
    n_eff = min(c_k, max(1, len(bpos)))
    b_eff = max(32, len(toks) // n_eff)
    # Ideal split points; snap each to nearest preceding boundary (≤20 chars back)
    ideal = [int(len(t) * (i + 1) / n_eff) for i in range(n_eff - 1)]
    splits: list[int] = []
    for ip in ideal:
        best = next((b for b in sorted(bpos, reverse=True) if ip - 20 <= b <= ip), ip)
        splits.append(best)
    splits.append(len(t))
    prev, segs = 0, []
    for sp in splits:
        chunk = backend.encode(t[prev:sp])[:max_tokens]
        if chunk:
            segs.append(backend.decode(chunk))
        prev = sp
    return [s for s in segs if s.strip()], n_eff, b_eff


# ── Two-phase generation ───────────────────────────────────────────────────────


def _two_phase_generate(
    ctx: RolloutContext, b_eff: int, b_ext: int, temp: float, cycle_idx: int
) -> tuple[Any, bool]:
    """Two-phase per-cycle generation.

    Phase A: generate up to b_eff tokens; stop only on </think>.
    Phase B: if A didn't close </think>, extend up to b_ext with natural stops.

    Returns:
        (last_gen_result, closed_think: bool)
    """
    gen_a = ctx.generate(
        max_tokens=max(1, b_eff),
        stop=["</think>"],
        temperature=temp,
        tag=f"cycle_{cycle_idx}",
    )
    if gen_a.stop_reason == "stop_sequence" or "</think>" in gen_a.text:
        return gen_a, True
    if b_ext > 0:
        gen_b = ctx.generate(
            max_tokens=b_ext,
            stop=_NAT_STOPS,
            temperature=temp,
            tag=f"cycle_{cycle_idx}_ext",
        )
        closed = (
            gen_b.stop_reason == "stop_sequence" and gen_b.stop_sequence == "</think>"
        ) or "</think>" in gen_b.text
        return gen_b, closed
    return gen_a, False


# ── Tri-factor weight building ─────────────────────────────────────────────────


def _build_tri_factor_weights(
    result: RolloutResult, c_k: int, c_max: int, n_eff: int, tf_cfg: TriFactorConfig
) -> list[float]:
    """Per-token tri-factor importance weights.

    w(k, i, j) = α_h^(c_k/c_max) · α_v^(i/max(n_eff-1,1)) · α_p^j

    cycle_idx (i) increments on each INJECTED segment (= each scaffold injection),
    so both Phase A and Phase B of the same cycle share the same cycle_idx.

    Injected tokens: weight 1.0 (masked out; value irrelevant — length alignment only).
    Answer-tagged tokens: weight 1.0 (no downweighting for answer phase).
    c_k=0 (naked member): all 1.0 via early exit.

    Returns a flat list aligned with result.loss_mask (identical length).
    """
    non_prompt = [s for s in result.segments if s.source != SegmentSource.PROMPT]
    n_total = sum(len(s.tokens) for s in non_prompt)
    if c_k == 0:
        return [1.0] * n_total
    alpha_h = tf_cfg.alpha_h ** (c_k / c_max)
    weights: list[float] = []
    cycle_idx = -1  # becomes 0 on first scaffold injection
    causal_dist = 0
    for seg in non_prompt:
        if seg.source == SegmentSource.INJECTED:
            weights.extend([1.0] * len(seg.tokens))  # align; masked out anyway
            causal_dist = 0
            cycle_idx += 1  # each scaffold injection = new cycle
            continue
        if seg.tag == "answer":
            weights.extend([1.0] * len(seg.tokens))  # answer phase: full weight
            continue
        ci = max(cycle_idx, 0)
        for pos in range(len(seg.tokens)):
            weights.append(
                alpha_h
                * tf_cfg.alpha_v ** (ci / max(n_eff - 1, 1))
                * tf_cfg.alpha_p ** (causal_dist + pos)
            )
        causal_dist += len(seg.tokens)
    return weights


# ── Group-level reward shaping ─────────────────────────────────────────────────


def _apply_group_shaping(
    group_rollouts: list[RolloutResult],
    rs_cfg: RewardShapingConfig,
    c_max: int,
    iteration: int,
    total_iters: int,
) -> None:
    """Apply bidirectional group-level reward shaping to all members (in-place).

    Must be called AFTER all members (including naked) have finished rolling out,
    so any_correct can be computed across the full group.

    λ_s = lambda_s_max * progress   (0 → max, INCREASES: reward independence late)
    λ_f = lambda_f_max * (1-progress) (max → 0, DECREASES: structure nudge fades)

    any_correct gate: rw >= rs_cfg.any_correct_threshold (default 0.5)
    Success path (any_correct=True):
        v1 (default): shaped = raw * (1 - λ_s * c_k / c_max)
        v2 (reward_shaping_v2=True): shaped = raw + λ_s * (1-raw) * (1-c_k/c_max)
          — v2 never penalises correct answers; boosts correct with less scaffold.
    Failure path (any_correct=False):
        shaped = min(λ_f * c_k / c_max, rs_cfg.failure_floor) (default 0.0)
    Naked view: shaped = raw (no shaping).
    """
    raw_rewards = [r.metadata["_esc_reward"] for r in group_rollouts]
    thr = rs_cfg.any_correct_threshold
    any_correct = any(rw >= thr for rw in raw_rewards)
    progress = min(1.0, iteration / max(total_iters, 1))
    lambda_s = rs_cfg.lambda_s_max * progress
    lambda_f = rs_cfg.lambda_f_max * (1.0 - progress)
    cm = max(c_max, 1)
    for r in group_rollouts:
        c_k = r.metadata["_esc_allocated_chunks"]
        raw = r.metadata["_esc_reward"]
        if r.metadata.get("_view_is_naked"):
            shaped = raw
        elif r.metadata.get("_esc_forced_answer"):
            # Forced-answer member: GT was injected → reward was set to 1.0 explicitly.
            # Group shaping must not discount it — it's a guaranteed ICL signal.
            shaped = raw
        elif any_correct:
            if rs_cfg.reward_shaping_v2:
                # v2: boost correct toward 1.0; less scaffold → bigger boost.
                # raw=0 (wrong in success group) → stays 0 (gate on raw > 0).
                # raw=1.0 → 1-raw=0 → shaped=1.0 (no over-inflation).
                # c_k=cm → 1-c_k/cm=0 → shaped=raw (no change at max scaffold).
                shaped = raw + lambda_s * (1.0 - raw) * (1.0 - c_k / cm) if raw > 0.0 else 0.0
            else:
                # v1 legacy: reduce correct proportional to scaffold used
                shaped = raw * (1.0 - lambda_s * c_k / cm)
        else:
            shaped = min(lambda_f * c_k / cm, rs_cfg.failure_floor)
        r.metadata["_esc_shaped_reward"] = shaped

    # prefer_scaffolded: when naked also passed (raw>=1.0), add a bonus to scaffolded
    # shaped rewards so structured members win the advantage tiebreaker.
    # Applied AFTER λ shaping so the bonus is clean and not discounted by λ_s.
    if rs_cfg.prefer_scaffolded and rs_cfg.prefer_scaffolded_bonus > 0:
        naked_passed = any(
            r.metadata.get("_view_is_naked") and r.metadata["_esc_reward"] >= 1.0
            for r in group_rollouts
        )
        if naked_passed:
            for r in group_rollouts:
                if not r.metadata.get("_view_is_naked") and r.metadata["_esc_reward"] >= 1.0:
                    r.metadata["_esc_shaped_reward"] += rs_cfg.prefer_scaffolded_bonus


# ── Jaccard anchor ────────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "they",
        "have",
        "been",
        "will",
        "what",
        "when",
        "where",
        "which",
        "there",
        "their",
        "about",
        "would",
        "could",
        "should",
        "does",
        "then",
        "than",
        "more",
        "also",
        "into",
        "some",
        "other",
        "such",
        "each",
        "these",
        "those",
        "were",
        "very",
        "just",
        "like",
    }
)


def _jaccard_anchor_from_naked(naked_text: str, expert_text: str, max_words: int = 80) -> str:
    """Build a targeted anchor from the intersection of naked output and expert trace.

    Finds content words shared between the model's naked reasoning and the expert CoT,
    then extracts the expert sentences densest in those terms. This surfaces the reasoning
    branches the model was already approaching, making the anchor more targeted than a
    generic summary. Returns "" when there is no meaningful word overlap.
    """

    def _words(text: str) -> set[str]:
        return {
            w.lower().strip(".,;:!?()[]{}\"'<>")
            for w in text.split()
            if len(w) > 3 and w.lower().strip(".,;:!?") not in _STOPWORDS
        }

    common = _words(naked_text) & _words(expert_text)
    if not common:
        return ""
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", expert_text) if len(s.strip()) > 10
    ]
    scored = sorted(
        [(sum(1 for w in common if w in s.lower()), s) for s in sentences],
        key=lambda x: -x[0],
    )
    parts: list[str] = []
    total = 0
    for score, sent in scored:
        if score == 0:
            break
        n = len(sent.split())
        if total + n > max_words:
            break
        parts.append(sent)
        total += n
    return " ".join(parts)


# ── Answer extraction ──────────────────────────────────────────────────────────


def _anchor_text_for_scaffold(prompt: str, target_text: str, answer: str) -> str:
    """Get reason_anchor compressed skeleton as scaffold source.

    Strips <think> tags from target_text, calls generate_anchor, returns the
    compact logical skeleton. Falls back to raw target_text on any failure.
    """
    try:
        from reason_anchor import generate_anchor  # type: ignore[import]  # noqa: PLC0415

        t = target_text.lstrip()
        if t.startswith("<think>"):
            t = t[len("<think>") :]
        end = t.find("</think>")
        t = (t[:end] if end != -1 else t).strip()
        result = generate_anchor(prompt or "", t, answer or "")
        return result.anchor if result.anchor else target_text
    except Exception:
        return target_text


def _answer_prefix_text(target_text: str, n_tokens: int, backend: Any) -> str:
    """Extract first n_tokens from the ground-truth answer for prefix scaffolding.

    Prefers content inside \\boxed{...}; falls back to text after </think>.
    Returns "" when n_tokens == 0 or no answer content found.
    """
    if n_tokens <= 0 or not target_text.strip():
        return ""
    import re

    m = re.search(r"\\boxed\{([^}]+)\}", target_text)
    content = m.group(1) if m else re.sub(r"<think>.*?</think>", "", target_text, flags=re.DOTALL)
    content = content.strip()
    if not content:
        return ""
    toks = backend.encode(content)
    return backend.decode(toks[:n_tokens]) if toks else ""


def _strip_answer_close(text: str) -> str:
    """Strip trailing close-brace from answer segment if model closed \\boxed{} itself.

    Model sometimes generates the closing '}' inside the answer context, producing
    text like "C}" or "8}\\nWait...". Split on first '}' to recover the clean value.
    """
    return text.split("}")[0].strip()


def merge_esc_cfg(base: Any, overrides: dict[str, Any] | None) -> Any:
    """Return a new EscalationConfig with per-type overrides applied (in-place merge).

    Only specified sub-fields change; everything else inherits from *base*.
    Unknown keys are silently ignored (forward-compatible).

    YAML example::

        type_overrides:
          math:
            escalation:
              horizontal:
                forced_answer_on_last_member: true
                forced_answer_inject_chars: 10
              scaffold:
                max_tokens: 256
    """
    if not overrides or not isinstance(overrides, dict):
        return base
    from dataclasses import replace as _dr  # noqa: PLC0415

    from ..config.escalation import (  # noqa: PLC0415
        EscalationScaffoldConfig,
        EscalationTrackingConfig,
        HorizontalConfig,
        RewardShapingConfig,
        TriFactorConfig,
        VerticalConfig,
    )

    _sub_cls: dict[str, Any] = {
        "vertical": VerticalConfig,
        "horizontal": HorizontalConfig,
        "scaffold": EscalationScaffoldConfig,
        "tracking": EscalationTrackingConfig,
        "reward_shaping": RewardShapingConfig,
        "tri_factor": TriFactorConfig,
    }
    kwargs: dict[str, Any] = {}
    top_flds = base.__dataclass_fields__
    for k, v in overrides.items():
        if k in _sub_cls and isinstance(v, dict):
            cls = _sub_cls[k]
            valid = {f: vv for f, vv in v.items() if f in cls.__dataclass_fields__}
            if "split_boundaries" in valid and isinstance(valid["split_boundaries"], list):
                valid["split_boundaries"] = tuple(valid["split_boundaries"])
            if valid:
                kwargs[k] = _dr(getattr(base, k), **valid)
        elif k in top_flds and k not in _sub_cls:
            kwargs[k] = v
    return _dr(base, **kwargs) if kwargs else base


def _extract_forced_answer(ground_truth: str) -> str:
    """Extract clean answer text suitable for forced injection into \\boxed{}.

    Strips \\boxed{} wrapping and <think>...</think> blocks.
    Returns the raw ground_truth string if neither is present.
    """
    if not ground_truth:
        return ""
    m = re.search(r"\\boxed\{([^}]+)\}", ground_truth)
    if m:
        return m.group(1).strip()
    clean = re.sub(r"<think>.*?</think>", "", ground_truth, flags=re.DOTALL).strip()
    return clean or ground_truth.strip()
