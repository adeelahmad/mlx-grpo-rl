"""Metal GPU time-budget calculator for macOS training.

macOS enforces an ~8-second watchdog on Metal command buffers.  A single
nn.value_and_grad call that produces a command buffer exceeding this limit
triggers kIOGPUCommandBufferCallbackErrorImpactingInteractivity and aborts
the process (exit code 134).

Gradient computation rate depends on model size, LoRA rank, and sequence
length.  Empirical reference point (Qwen-4B, rank-48, M4 Max):
    generation (autoregressive): ~11 ms/token  →  580t ≈ 6.4 s
    gradient (batch forward+backward): roughly 0.6–1.0× generation rate

The safe ceiling is ~618 tokens at the gradient step to stay under 8 s with
a 15% safety margin.  The extension path in mlx_ops.compute_grads can push
sequences to ``max_grad_seq_len + 96`` when scaffold fills the normal budget,
so max_grad_seq_len=512 → 608 tok × 11ms = 6.7s (safe).

Usage::

    from mlx_grpo.config.metal_budget import check_metal_budget
    check_metal_budget(config)   # called automatically by load_config
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Config  # noqa: F401 (type-check only)

# Empirically measured on Qwen-4B, rank-48, M4 Max
_MS_PER_TOKEN_GEN: float = 11.0  # ms, autoregressive generation
# Gradient (forward+backward) rate matches generation — batch-efficiency ratio of
# 0.8× does NOT hold for LoRA training on MLX.  Measured: ~11 ms/tok actual.
_MS_PER_TOKEN_GRAD: float = _MS_PER_TOKEN_GEN  # 11.0 ms/tok
_WATCHDOG_MS: float = 8_000.0  # macOS Metal hard limit
_SAFETY_FACTOR: float = 0.85  # 15 % headroom below watchdog
_SAFE_MS: float = _WATCHDOG_MS * _SAFETY_FACTOR  # 6 800 ms

# Max tokens that can safely fit in one Metal command buffer at gradient time
SAFE_GRAD_TOKENS: int = int(_SAFE_MS / _MS_PER_TOKEN_GRAD)  # ~618


def _max_rollout_tokens(type_overrides: dict[str, Any], gen_max: int) -> int:
    """Largest rollout_max_tokens across all type overrides and generation config."""
    best = gen_max
    for cfg in type_overrides.values():
        best = max(best, cfg.get("rollout_max_tokens", gen_max))
    return best


def _max_thinking_budget(type_overrides: dict[str, Any]) -> int:
    """Largest thinking_budget across all type overrides."""
    return max((cfg.get("thinking_budget", 0) for cfg in type_overrides.values()), default=0)


def check_metal_budget(config: Any) -> None:  # noqa: C901
    """Compute and print Metal time-budget summary; warn on unsafe configs.

    Called automatically by load_config().  Prints nothing when all values
    are safe so that normal training output is uncluttered.

    Checks
    ------
    1. ``max_grad_seq_len`` vs SAFE_GRAD_TOKENS.
    2. Effective grad length = min(max_grad_seq_len, used_tokens + est_prompt)
       plus the +256 extension that mlx_ops can add when scaffold fills budget.
    3. Warns when effective grad length > SAFE_GRAD_TOKENS.
    """
    tr = config.training
    gen = config.generation
    overrides: dict[str, Any] = config.type_overrides or {}

    msl: int = getattr(tr, "max_grad_seq_len", 1024)
    used: int | None = getattr(gen, "used_tokens", None)
    gen_max: int = getattr(gen, "max_tokens", 64)

    max_rollout = _max_rollout_tokens(overrides, gen_max)
    max_think = _max_thinking_budget(overrides)
    # Completion budget after used_tokens truncation (if set)
    comp_budget = min(used, max_rollout) if used is not None else max_rollout

    # Prompt length estimate: configurable via metal_est_prompt_tokens; default 900
    # covers Qwen-Thinking chat template. Set to ~150-212 for exam/MCQ types.
    est_prompt: int = getattr(tr, "metal_est_prompt_tokens", 900)
    # grad_seq = what actually enters compute_grads after _truncate_completions
    grad_seq = min(msl, est_prompt + comp_budget)
    # Extension path in mlx_ops: when scaffold fills budget, adds up to +96 tokens
    # (first_train≤64 + keep=32). Cap matches mlx_ops so estimates are accurate.
    grad_seq_ext = min(grad_seq + 96, msl * 2)

    warnings: list[str] = []

    if msl > SAFE_GRAD_TOKENS:
        warnings.append(
            f"  max_grad_seq_len={msl} > safe ceiling {SAFE_GRAD_TOKENS} tok "
            f"({msl * _MS_PER_TOKEN_GRAD:.0f} ms est, watchdog={_WATCHDOG_MS:.0f} ms)"
        )

    if grad_seq_ext > SAFE_GRAD_TOKENS:
        warnings.append(
            f"  effective grad seq (w/ scaffold ext) ≈ {grad_seq_ext} tok "
            f"({grad_seq_ext * _MS_PER_TOKEN_GRAD:.0f} ms est) — may exceed watchdog"
        )

    if max_think > 1024:
        warnings.append(
            f"  thinking_budget={max_think} → rollouts up to "
            f"{max_think + 50} tok completion; "
            f"reduce to ≤1024 for Metal safety"
        )

    # Always show the budget table so the user can verify
    safe_msl = max(est_prompt + 32, min(msl, SAFE_GRAD_TOKENS))
    print(
        f"[metal_budget] grad_seq≈{grad_seq_ext} tok | "
        f"safe_ceiling={SAFE_GRAD_TOKENS} tok | "
        f"watchdog={_WATCHDOG_MS / 1000:.1f}s | "
        f"est_grad_time≈{grad_seq_ext * _MS_PER_TOKEN_GRAD / 1000:.2f}s"
    )
    if warnings:
        print("[metal_budget] ⚠  WARNINGS — risk of Metal GPU watchdog (exit 134):")
        for w in warnings:
            print(f"[metal_budget]{w}")
        print(
            f"[metal_budget]    Recommended: max_grad_seq_len≤{safe_msl}, "
            f"thinking_budget≤1024, rollout_max_tokens≤1200"
        )


# Empirically measured batch-prefill rate during *generation* (not gradient).
# Prefill is parallel but bounded by KV write bandwidth; ~5 ms/token on M-series.
_MS_PER_TOKEN_PREFILL: float = 5.0  # ms, batch-prefill during autoregressive gen


def _worst_case_escalation_prefill(n_chunks: int, eff_b: int, b_ext: int, sc_toks: int) -> int:
    """Worst-case non-prompt prefill tokens for one escalation ctx.generate() call.

    b_ext > 0: Phase B of last cycle is worst → (n_chunks-1)×(sc+eff_b+b_ext)+sc+eff_b
    b_ext == 0: Phase A of last cycle is worst → (n_chunks-1)×(sc+eff_b)+sc
    """
    if b_ext > 0:
        return (n_chunks - 1) * (sc_toks + eff_b + b_ext) + sc_toks + eff_b
    return (n_chunks - 1) * (sc_toks + eff_b) + sc_toks


def check_escalation_metal_budget(config: Any) -> None:
    """Warn when escalation config risks Metal watchdog during generation prefill.

    Each ctx.generate() only caches the initial prompt; later tokens need full
    batch-prefill on every call.  Worst case is Phase B of the last vertical cycle.
    """
    esc = getattr(config, "escalation", None)
    if esc is None or not esc.enabled:
        return

    n_chunks: int = esc.vertical.n_chunks
    b_ext: int = esc.tri_factor.b_ext
    sc_toks: int = esc.scaffold.max_tokens
    overrides: dict[str, Any] = config.type_overrides or {}
    safe_prefill = int(_WATCHDOG_MS * _SAFETY_FACTOR / _MS_PER_TOKEN_PREFILL)  # ~1360

    for type_name, tcfg in overrides.items():
        tb: int = tcfg.get("thinking_budget", 512)
        eff_b = max(32, tb // max(n_chunks, 1))
        worst = _worst_case_escalation_prefill(n_chunks, eff_b, b_ext, sc_toks)
        est_ms = worst * _MS_PER_TOKEN_PREFILL
        if worst > safe_prefill:
            phase_note = f"b_ext={b_ext} (Phase B ON)" if b_ext > 0 else "b_ext=0 (Phase B off)"
            print(
                f"[metal_budget] ⚠  ESCALATION PREFILL — {type_name}: "
                f"worst-case prefill={worst} tok ≈{est_ms:.0f} ms "
                f"({phase_note}, safe≤{safe_prefill} tok / "
                f"{safe_prefill * _MS_PER_TOKEN_PREFILL:.0f} ms)"
            )
            if b_ext > 0:
                safe_tb = (safe_prefill - sc_toks) // max(n_chunks, 1) - b_ext
                print(
                    f"[metal_budget]    → Set escalation.tri_factor.b_ext: 0  "
                    f"or reduce thinking_budget≤{max(0, safe_tb * n_chunks)}"
                )
            else:
                safe_tb = (safe_prefill - sc_toks) // max(n_chunks - 1, 1) - sc_toks
                print(f"[metal_budget]    → Reduce thinking_budget≤{max(0, safe_tb * n_chunks)}")


def recommended_max_grad_seq_len(est_prompt_tokens: int = 900) -> int:
    """Return recommended max_grad_seq_len for the given prompt length estimate.

    Leaves 96-token headroom for the prefix-scaffold extension path in mlx_ops
    (first_train=64 + keep=32), so that max_grad_seq_len + 96 ≤ SAFE_GRAD_TOKENS.

    Returns max(est_prompt_tokens + 32, usable) where usable = SAFE_GRAD_TOKENS - 96.
    For short prompts the safe ceiling dominates; for long prompts (≥ usable − 32)
    the returned value exceeds the safe ceiling — those sequences are skipped
    by the truncation logic in compute_grads anyway.

    Example::

        # MCQ: short prompts ~100 tok
        safe = recommended_max_grad_seq_len(100)  # → 522  →  use 512
        # Qwen-4B math: long prompts ~900 tok → all exceed max_grad_seq_len, skipped
        # Set in config: max_grad_seq_len: 512
    """
    extension_reserve = 96  # first_train=64 + keep=32 in mlx_ops extension path
    usable = int(_SAFE_MS / _MS_PER_TOKEN_GRAD) - extension_reserve
    return max(est_prompt_tokens + 32, usable)
