"""MLX gradient operations — the production training hot path.

The critical constraint: MLX's value_and_grad must trace through the
ENTIRE computation (forward pass → loss) to compute gradients. You
cannot differentiate a pre-computed scalar.

Correct pattern:
    def loss_fn(model):
        logits = model(tokens)          # Forward pass (traced)
        loss = compute_loss(logits)     # Loss from logits (traced)
        return loss                     # MLX differentiates this

    loss, grads = nn.value_and_grad(model, loss_fn)(model)

This module provides that exact pipeline for GRPO training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass(slots=True)
class GradStepResult:
    """Result from a complete gradient step."""

    loss: float = 0.0
    policy_loss: float = 0.0
    kl_penalty: float = 0.0
    kl_divergence: float = 0.0
    grad_norm: float = 0.0
    effective_tokens: int = 0
    clip_frac_low: float = 0.0
    clip_frac_high: float = 0.0


def compute_log_probs_for_tokens(
    model: Any,
    token_ids: list[int],
) -> Any:
    """Forward pass → per-token log probs for a sequence.

    Uses nn.losses.cross_entropy with reduction='none' for efficiency.
    This fuses log_softmax + gather into a single kernel, avoiding a
    full [T, vocab] intermediate tensor that causes Metal GPU timeouts.

    Args:
        model: MLX model — model(input_ids) → logits [1, T, vocab]
        token_ids: Full token sequence [prompt + completion]

    Returns:
        mx.array of shape [T-1] — log P(token_{t+1} | tokens_{0..t})
    """
    # stop_gradient on token arrays: they are integer indices (not trainable).
    # Without this, MLX 0.30+ may attempt GatherQMM::vjp wrt the indices on
    # models with tied/quantized embeddings (QuantizedEmbedding), which fails.
    tokens = mx.stop_gradient(mx.array([token_ids]))
    logits = model(tokens)  # [1, T, vocab]

    # Use fused cross-entropy: avoids materializing full [T, vocab] log_probs
    n_t = len(token_ids) - 1
    target_ids = mx.stop_gradient(mx.array(token_ids[1:]))  # [n_t]
    shift_logits = logits[0, :n_t]  # [n_t, vocab]

    # cross_entropy with reduction='none' returns NLL per token
    nll = nn.losses.cross_entropy(shift_logits, target_ids, reduction="none")
    return -nll  # [T] — log probs


def precompute_ref_logps(
    ref_model: Any,
    sequences: list[list[int]],
) -> list[Any]:
    """Pre-compute reference model log probs OUTSIDE the gradient scope.

    This MUST be called before nn.value_and_grad. The results are
    materialized with mx.eval() and wrapped in mx.stop_gradient()
    so they are treated as constants during backprop.

    Args:
        ref_model: Frozen reference model
        sequences: Full token sequences [prompt + completion] per rollout

    Returns:
        List of mx.array, one per sequence, each of shape [T-1]
    """
    eval_batch = 1  # Eval each ref pass individually to prevent Metal GPU timeout
    ref_logps = []
    pending: list[int] = []  # Indices of arrays needing eval

    for i, seq in enumerate(sequences):
        if len(seq) < 2:
            ref_logps.append(mx.array([0.0]))
            continue
        lp = compute_log_probs_for_tokens(ref_model, seq)
        ref_logps.append(lp)
        pending.append(i)

        # Batch eval to keep Metal command buffers manageable
        if len(pending) >= eval_batch:
            mx.eval(*[ref_logps[j] for j in pending])
            pending.clear()
            mx.clear_cache()

    # Eval remaining
    if pending:
        mx.eval(*[ref_logps[j] for j in pending])
        mx.clear_cache()

    ref_logps = [mx.stop_gradient(lp) for lp in ref_logps]
    return ref_logps


def grpo_loss_fn(
    model: Any,
    ref_logps: list[Any],
    sequences: list[list[int]],
    prompt_lengths: list[int],
    masks: list[list[bool]],
    advantages: list[float],
    beta: float = 0.04,
    segment_weights: list[list[float]] | None = None,
    epsilon: float = 0.0,
    epsilon_high: float | None = None,
    _clip_stats: list | None = None,
    _kl_out: list | None = None,
) -> mx.array:
    """Differentiable GRPO loss — the function that gets differentiated.

    This runs INSIDE nn.value_and_grad, so the forward pass through
    `model` is traced and gradients flow back through it.

    Args:
        model: Trainable model (current policy, has LoRA)
        ref_logps: Pre-computed reference log probs (from precompute_ref_logps)
        sequences: Full token sequences [prompt + completion] per rollout
        prompt_lengths: Length of prompt prefix per rollout (skip these)
        masks: Per-completion-token loss masks [True = train]
        advantages: Per-rollout GRPO advantages
        beta: KL penalty weight
        segment_weights: Optional per-token gradient weights for each rollout.
        epsilon: PPO-style lower clipping bound. 0 = no clipping.
        epsilon_high: Upper clipping bound (DAPO-style asymmetric). None = use epsilon.
        _clip_stats: Mutable list for collecting clip fractions outside grad scope.
            Caller passes in an empty list; this function appends
            [clip_low_count, clip_high_count, total_mask_count].
        _kl_out: Mutable list for collecting mean KL outside grad scope.
            Caller passes in an empty list; this function appends [mean_kl_float].

    Returns:
        Scalar loss (mx.array) — MLX will differentiate this w.r.t. model
    """
    use_clipping = epsilon > 0
    eps_high = epsilon_high if epsilon_high is not None else epsilon

    total_policy = mx.array(0.0)
    total_kl = mx.array(0.0)
    total_tokens = mx.array(0.0)
    clip_low_acc = mx.array(0.0)
    clip_high_acc = mx.array(0.0)
    total_mask_acc = mx.array(0.0)

    n_active_seqs = 0  # Python counter for zero-check (avoids .item() in traced scope)

    # Build iterator with or without segment weights
    if segment_weights is not None:
        items = zip(
            sequences, prompt_lengths, masks, advantages, ref_logps, segment_weights, strict=True
        )
    else:
        items = zip(
            sequences,
            prompt_lengths,
            masks,
            advantages,
            ref_logps,
            [None] * len(sequences),
            strict=True,
        )

    for seq, plen, mask, adv, ref_lp, seg_w in items:
        if not seq or plen >= len(seq):
            continue

        # Forward pass through current model (TRACED by autograd)
        current_lp = compute_log_probs_for_tokens(model, seq)

        # Only look at completion tokens (skip prompt prefix).
        # Log probs are shifted: lp[i] = log P(token_{i+1} | tokens_{0..i}).
        # The first completion token (at position plen) has its log prob
        # at index plen-1, so we slice from max(plen-1, 0).
        start = max(plen - 1, 0)
        comp_current = current_lp[start:]
        comp_ref = ref_lp[start:]

        # Apply mask
        n_t = min(len(mask), comp_current.shape[0], comp_ref.shape[0])
        if n_t == 0:
            continue

        # Check mask has active tokens using Python (avoids .item() in traced scope)
        if not any(mask[:n_t]):
            continue

        n_active_seqs += 1
        # binary_mask: 1.0 where token is trainable, 0.0 elsewhere.
        # Used for n_active normalization so that IS weights embedded in mask_arr
        # do not cancel with the denominator (length scale would cancel otherwise).
        binary_mask = mx.array(mask[:n_t], dtype=mx.float32)

        # Apply segment weights (IS positional decay or segment-level weights)
        if seg_w is not None:
            weight_arr = mx.array(seg_w[:n_t], dtype=mx.float32)
            mask_arr = binary_mask * weight_arr
        else:
            mask_arr = binary_mask

        n_active = mx.sum(binary_mask)  # Binary count — IS weights must not cancel

        # Log importance ratio: log(pi / pi_ref)
        log_ratio = comp_current[:n_t] - comp_ref[:n_t]
        ratio = mx.exp(log_ratio)

        if use_clipping:
            # PPO-style clipped objective: -min(r*A, clip(r)*A)
            clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + eps_high)
            unclipped_obj = ratio * adv
            clipped_obj = clipped_ratio * adv
            pg_per_token = -mx.minimum(unclipped_obj, clipped_obj)
            pg_term = mx.sum(pg_per_token * mask_arr)

            # Track clipping — binary_mask already defined above
            is_low = (ratio < (1.0 - epsilon)).astype(mx.float32) * binary_mask
            is_high = (ratio > (1.0 + eps_high)).astype(mx.float32) * binary_mask
            if adv < 0:
                clip_low_acc = clip_low_acc + mx.sum(is_low)
            if adv > 0:
                clip_high_acc = clip_high_acc + mx.sum(is_high)
            total_mask_acc = total_mask_acc + mx.sum(binary_mask)
        else:
            # Standard GRPO: -advantage * log_ratio * mask
            pg_term = -adv * mx.sum(log_ratio * mask_arr)

        total_policy = total_policy + pg_term

        # KL divergence (Schulman k3 estimator): r - log(r) - 1, always >= 0
        # where r = pi_ref / pi = exp(ref - policy)
        log_ratio_ref = comp_ref[:n_t] - comp_current[:n_t]
        ratio_ref = mx.exp(log_ratio_ref)
        kl_per_token = ratio_ref - log_ratio_ref - 1.0
        total_kl = total_kl + mx.sum(kl_per_token * mask_arr)

        total_tokens = total_tokens + n_active

    # Report clip stats as raw mx.arrays — caller does .item() OUTSIDE v&g scope
    if _clip_stats is not None:
        _clip_stats.extend([clip_low_acc, clip_high_acc, total_mask_acc])

    # Zero-check via Python counter (no .item() inside traced scope)
    if n_active_seqs == 0:
        if _kl_out is not None:
            _kl_out.append(mx.array(0.0))
        return mx.array(0.0)

    # Normalize by token count (total_tokens > 0 guaranteed when n_active_seqs > 0)
    loss = total_policy / total_tokens + beta * total_kl / total_tokens

    # Report mean KL as raw mx.array — caller does .item() OUTSIDE v&g scope
    if _kl_out is not None:
        _kl_out.append(total_kl / total_tokens)

    return loss


def sft_loss_fn(model: Any, token_ids: list[int], prompt_length: int) -> mx.array:
    """Standard cross-entropy loss on target tokens (SFT anchor).

    Uses fused cross-entropy to avoid materializing a full [T, vocab] tensor.

    Args:
        model: Trainable model
        token_ids: Full sequence [prompt + gold answer]
        prompt_length: Number of prompt tokens to skip in loss
    """
    tokens = mx.array([token_ids])
    logits = model(tokens)  # [1, n_t, vocab]

    n_t = len(token_ids) - 1
    targets = mx.array(token_ids[1:])  # [n_t]
    shift_logits = logits[0, :n_t]  # [n_t, vocab]

    # Fused cross-entropy: avoids full [T, vocab] log_probs materialization
    nll = nn.losses.cross_entropy(shift_logits, targets, reduction="none")

    # Only compute loss on answer tokens (after prompt).
    # Loss is shifted: nll[i] = -log P(token_{i+1} | tokens_{0..i}).
    # First answer token (at position prompt_length) has its loss at
    # index prompt_length-1.
    start = max(prompt_length - 1, 0)
    if start >= n_t:
        return mx.array(0.0)

    answer_nll = nll[start:]
    return mx.mean(answer_nll)


def sft_gradient_step(
    model: Any,
    optimizer: Any,
    token_ids: list[int],
    prompt_length: int,
    lr_multiplier: float = 0.1,
) -> float:
    """SFT anchor gradient step with reduced learning rate."""

    def loss_fn(m):
        return sft_loss_fn(m, token_ids, prompt_length)

    loss_val, grads = nn.value_and_grad(model, loss_fn)(model)

    if lr_multiplier != 1.0:
        grads = _tree_scale(grads, lr_multiplier)

    apply_grads(model, optimizer, grads)
    return float(loss_val)


def compute_grads(
    model: Any,
    ref_model: Any,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    masks: list[list[bool]],
    advantages: list[float],
    beta: float | list[float] = 0.04,
    max_grad_norm: float = 1.0,
    segment_weights: list[list[float] | None] | None = None,
    epsilon: float | list[float] = 0.0,
    epsilon_high: float | list[float] | None = None,
    loss_type: str = "grpo",
    max_seq_len: int = 1024,
    gpu_cooldown: float = 0.0,
    per_seq_grad_norm: list[float] | None = None,
    aux_loss_fn: Callable[[Any], mx.array] | None = None,
    verbose: bool = False,
) -> tuple[float, Any, float, float, float, float]:
    """Compute gradients without applying optimizer update.

    Processes each sequence with its own value_and_grad call and
    accumulates gradients. mx.eval() between sequences keeps each
    Metal command buffer small, preventing macOS GPU timeouts.

    All loss types (GRPO, BNPO, DR-GRPO) pre-compute reference log probs
    OUTSIDE the v&g scope, then use them as constants inside. This avoids
    mx.eval()/.item() inside the traced scope and prevents ref model
    LoRA scale mutation from affecting the gradient graph.

    Args:
        max_seq_len: Skip sequences longer than this in the gradient step.
            Prevents macOS Metal GPU timeout on long sequences. Default 400.
        gpu_cooldown: Seconds to sleep between per-sequence v&g calls.
            Prevents GPU thermal throttling on sustained compute. Default 0.

    Returns:
        (loss, grads, grad_norm, clip_frac_low, clip_frac_high, kl_divergence)
    """
    mx.clear_cache()
    seg_list = list(segment_weights) if segment_weights is not None else [None] * len(sequences)
    accum_grads = None
    accum_loss = 0.0
    accum_kl = 0.0
    n_valid = 0
    n_skipped = 0
    total_clip_low = 0
    total_clip_high = 0
    total_clip_mask = 0

    # ── Truncate sequences that exceed max_seq_len ──
    # Strategy depends on scaffold position (inferred from loss mask):
    #
    # Prefix scaffold  [masked...][generated...]  →  trim scaffold tail first.
    #   Scaffold tokens are masked (zero loss contribution) so trimming them
    #   costs nothing. This preserves ALL generated tokens, including the
    #   far-from-scaffold ones that carry the highest IS weight.
    #
    # Postfix/naked/other  →  tail truncation (current behaviour).
    #   For postfix scaffold the generated tokens are at the front, so tail
    #   truncation already removes the low-value scaffold region.
    for i in range(len(sequences)):
        if len(sequences[i]) <= max_seq_len:
            continue
        plen_i = prompt_lengths[i]
        budget = max_seq_len - plen_i
        if budget <= 0:
            n_skipped += 1
            sequences[i] = []
            continue

        comp_mask = masks[i]
        # Detect prefix scaffold: leading block of masked (False) tokens
        first_train = next((j for j, m in enumerate(comp_mask) if m), len(comp_mask))
        n_train = sum(1 for m in comp_mask if m)

        if first_train > 0 and n_train <= budget:
            # Prefix scaffold detected and all generated tokens fit in budget.
            # Trim scaffold from its end to free space; keep every generated token.
            scaffold_keep = budget - n_train
            trim = first_train - scaffold_keep  # scaffold tokens removed
            if trim > 0:
                sequences[i] = (
                    sequences[i][: plen_i + scaffold_keep] + sequences[i][plen_i + first_train :]
                )
                masks[i] = comp_mask[:scaffold_keep] + comp_mask[first_train:]
                if seg_list[i] is not None:
                    seg_list[i] = seg_list[i][:scaffold_keep] + seg_list[i][first_train:]
                if verbose:
                    print(
                        f"  [grad] seq {i}: scaffold-first truncation "
                        f"(removed {trim} scaffold tok, kept all {n_train} generated)"
                    )
            continue

        if n_train == 0:
            n_skipped += 1
            sequences[i] = []
            continue

        # Fallback: tail truncation (postfix scaffold, naked, or generated > budget)
        if not any(comp_mask[:budget]):
            # Scaffold fills entire budget — extend by at most +96 tok to keep some signal.
            # Cap is max_seq_len+96 (first_train≤64 + keep=32) so that
            # max_seq_len=512 → max total = 608 tok × 11ms = 6.7s < 8s watchdog.
            keep = min(32, n_train)
            eff = min(plen_i + first_train + keep, max_seq_len + 96, len(sequences[i]))
            budget = eff - plen_i
            if budget <= 0 or not any(comp_mask[:budget]):
                n_skipped += 1
                sequences[i] = []
                continue
            if verbose:
                print(
                    f"  [grad] seq {i}: extended {max_seq_len}→{eff} "
                    f"to keep trainable tokens (scaffold={first_train} toks)"
                )
        sequences[i] = sequences[i][: plen_i + budget]
        masks[i] = comp_mask[:budget]
        if seg_list[i] is not None:
            seg_list[i] = seg_list[i][:budget]

    # ── Phase 1: Pre-compute ALL reference log probs OUTSIDE v&g scope ──
    # Critical: ref logps MUST be computed outside nn.value_and_grad to avoid:
    #   (a) mx.eval() inside traced scope (breaks gradient graph)
    #   (b) ref model LoRA scale mutation inside traced scope
    #   (c) doubled GPU work (ref + policy forward in same Metal command buffer)
    ref_logps: list[Any] = [None] * len(sequences)
    for i, seq in enumerate(sequences):
        plen = prompt_lengths[i]
        mask = masks[i]
        if not seq or plen >= len(seq) or not any(mask):
            continue
        ref_lp = compute_log_probs_for_tokens(ref_model, seq)
        mx.eval(ref_lp)
        ref_logps[i] = mx.stop_gradient(ref_lp)
        mx.clear_cache()
    # GPU cooldown between ref phase and gradient phase
    mx.synchronize()
    mx.clear_cache()
    if gpu_cooldown > 0:
        import time as _t

        _t.sleep(gpu_cooldown)

    # ── Phase 2: gradient computation (policy forward + backward in v&g) ──
    for i, (seq, plen, mask, adv, seg_w) in enumerate(
        zip(sequences, prompt_lengths, masks, advantages, seg_list, strict=True)
    ):
        if not seq or plen >= len(seq) or not any(mask):
            continue

        ref_lp = ref_logps[i]
        if ref_lp is None:
            continue

        # Per-sequence loss params (scalar or list)
        beta_i = beta[i] if isinstance(beta, list) else beta
        eps_i = epsilon[i] if isinstance(epsilon, list) else epsilon
        epsh_i = epsilon_high[i] if isinstance(epsilon_high, list) else epsilon_high

        kl_out: list[Any] = []
        clip_stats: list[Any] = []

        def loss_fn(
            m,
            _seq=seq,
            _plen=plen,
            _mask=mask,
            _adv=adv,
            _ref_lp=ref_lp,
            _seg_w=seg_w,
            _cs=clip_stats,
            _ko=kl_out,
            _beta=beta_i,
            _eps=eps_i,
            _epsh=epsh_i,
            _aux=aux_loss_fn,
        ):
            loss = grpo_loss_fn(
                m,
                [_ref_lp],
                [_seq],
                [_plen],
                [_mask],
                [_adv],
                _beta,
                [_seg_w] if _seg_w is not None else None,
                _eps,
                _epsh,
                _cs,
                _ko,
            )
            if _aux is not None:
                loss = loss + _aux(m)
            return loss

        loss_i, grads_i = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss_i)
        _tree_eval_chunked(grads_i)

        # Per-sequence gradient norm clipping (from type-level max_grad_norm)
        if per_seq_grad_norm is not None:
            gn_cap = per_seq_grad_norm[i]
            if gn_cap > 0:
                gnorm_i = _tree_norm(grads_i)
                if gnorm_i > gn_cap:
                    grads_i = _tree_scale(grads_i, gn_cap / gnorm_i)
                    _tree_eval(grads_i)

        # Extract clip stats — now raw mx.arrays, safe to .item() after v&g
        if len(clip_stats) == 3:
            tc = int(clip_stats[2].item())
            if tc > 0:
                total_clip_low += int(clip_stats[0].item())
                total_clip_high += int(clip_stats[1].item())
                total_clip_mask += tc

        accum_loss += float(loss_i)
        if kl_out:
            accum_kl += float(kl_out[0].item())
        accum_grads = grads_i if accum_grads is None else _tree_add(accum_grads, grads_i)
        n_valid += 1

        del loss_i, grads_i
        mx.synchronize()
        mx.clear_cache()
        if gpu_cooldown > 0:
            import time as _t

            _t.sleep(gpu_cooldown)

    if n_skipped > 0 and verbose:
        print(f"  [grad] skipped {n_skipped}/{len(sequences)} seqs (prompt alone > {max_seq_len})")
    if n_valid == 0 or accum_grads is None:
        return 0.0, {}, 0.0, 0.0, 0.0, 0.0

    if n_valid > 1:
        accum_grads = _tree_scale(accum_grads, 1.0 / n_valid)
    _tree_eval(accum_grads)
    mx.clear_cache()
    accum_loss /= n_valid

    grad_norm = _tree_norm(accum_grads)
    if max_grad_norm > 0 and grad_norm > max_grad_norm and grad_norm > 0:
        accum_grads = _tree_scale(accum_grads, max_grad_norm / grad_norm)
        _tree_eval(accum_grads)

    clip_low = total_clip_low / max(total_clip_mask, 1)
    clip_high = total_clip_high / max(total_clip_mask, 1)
    avg_kl = accum_kl / max(n_valid, 1)

    return accum_loss, accum_grads, float(grad_norm), clip_low, clip_high, avg_kl


def apply_grads(model: Any, optimizer: Any, grads: Any) -> None:
    """Apply pre-computed gradients and materialize parameters.

    Evaluates trainable parameters and optimizer state, then clears the
    Metal cache to prevent macOS interactivity timeout.
    """
    optimizer.update(model, grads)
    mx.eval(model.trainable_parameters(), optimizer.state)
    mx.clear_cache()


def gradient_step(
    model: Any,
    ref_model: Any,
    optimizer: Any,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    masks: list[list[bool]],
    advantages: list[float],
    beta: float | list[float] = 0.04,
    max_grad_norm: float = 1.0,
    segment_weights: list[list[float] | None] | None = None,
    epsilon: float | list[float] = 0.0,
    epsilon_high: float | list[float] | None = None,
    loss_type: str = "grpo",
    max_seq_len: int = 1024,
    gpu_cooldown: float = 0.0,
    per_seq_grad_norm: list[float] | None = None,
    aux_loss_fn: Callable[[Any], mx.array] | None = None,
    verbose: bool = False,
) -> GradStepResult:
    """Complete gradient step: forward → loss → backward → update.

    Convenience wrapper around compute_grads + apply_grads.
    """
    loss, grads, grad_norm, clip_low, clip_high, kl_div = compute_grads(
        model,
        ref_model,
        sequences,
        prompt_lengths,
        masks,
        advantages,
        beta,
        max_grad_norm,
        segment_weights,
        epsilon,
        epsilon_high,
        loss_type,
        max_seq_len,
        gpu_cooldown,
        per_seq_grad_norm,
        aux_loss_fn,
        verbose,
    )
    if grads:  # Empty when all sequences skipped (adv=0, too long, etc.)
        apply_grads(model, optimizer, grads)

    return GradStepResult(
        loss=loss,
        grad_norm=grad_norm,
        kl_divergence=kl_div,
        effective_tokens=sum(sum(1 for m in mask if m) for mask in masks),
        clip_frac_low=clip_low,
        clip_frac_high=clip_high,
    )


def extract_sequences_from_rollouts(
    rollouts: list,
    backend: Any,
) -> tuple[list[list[int]], list[int], list[list[bool]]]:
    """Extract token sequences, prompt lengths, and masks from rollouts.

    Converts RolloutResult objects into the raw arrays needed by grpo_loss_fn.

    Args:
        rollouts: List of RolloutResult
        backend: Generation backend (for encode)

    Returns:
        (sequences, prompt_lengths, masks)
    """
    sequences = []
    prompt_lengths = []
    masks = []

    for r in rollouts:
        # Full sequence = prompt tokens + completion tokens
        prompt_toks = r.prompt_tokens
        comp_toks = r.completion_tokens

        full_seq = prompt_toks + comp_toks
        sequences.append(full_seq)
        prompt_lengths.append(len(prompt_toks))
        masks.append(r.loss_mask)

    return sequences, prompt_lengths, masks


# ── Tree utilities for nested MLX arrays ──


def _tree_norm(tree: Any) -> float:
    """Compute L2 norm of a nested dict/list of mx.arrays.

    Uses tree_flatten + single lazy graph to avoid per-leaf .item() overhead.
    """
    from mlx.utils import tree_flatten

    leaves = tree_flatten(tree)
    if not leaves:
        return 0.0
    sq_sum = mx.array(0.0)
    for _, v in leaves:
        if hasattr(v, "shape"):
            sq_sum = sq_sum + mx.sum(mx.square(v))
    return float(mx.sqrt(sq_sum).item())


def _tree_add(a: Any, b: Any) -> Any:
    """Element-wise add two nested dicts/lists of mx.arrays."""
    if isinstance(a, dict):
        return {k: _tree_add(a[k], b[k]) for k in a}
    elif isinstance(a, list):
        return [_tree_add(ai, bi) for ai, bi in zip(a, b, strict=True)]
    elif hasattr(a, "shape"):
        return a + b
    return a


def _tree_scale(tree: Any, scale: float) -> Any:
    """Scale all arrays in a nested dict/list by a constant."""
    if isinstance(tree, dict):
        return {k: _tree_scale(v, scale) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_scale(v, scale) for v in tree]
    elif hasattr(tree, "shape"):
        return tree * scale
    return tree


def _tree_eval(tree: Any) -> None:
    """Force evaluation of all mx.arrays in a nested dict/list.

    Prevents lazy chain accumulation across iterations by materializing
    gradient arrays before they're passed to the optimizer.
    """
    from mlx.utils import tree_flatten

    leaves = tree_flatten(tree)
    arrays = [v for _, v in leaves if hasattr(v, "shape")]
    if arrays:
        mx.eval(*arrays)


def _tree_eval_chunked(tree: Any, chunk: int = 8) -> None:
    """Evaluate gradient tree in reverse-layer chunks to avoid Metal timeout.

    Gradient arrays are ordered input→output (layers[0], layers[1], ...).
    Evaluating in REVERSE order (output→input) means each chunk only
    triggers one layer's backward recomputation + gradient computation,
    since downstream gradients are already materialized.
    """
    from mlx.utils import tree_flatten

    leaves = tree_flatten(tree)
    arrays = [v for _, v in leaves if hasattr(v, "shape")]
    arrays.reverse()
    for i in range(0, len(arrays), chunk):
        mx.eval(*arrays[i : i + chunk])


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: pip install mlx (required for gradient operations)
# Usage: result = gradient_step(model, ref_model, optimizer, sequences, ...)
#        seqs, plens, masks = extract_sequences_from_rollouts(rollouts, backend)
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
