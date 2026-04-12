"""PyTorch gradient operations — GRPO training on CUDA/MPS/CPU.

Mirror of mlx_ops.py using standard PyTorch autograd. No lazy evaluation,
no traced scopes — just loss.backward() and optimizer.step().

Reuses GradStepResult and extract_sequences_from_rollouts from mlx_ops
(they're pure Python, importable without MLX).
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch.nn import functional as f_nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from mlx_grpo.loss.mlx_ops import GradStepResult


def _forward_pass_fn(
    model: Any,
    tokens: Any,
    device: str = "cuda",
) -> Any:
    """Wrapper for forward pass to enable gradient checkpointing.

    Args:
        model: PyTorch causal LM model.
        tokens: Input token tensor of shape [1, seq_len].
        device: Device string for operations.

    Returns:
        Model output (logits or CausalLMOutput).
    """
    output = model(tokens)
    return output


def compute_log_probs_for_tokens(
    model: Any,
    token_ids: list[int],
    device: str = "cuda",
    use_checkpoint: bool = False,
) -> Any:
    """Forward pass -> per-token log probs for a sequence.

    Uses f_nn.cross_entropy(reduction='none') for efficiency.
    Optionally uses gradient checkpointing to reduce memory usage.

    Args:
        model: PyTorch causal LM — model(input_ids) returns logits or CausalLMOutput
        token_ids: Full token sequence [prompt + completion]
        device: Target device string
        use_checkpoint: Enable gradient checkpointing for reduced memory usage

    Returns:
        Tensor of shape [T-1] — log P(token_{t+1} | tokens_{0..t})
    """
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    if use_checkpoint and hasattr(torch.utils.checkpoint, "checkpoint"):
        output = torch.utils.checkpoint.checkpoint(
            _forward_pass_fn,
            model,
            tokens,
            device,
            use_reentrant=False,
        )
    else:
        output = model(tokens)

    logits = output.logits if hasattr(output, "logits") else output

    seq_len = len(token_ids) - 1
    shift_logits = logits[0, :seq_len]  # [seq_len, vocab]
    target_ids = tokens[0, 1 : seq_len + 1]  # [seq_len]

    nll = f_nn.cross_entropy(shift_logits, target_ids, reduction="none")
    return -nll  # [seq_len] — log probs


def compute_log_probs_batch(
    model: Any,
    token_id_lists: list[list[int]],
    device: str = "cuda",
    use_checkpoint: bool = False,
) -> list[Any]:
    """Batched forward pass → per-token log probs for multiple sequences.

    Pads sequences to the same length and runs a SINGLE forward pass through
    the model. On H100/A100 this is 5-20× faster than sequential calls.

    Args:
        model: PyTorch causal LM (or PeftModel wrapper)
        token_id_lists: List of token sequences (variable length)
        device: Target device string
        use_checkpoint: Enable gradient checkpointing

    Returns:
        List of tensors, one per sequence, each of shape [T_i - 1]
    """
    if not token_id_lists:
        return []

    batch_size = len(token_id_lists)
    seq_lens = [len(s) for s in token_id_lists]
    max_len = max(seq_lens)

    # Pad sequences and build attention mask
    padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    attn_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(token_id_lists):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, : len(seq)] = 1

    # Single batched forward pass
    if use_checkpoint and hasattr(torch.utils.checkpoint, "checkpoint"):
        output = torch.utils.checkpoint.checkpoint(
            lambda ids, mask: model(ids, attention_mask=mask),
            padded,
            attn_mask,
            use_reentrant=False,
        )
    else:
        output = model(padded, attention_mask=attn_mask)

    logits = output.logits if hasattr(output, "logits") else output

    # Extract per-sequence log probs from the batched output
    results = []
    for i in range(batch_size):
        sl = seq_lens[i] - 1
        if sl <= 0:
            results.append(torch.tensor([0.0], device=device))
            continue
        nll = f_nn.cross_entropy(logits[i, :sl], padded[i, 1 : sl + 1], reduction="none")
        results.append(-nll)

    return results


def precompute_ref_logps(
    ref_model: Any,
    sequences: list[list[int]],
    device: str = "cuda",
) -> list[Any]:
    """Pre-compute reference model log probs (batched).

    Args:
        ref_model: Frozen reference model (or adapter-disabled wrapper)
        sequences: Full token sequences [prompt + completion] per rollout
        device: Target device string

    Returns:
        List of detached tensors, one per sequence, each of shape [T-1]
    """
    valid = [(i, seq) for i, seq in enumerate(sequences) if len(seq) >= 2]
    ref_logps: list[Any] = [torch.tensor([0.0], device=device)] * len(sequences)

    if not valid:
        return ref_logps

    with torch.no_grad():
        batch_seqs = [seq for _, seq in valid]
        batch_lps = compute_log_probs_batch(ref_model, batch_seqs, device)
        for (orig_idx, _), lp in zip(valid, batch_lps):
            ref_logps[orig_idx] = lp.detach()

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
    device: str = "cuda",
    use_checkpoint: bool = False,
) -> tuple[Any, float, float, float]:
    """Differentiable GRPO loss for PyTorch.

    Returns:
        (loss_tensor, mean_kl, clip_frac_low, clip_frac_high)
    """
    use_clipping = epsilon > 0
    eps_high = epsilon_high if epsilon_high is not None else epsilon

    total_policy = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_tokens = 0
    clip_low_count = 0
    clip_high_count = 0
    total_mask_count = 0

    if segment_weights is not None:
        items = list(
            zip(
                sequences,
                prompt_lengths,
                masks,
                advantages,
                ref_logps,
                segment_weights,
                strict=False,
            )
        )
    else:
        items = list(
            zip(
                sequences,
                prompt_lengths,
                masks,
                advantages,
                ref_logps,
                [None] * len(sequences),
                strict=False,
            )
        )

    for seq, plen, mask, adv, ref_lp, seg_w in items:
        if not seq or plen >= len(seq):
            continue

        current_lp = compute_log_probs_for_tokens(model, seq, device, use_checkpoint)

        # Log probs are shifted: lp[i] = log P(token_{i+1} | tokens_{0..i}).
        # First completion token (at position plen) has its log prob at
        # index plen-1, so slice from max(plen-1, 0).
        start = max(plen - 1, 0)
        comp_current = current_lp[start:]
        comp_ref = ref_lp[start:]

        t_len = min(len(mask), comp_current.shape[0], comp_ref.shape[0])
        if t_len == 0:
            continue
        if not any(mask[:t_len]):
            continue

        mask_arr = torch.tensor(mask[:t_len], dtype=torch.float32, device=device)

        if seg_w is not None:
            weight_arr = torch.tensor(seg_w[:t_len], dtype=torch.float32, device=device)
            mask_arr = mask_arr * weight_arr

        n_active = mask_arr.sum()

        log_ratio = comp_current[:t_len] - comp_ref[:t_len]
        ratio = torch.exp(log_ratio)

        if use_clipping:
            clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + eps_high)
            unclipped_obj = ratio * adv
            clipped_obj = clipped_ratio * adv
            pg_per_token = -torch.minimum(unclipped_obj, clipped_obj)
            pg_term = (pg_per_token * mask_arr).sum()

            binary_mask = torch.tensor(
                mask[:t_len],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                is_low = ((ratio < (1.0 - epsilon)).float() * binary_mask).sum().item()
                is_high = ((ratio > (1.0 + eps_high)).float() * binary_mask).sum().item()
                if adv < 0:
                    clip_low_count += int(is_low)
                if adv > 0:
                    clip_high_count += int(is_high)
                total_mask_count += int(binary_mask.sum().item())
        else:
            pg_term = -adv * (log_ratio * mask_arr).sum()

        total_policy = total_policy + pg_term

        # KL divergence (Schulman k3): r - log(r) - 1 where r = pi_ref/pi
        # Upcast to float32: bfloat16 causes catastrophic cancellation in
        # exp(x) - x - 1 for small x, producing negative KL values.
        log_ratio_ref = (comp_ref[:t_len] - comp_current[:t_len]).float()
        ratio_ref = torch.exp(log_ratio_ref)
        kl_per_token = (ratio_ref - log_ratio_ref - 1.0).clamp(min=0.0)
        total_kl = total_kl + (kl_per_token * mask_arr.float()).sum()

        total_tokens += int(n_active.item())

    if total_tokens == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, 0.0, 0.0, 0.0

    loss = total_policy / total_tokens + beta * total_kl / total_tokens
    mean_kl = float(total_kl.detach().item()) / total_tokens

    clip_low = clip_low_count / max(total_mask_count, 1)
    clip_high = clip_high_count / max(total_mask_count, 1)

    return loss, mean_kl, clip_low, clip_high


def grpo_loss_from_logps(
    policy_logps: list[Any],
    ref_logps: list[Any],
    sequences: list[list[int]],
    prompt_lengths: list[int],
    masks: list[list[bool]],
    advantages: list[float],
    beta: float = 0.04,
    segment_weights: list[list[float]] | None = None,
    epsilon: float = 0.0,
    epsilon_high: float | None = None,
    device: str = "cuda",
) -> tuple[Any, float, float, float]:
    """GRPO loss from pre-computed policy and reference log probs.

    This is the same as grpo_loss_fn but skips the forward pass,
    enabling batched forward pass precomputation.

    Args:
        policy_logps: Pre-computed policy model log probs per sequence
        ref_logps: Pre-computed reference model log probs per sequence
        (rest same as grpo_loss_fn)

    Returns:
        (loss_tensor, mean_kl, clip_frac_low, clip_frac_high)
    """
    use_clipping = epsilon > 0
    eps_high = epsilon_high if epsilon_high is not None else epsilon

    total_policy = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_tokens = 0
    clip_low_count = 0
    clip_high_count = 0
    total_mask_count = 0

    if segment_weights is not None:
        items = list(
            zip(
                policy_logps,
                ref_logps,
                sequences,
                prompt_lengths,
                masks,
                advantages,
                segment_weights,
                strict=False,
            )
        )
    else:
        items = list(
            zip(
                policy_logps,
                ref_logps,
                sequences,
                prompt_lengths,
                masks,
                advantages,
                [None] * len(sequences),
                strict=False,
            )
        )

    for current_lp, ref_lp, seq, plen, mask, adv, seg_w in items:
        if not seq or plen >= len(seq):
            continue

        # Log probs are shifted: lp[i] = log P(token_{i+1} | tokens_{0..i}).
        start = max(plen - 1, 0)
        comp_current = current_lp[start:]
        comp_ref = ref_lp[start:]

        t_len = min(len(mask), comp_current.shape[0], comp_ref.shape[0])
        if t_len == 0:
            continue
        if not any(mask[:t_len]):
            # Debug: mask has no active tokens
            import logging

            logging.getLogger(__name__).warning(
                "Skipping: all-zero mask. sum(mask[:t_len])=%.1f, mask[:10]=%s, "
                "seq_len=%d, plen=%d, comp shapes: cur=%s ref=%s",
                sum(mask[:t_len]),
                mask[:10],
                len(seq),
                plen,
                comp_current.shape,
                comp_ref.shape,
            )
            continue

        mask_arr = torch.tensor(mask[:t_len], dtype=torch.float32, device=device)

        if seg_w is not None:
            weight_arr = torch.tensor(seg_w[:t_len], dtype=torch.float32, device=device)
            mask_arr = mask_arr * weight_arr

        n_active = mask_arr.sum()

        log_ratio = comp_current[:t_len] - comp_ref[:t_len]
        ratio = torch.exp(log_ratio)

        if use_clipping:
            clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + eps_high)
            unclipped_obj = ratio * adv
            clipped_obj = clipped_ratio * adv
            pg_per_token = -torch.minimum(unclipped_obj, clipped_obj)
            pg_term = (pg_per_token * mask_arr).sum()

            binary_mask = mask_arr.clone()
            with torch.no_grad():
                is_low = ((ratio < (1.0 - epsilon)).float() * binary_mask).sum().item()
                is_high = ((ratio > (1.0 + eps_high)).float() * binary_mask).sum().item()
                if adv < 0:
                    clip_low_count += int(is_low)
                if adv > 0:
                    clip_high_count += int(is_high)
                total_mask_count += int(binary_mask.sum().item())
        else:
            pg_term = -adv * (log_ratio * mask_arr).sum()

        total_policy = total_policy + pg_term

        # KL divergence (Schulman k3): r - log(r) - 1 where r = pi_ref/pi
        # Upcast to float32: bfloat16 causes catastrophic cancellation in
        # exp(x) - x - 1 for small x, producing negative KL values.
        log_ratio_ref = (comp_ref[:t_len] - comp_current[:t_len]).float()
        ratio_ref = torch.exp(log_ratio_ref)
        kl_per_token = (ratio_ref - log_ratio_ref - 1.0).clamp(min=0.0)
        total_kl = total_kl + (kl_per_token * mask_arr.float()).sum()

        total_tokens += int(n_active.item())

    if total_tokens == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, 0.0, 0.0, 0.0

    loss = total_policy / total_tokens + beta * total_kl / total_tokens
    mean_kl = float(total_kl.detach().item()) / total_tokens

    clip_low = clip_low_count / max(total_mask_count, 1)
    clip_high = clip_high_count / max(total_mask_count, 1)

    return loss, mean_kl, clip_low, clip_high


def compute_grads(
    model: Any,
    ref_model: Any,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    masks: list[list[bool]],
    advantages: list[float],
    beta: float = 0.04,
    max_grad_norm: float = 1.0,
    segment_weights: list[list[float]] | None = None,
    epsilon: float = 0.0,
    epsilon_high: float | None = None,
    device: str = "cuda",
    use_checkpoint: bool = False,
) -> tuple[float, float, float, float, float]:
    """Compute and accumulate gradients into param.grad.

    Optimization: Precompute all policy forward passes first, then accumulate
    gradients via .backward(). This enables better GPU scheduling vs per-sequence
    forward+backward.

    Returns:
        (loss, grad_norm, clip_frac_low, clip_frac_high, kl_divergence)
    """
    import logging
    import sys

    _log = logging.getLogger(__name__)

    # ── Version gate: confirm the fixed code is running ──
    if not getattr(compute_grads, "_announced", False):
        print(
            "[torch_ops] compute_grads v3: batched forward + enable_grad + "
            f"disable_adapter ctx mgr | device={device}",
            file=sys.stderr,
            flush=True,
        )
        compute_grads._announced = True  # type: ignore[attr-defined]

    # Precompute all reference log probs (these don't require grad)
    ref_logps = precompute_ref_logps(ref_model, sequences, device)

    # ── Ensure model is ready for gradient computation ──
    # 1. train() mode — required for LoRA/dropout to work properly
    # 2. enable_grad() — overrides any outer torch.no_grad() context
    # NOTE: enable_grad() cannot override torch.inference_mode().
    model.train()

    # Identify valid sequences (skip empty / all-masked)
    valid_indices: list[int] = []
    for i, (seq, plen, mask) in enumerate(
        zip(sequences, prompt_lengths, masks, strict=False)
    ):
        if seq and plen < len(seq) and any(mask):
            valid_indices.append(i)

    if not valid_indices:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Batched forward pass — single GPU call for all sequences at once
    with torch.enable_grad():
        valid_seqs = [sequences[i] for i in valid_indices]
        valid_logps = compute_log_probs_batch(model, valid_seqs, device, use_checkpoint)

    # Map back to full-size list (None for skipped sequences)
    policy_logps: list[Any] = [None] * len(sequences)
    for idx, lp in zip(valid_indices, valid_logps):
        policy_logps[idx] = lp

    # ── Verify gradient graph is connected ──
    _sample = policy_logps[valid_indices[0]]
    if _sample is not None and not _sample.requires_grad:
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        _log.error(
            "GRAD BROKEN: policy logps requires_grad=False! "
            "trainable_params=%d, model.training=%s, grad_enabled=%s, "
            "inference_mode=%s, grad_fn=%s",
            n_train,
            getattr(model, "training", "?"),
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled()
            if hasattr(torch, "is_inference_mode_enabled")
            else "n/a",
            type(_sample.grad_fn).__name__ if _sample.grad_fn else "None",
        )

    # Now compute loss and accumulate gradients
    model.zero_grad()
    seg_list = list(segment_weights) if segment_weights is not None else [None] * len(sequences)

    accum_loss = 0.0
    accum_kl = 0.0
    total_clip_low = 0.0
    total_clip_high = 0.0

    # Accumulate loss across all sequences, then ONE backward through the
    # shared batched computation graph.  Per-sequence backward would fail
    # because the batched forward pass creates a single shared graph.
    batch_loss = torch.tensor(0.0, device=device)
    with torch.enable_grad():
        for idx in valid_indices:
            seq = sequences[idx]
            plen = prompt_lengths[idx]
            mask = masks[idx]
            adv = advantages[idx]
            ref_lp = ref_logps[idx]
            policy_lp = policy_logps[idx]
            seg_w = seg_list[idx]

            loss_i, kl_i, cl, ch = grpo_loss_from_logps(
                [policy_lp],
                [ref_lp],
                [seq],
                [plen],
                [mask],
                [adv],
                beta,
                [seg_w] if seg_w is not None else None,
                epsilon,
                epsilon_high,
                device,
            )

            batch_loss = batch_loss + loss_i
            accum_loss += float(loss_i.detach().item())
            accum_kl += kl_i
            total_clip_low += cl
            total_clip_high += ch

        # Single backward through the entire batched graph.
        # Must stay inside enable_grad: the division creates a new graph node.
        if batch_loss.requires_grad:
            (batch_loss / max(len(valid_indices), 1)).backward()
        else:
            _log.error(
                "GRAD BROKEN: batch loss requires_grad=False. "
                "Check model.train() and adapter setup."
            )

    n_valid = len(valid_indices)

    # No need for manual grad averaging — single backward on (batch_loss / n_valid)
    # already produces correctly averaged gradients.

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_grad_norm,
    ).item()

    # ── Debug: write to file (rich TUI eats stderr) ──
    _dbg_path = "/tmp/_grpo_grad_debug.log"
    try:
        _sample_lp = policy_logps[valid_indices[0]]
        _n_train = sum(1 for p in model.parameters() if p.requires_grad)
        _n_with_grad = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().max().item() > 0
        )
        with open(_dbg_path, "a") as _f:
            _f.write(
                f"compute_grads: grad_norm={grad_norm:.6f} kl={accum_kl/max(n_valid,1):.6f} "
                f"loss={accum_loss/max(n_valid,1):.6f} n_valid={n_valid} "
                f"batch_loss.requires_grad={batch_loss.requires_grad} "
                f"policy_lp.requires_grad={_sample_lp.requires_grad if _sample_lp is not None else 'N/A'} "
                f"trainable={_n_train} params_with_grad={_n_with_grad} "
                f"model.training={getattr(model, 'training', '?')}\n"
            )
    except Exception:
        pass

    avg_loss = accum_loss / n_valid
    avg_kl = accum_kl / n_valid
    avg_cl = total_clip_low / n_valid
    avg_ch = total_clip_high / n_valid

    return avg_loss, grad_norm, avg_cl, avg_ch, avg_kl


def apply_grads(
    model: Any,
    optimizer: Any,
    max_grad_norm: float = 1.0,
) -> None:
    """Apply accumulated gradients: clip + step + zero_grad."""
    optimizer.step()
    optimizer.zero_grad()


def gradient_step(
    model: Any,
    ref_model: Any,
    optimizer: Any,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    masks: list[list[bool]],
    advantages: list[float],
    beta: float = 0.04,
    max_grad_norm: float = 1.0,
    segment_weights: list[list[float]] | None = None,
    epsilon: float = 0.0,
    epsilon_high: float | None = None,
    device: str = "cuda",
    use_checkpoint: bool = False,
) -> GradStepResult:
    """Complete gradient step: forward -> loss -> backward -> update.

    Args:
        use_checkpoint: Enable gradient checkpointing to reduce memory usage.
                       Trades memory for recomputation during backward pass.
    """
    loss, grad_norm, clip_low, clip_high, kl_div = compute_grads(
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
        device,
        use_checkpoint,
    )
    apply_grads(model, optimizer, max_grad_norm)

    # Diagnostics: surface root cause when grad is zero
    if grad_norm == 0.0:
        import logging

        _log = logging.getLogger(__name__)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        has_any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        _log.warning(
            "Zero grad_norm — trainable_params=%d, any_nonzero_grad=%s, "
            "advantages=%s, kl=%.6f, loss=%.6f",
            n_trainable,
            has_any_grad,
            [round(a, 4) for a in advantages[:8]],
            kl_div,
            loss,
        )

    return GradStepResult(
        loss=loss,
        grad_norm=grad_norm,
        kl_divergence=kl_div,
        effective_tokens=sum(sum(1 for m in mask if m) for mask in masks),
        clip_frac_low=clip_low,
        clip_frac_high=clip_high,
    )


def sft_gradient_step(
    model: Any,
    optimizer: Any,
    token_ids: list[int],
    prompt_length: int,
    lr_multiplier: float = 0.1,
    device: str = "cuda",
) -> float:
    """SFT anchor gradient step with reduced learning rate."""
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    output = model(tokens)
    logits = output.logits if hasattr(output, "logits") else output

    seq_len = len(token_ids) - 1
    # Log probs are shifted: logits[i] predicts token_{i+1}. First answer
    # token (at position prompt_length) has logits at index prompt_length-1.
    start = max(prompt_length - 1, 0)
    if start >= seq_len:
        return 0.0

    shift_logits = logits[0, start:seq_len]
    targets = tokens[0, start + 1 : seq_len + 1]

    loss = f_nn.cross_entropy(shift_logits, targets)

    # Scale loss by lr_multiplier to effectively reduce learning rate
    scaled_loss = loss * lr_multiplier
    optimizer.zero_grad()
    scaled_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return float(loss.detach().item())
