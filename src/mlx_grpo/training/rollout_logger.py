"""Rollout logging — comprehensive JSONL logs for training analysis.

Each training step logs one entry per rollout with:
  - Input: prompt, ground truth answer, type, generation params
  - Output: full completion text, per-segment breakdown (generated/injected/masked)
  - Scoring: reward, advantage, correctness, token counts
  - Step context: iteration, group index, step-level loss/KL/grad_norm
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_grpo.training.segment_enricher import enrich_segment


def _compute_gradient_text(
    entry: dict,
    advantage: float,
    rollout: Any,
    decode_fn: Any,
) -> dict[str, Any]:
    """Compute per-token gradient direction fields.

    Returns dict with:
      text_with_positive_gradient: decoded tokens that get REINFORCED
      text_with_negative_gradient: decoded tokens that get SUPPRESSED
      gradient_token_counts: {positive, negative, zero} token counts
    """
    loss_mask = rollout.loss_mask
    all_tokens = [t for s in rollout.segments for t in s.tokens]

    # Gather per-token effective weight from enriched segments
    enriched_multipliers: list[float] = []
    for seg in entry.get("segments", []):
        for td in seg.get("token_details", []):
            enriched_multipliers.append(td.get("gradient_multiplier", 1.0))

    pos_tokens: list[int] = []
    neg_tokens: list[int] = []
    for i, tok in enumerate(all_tokens):
        w = loss_mask[i] if i < len(loss_mask) else 0.0
        if w == 0.0:
            continue
        # Enriched multiplier refines the weight (jaccard, IS, etc.)
        em = enriched_multipliers[i] if i < len(enriched_multipliers) else 1.0
        effective = advantage * w * em
        if effective > 0:
            pos_tokens.append(tok)
        elif effective < 0:
            neg_tokens.append(tok)

    pos_text = decode_fn(pos_tokens) if decode_fn and pos_tokens else None
    neg_text = decode_fn(neg_tokens) if decode_fn and neg_tokens else None

    return {
        "text_with_positive_gradient": pos_text,
        "text_with_negative_gradient": neg_text,
        "gradient_token_counts": {
            "positive": len(pos_tokens),
            "negative": len(neg_tokens),
            "zero": len(all_tokens) - len(pos_tokens) - len(neg_tokens),
        },
    }


@dataclass(slots=True)
class StepContext:
    """Step-level context passed alongside rollouts."""

    iteration: float = 0.0
    total_iterations: int = 0
    loss: float = 0.0
    kl_divergence: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    generation_sec: float = 0.0
    elapsed_sec: float = 0.0
    mean_reward: float = 0.0
    reward_std: float = 0.0
    cka_drift: float | None = None  # last CKAProbe evaluation, None if probe not active
    dataset_row: int = -1  # dataset index of the sample (-1 = unknown)
    repeat_idx: int = 0  # which repetition within sample_repeat (1-based; 0 = single step)
    naked_phase: str = ""  # "before", "normal", "after", or "" if naked repeats disabled
    weight_angle_avg: float | None = None  # mean cosine angle from init across trained layers
    gradient_multipliers: dict[str, float] | None = None  # per-layer L20-L35 multipliers
    gradient_multiplier_max: float | None = None  # max multiplier across layers
    gradient_multiplier_mean: float | None = None  # mean multiplier across layers
    accumulation_step: int = 0  # current gradient accumulation step
    accumulation_steps_total: int = 0  # total accumulation steps for this iteration
    accumulation_over_repeats: int = 0  # how many repeats accumulated in this step
    gar_threshold_deg: float | None = None  # GAR angle threshold (degrees); None = not in GAR mode
    gar_normalize: bool | None = None  # whether GAR angles are normalized


class RolloutLogger:
    """Appends detailed rollout entries to a JSONL file.

    Each line is one rollout with full input/output/scoring details.
    """

    def __init__(
        self,
        path: str | Path,
        log_tokens: bool = True,
        correct_threshold: float = 0.5,
        decode_fn: Any = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._log_tokens = log_tokens
        self._correct_threshold = correct_threshold
        self._decode_fn = decode_fn
        self._file = open(self._path, "a")  # noqa: SIM115

    def log(
        self,
        iteration: int,
        rollouts: list[Any],
        rewards: list[float],
        advantages: list[float],
        prompts: list[str] | None = None,
        answers: list[str] | None = None,
        targets: list[str] | None = None,
        step_ctx: StepContext | None = None,
        group_size: int | None = None,
    ) -> None:
        """Log a batch of rollouts with full context.

        Field layout (top-level, human-readable):
          prompt              — input prompt
          generated_completion — full model output (injected + generated)
          generated_answer    — extracted answer from model output
          target_completion   — gold completion (may include thinking trace)
          target_answer       — clean ground-truth answer for comparison
          reward / advantage / correct — scoring
          type / token_counts / segments / variables — detail
          step                — iteration-level metrics

        Args:
            group_size: members per group. If set, "correct" = True for max-advantage member.
                        If None, uses reward threshold (legacy behavior).
        """
        _prompts = prompts or [""] * len(rollouts)
        _answers = answers or [""] * len(rollouts)
        _targets = targets or _answers
        ts = time.time()

        # Compute max-advantage indices per group if group_size provided
        max_adv_indices: set[int] = set()
        if group_size and group_size > 1:
            for group_idx in range(0, len(advantages), group_size):
                group_advs = advantages[group_idx : group_idx + group_size]
                if group_advs:
                    max_idx = group_advs.index(max(group_advs))
                    max_adv_indices.add(group_idx + max_idx)

        for i, (rollout, reward, adv) in enumerate(
            zip(rollouts, rewards, advantages, strict=True),
        ):
            # Core rollout data (generated_completion, generated_answer, segments, etc.)
            entry = rollout.to_log_entry(include_tokens=self._log_tokens, decode_fn=self._decode_fn)

            # Enrich segments with gradient metadata (backward compatible: adds fields only)
            if "segments" in entry and "metadata" in entry:
                meta = entry["metadata"]
                jaccard_mask = meta.get("_jaccard_boosted_mask", [])
                is_weights = meta.get("_is_weights", [])
                token_offset = 0
                for segment in entry["segments"]:
                    segment, token_offset = enrich_segment(
                        segment, token_offset, jaccard_mask, is_weights, self._decode_fn
                    )

            # Per-token gradient direction: advantage * per_token_weight
            entry.update(_compute_gradient_text(entry, adv, rollout, self._decode_fn))

            # Context
            entry["iteration"] = iteration
            entry["group_index"] = i
            entry["timestamp"] = ts

            # Input / output / target — clear naming
            entry["prompt"] = _prompts[i % len(_prompts)]
            entry["target_completion"] = _answers[i % len(_answers)]
            entry["target_answer"] = _targets[i % len(_targets)]

            # Scoring — GAR mode has no reward/advantage (pure activation-supervision)
            if entry.get("gar_mode"):
                entry["reward"] = 0.0
                entry["advantage"] = 0.0
            else:
                entry["reward"] = reward
                entry["advantage"] = adv
            # "correct" = True if this member has max advantage in its group
            entry["correct"] = (
                i in max_adv_indices if max_adv_indices else reward > self._correct_threshold
            )

            # Step-level metrics (same for all rollouts in a step)
            if step_ctx:
                step_dict: dict[str, object] = {
                    "loss": step_ctx.loss,
                    "kl": step_ctx.kl_divergence,
                    "grad_norm": step_ctx.grad_norm,
                    "tok_per_sec": step_ctx.tokens_per_sec,
                    "generation_sec": step_ctx.generation_sec,
                    "elapsed_sec": step_ctx.elapsed_sec,
                    "mean_reward": step_ctx.mean_reward,
                    "reward_std": step_ctx.reward_std,
                    "progress": (step_ctx.iteration / max(step_ctx.total_iterations, 1)),
                    "dataset_row": step_ctx.dataset_row,
                    "repeat_idx": step_ctx.repeat_idx,
                    "naked_phase": step_ctx.naked_phase,
                }
                if step_ctx.cka_drift is not None:
                    step_dict["cka_drift"] = step_ctx.cka_drift
                if step_ctx.weight_angle_avg is not None:
                    step_dict["weight_angle_avg"] = step_ctx.weight_angle_avg

                # Step and gradient accumulation metadata
                metadata = {
                    "step": int(step_ctx.iteration) + 1,
                    "total_steps": step_ctx.total_iterations,
                }
                if step_ctx.accumulation_steps_total > 0:
                    metadata["accum_step"] = step_ctx.accumulation_step
                    metadata["accum_steps_total"] = step_ctx.accumulation_steps_total
                    metadata["accum_over_repeats"] = step_ctx.accumulation_over_repeats

                # Dynamic gradient scaling metadata
                if step_ctx.gradient_multipliers is not None:
                    metadata["gradient_multipliers"] = step_ctx.gradient_multipliers
                if step_ctx.gradient_multiplier_max is not None:
                    metadata["gradient_multiplier_max"] = step_ctx.gradient_multiplier_max
                if step_ctx.gradient_multiplier_mean is not None:
                    metadata["gradient_multiplier_mean"] = step_ctx.gradient_multiplier_mean

                if step_ctx.gar_threshold_deg is not None:
                    metadata["gar_threshold_deg"] = step_ctx.gar_threshold_deg
                if step_ctx.gar_normalize is not None:
                    metadata["gar_normalize"] = step_ctx.gar_normalize

                if metadata:
                    step_dict["metadata"] = metadata

                entry["step"] = step_dict

            self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        if self._file and not self._file.closed:
            self._file.close()
