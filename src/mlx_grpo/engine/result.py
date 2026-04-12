"""Rollout result types — the atoms of the template-driven framework.

Every rollout produces a RolloutResult: an ordered list of Segments,
each tagged with its source and masking. Loss computation, reward
functions, and logging all consume this single type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SegmentSource(StrEnum):
    """How a segment was produced."""

    GENERATED = "generated"  # Model output — train on this
    INJECTED = "injected"  # Forced text  — mask by default
    PROMPT = "prompt"  # Chat prefix  — always masked


@dataclass(slots=True)
class GenerateResult:
    """Result of a single {% generate %} call, captured via `as` clause.

    Behaves like a string in template expressions so you can write:
        if "</think>" in thinking:
        if thinking.stop_reason == "max_tokens":
        {{ thinking }}
    """

    text: str = ""
    tokens: list[int] = field(default_factory=list)
    stop_reason: str = "eos"  # "stop_sequence" | "max_tokens" | "eos"
    stop_sequence: str | None = None
    valid: bool = True  # For generate_if: did validator pass?
    logprobs: list[float] = field(default_factory=list)  # Per-token log probs

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    # ── String-like protocol ──
    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"GenerateResult({self.text!r})"

    def __contains__(self, s: str):
        return s in self.text  # type: ignore[override]

    def __len__(self) -> int:
        return len(self.text)

    def endswith(self, s: str) -> bool:
        return self.text.endswith(s)

    def startswith(self, s: str) -> bool:
        return self.text.startswith(s)


@dataclass(slots=True)
class Segment:
    """One contiguous piece of a rollout completion."""

    tokens: list[int]
    text: str
    source: SegmentSource
    masked: bool  # True = exclude from loss
    tag: str = ""  # Named tag: "thinking", "answer", etc.
    metadata: dict[str, Any] = field(default_factory=dict)
    loss_weight: float = 1.0  # Per-segment loss multiplier (>1 = upweight, 0 = masked)
    modality: str | None = None  # "vision" = no gradients, "text" = train, None = apply to all


@dataclass(slots=True)
class RolloutResult:
    """Complete result of executing a rollout template.

    This is the ONLY type that flows from engine → rewards → loss.
    Everything downstream reads this; nothing upstream knows about it.
    """

    segments: list[Segment] = field(default_factory=list)
    variables: dict[str, GenerateResult] = field(default_factory=dict)
    type_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    log_probs: list[float] = field(default_factory=list)  # Per-token log probs (policy)
    ref_log_probs: list[float] = field(default_factory=list)  # Per-token log probs (ref)
    prompt_tokens: list[int] = field(default_factory=list)  # Original prompt token IDs
    recovered: bool = False  # True if two-phase recovery was triggered
    system_message: str = ""  # System prompt used
    prompt_text: str = ""  # Raw text after chat template
    prompt_mask_strategy: str = "mask"  # "mask" = exclude prompt, "no_mask" = include prompt
    system_prompt_token_count: int = 0  # Number of system prompt tokens in prompt segment
    mask_system_prompt: bool = False  # exclude system prompt tokens from loss
    mask_user_prompt: bool = False  # exclude user prompt tokens from loss
    gar_activation_drift: dict[str, dict[str, Any]] | None = None  # per-layer GAR metrics

    # ── Derived properties ──

    @property
    def completion_text(self) -> str:
        return "".join(s.text for s in self.segments if s.source != SegmentSource.PROMPT)

    @property
    def generated_text(self) -> str:
        """Only model-generated text — excludes injected scaffold/prefill."""
        return "".join(s.text for s in self.segments if s.source == SegmentSource.GENERATED)

    @property
    def completion_tokens(self) -> list[int]:
        return [t for s in self.segments if s.source != SegmentSource.PROMPT for t in s.tokens]

    @property
    def loss_mask(self) -> list[float]:
        """Per-token weight: 0.0 = masked, >0 = train (supports weights > 1.0).

        Tokens with modality='vision' are always masked (0.0) to prevent
        backprop through vision embeddings. Only text modality receives gradients.
        Prompt masking is controlled by prompt_mask_strategy config and per-flag
        masks (mask_system_prompt, mask_user_prompt) for fine-grained control.
        """
        weights = []
        for s in self.segments:
            is_prompt = s.source == SegmentSource.PROMPT
            is_vision = s.modality == "vision"

            if is_prompt:
                # Split PROMPT segment into system and user parts
                system_token_count = self.system_prompt_token_count
                segment_weights = []

                for i, _token in enumerate(s.tokens):
                    # Determine which part this token belongs to
                    is_system_token = i < system_token_count
                    is_user_token = i >= system_token_count

                    # Check if we should mask this token
                    should_mask = False
                    if is_vision:
                        should_mask = True
                    elif self.prompt_mask_strategy == "mask":
                        # Legacy strategy: mask entire prompt
                        should_mask = True
                    elif (
                        is_system_token
                        and self.mask_system_prompt
                        or is_user_token
                        and self.mask_user_prompt
                    ):
                        should_mask = True

                    segment_weights.append(0.0 if should_mask else s.loss_weight)

                weights.extend(segment_weights)
            else:
                # Non-prompt segments
                should_mask_segment = is_vision or (s.masked and s.source != SegmentSource.PROMPT)
                w = 0.0 if should_mask_segment else s.loss_weight
                weights.extend([w] * len(s.tokens))

        return weights

    @property
    def generated_token_count(self) -> int:
        return sum(len(s.tokens) for s in self.segments if s.source == SegmentSource.GENERATED)

    @property
    def injected_token_count(self) -> int:
        return sum(len(s.tokens) for s in self.segments if s.source == SegmentSource.INJECTED)

    @property
    def prompt_token_count(self) -> int:
        return sum(len(s.tokens) for s in self.segments if s.source == SegmentSource.PROMPT)

    def tagged_segments(self, tag: str) -> list[Segment]:
        """Get all segments with a specific tag."""
        return [s for s in self.segments if s.tag == tag]

    def segment_text(self, tag: str) -> str:
        """Concatenated text of all segments with the given tag."""
        return "".join(s.text for s in self.segments if s.tag == tag)

    def to_reward_inputs(self) -> dict[str, Any]:
        """Everything a reward function needs."""
        return {
            "completion": self.completion_text,
            "variables": {k: v.text for k, v in self.variables.items()},
            "metadata": self.metadata,
            "type_name": self.type_name,
        }

    @property
    def extracted_answer(self) -> str:
        """Best-effort answer extraction from captured variables or generated text.

        Checks: (1) captured "answer" variable (cleaned), (2) \\boxed{} in
        generated-only text, (3) last non-empty generated line.
        Only looks at model-generated text — never scaffold/injected text.
        """
        import re

        av = self.variables.get("answer")
        if av and av.text.strip():
            raw = av.text.strip()
            # Clean trailing } from \boxed{ closure and leading/trailing punctuation
            m = re.match(
                r"\s*(?:\d+[.)\s]*)?\*{0,2}\(?([A-Ha-h])\)?\*{0,2}(?:[\s.,;:)\-}]|$)",
                raw,
            )
            if m:
                return m.group(1).upper()
            return raw.rstrip("}")
        # Fallback: only search model-generated text (not scaffold)
        gen = self.generated_text
        boxed = re.findall(r"\\boxed\{([^}]*)\}", gen)
        if boxed:
            return boxed[-1].strip()
        lines = [ln.strip() for ln in gen.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    # Metadata keys that are gradient-computation artifacts — never written to JSONL.
    # _esc_weights: per-token float array (hundreds of values per rollout).
    # _is_override, _loss_overrides: config/dict objects only needed at grad time.
    _LOG_EXCLUDE_META: frozenset[str] = frozenset(
        {"_esc_weights", "_is_override", "_loss_overrides"}
    )

    def to_log_entry(self, include_tokens: bool = True, decode_fn: Any = None) -> dict[str, Any]:
        """Full representation for rollout logging.

        Top-level fields are the human-readable essentials:
          generated_completion, generated_answer, token_counts
        Segment and variable detail is nested for drill-down.

        Args:
            include_tokens: When False, omit token ID arrays from segment
                entries to reduce log file size (token_count is always kept).
            decode_fn: Optional function to decode tokens (e.g., backend.decode).
                If provided, text_used_for_loss will be exact decoded text of
                trainable tokens. Otherwise, it's approximated from segments.
        """
        segments_detail = []
        accumulated = ""  # running completion text for injection context
        for s in self.segments:
            entry: dict[str, Any] = {
                "source": str(s.source),
                "tag": s.tag,
                "text": s.text,
                "token_count": len(s.tokens),
                "masked": s.masked,
            }
            if include_tokens:
                entry["tokens"] = s.tokens
            if s.source == SegmentSource.INJECTED:
                # Last 200 chars of accumulated text before this injection
                entry["context_before"] = accumulated[-200:] if accumulated else ""
            accumulated += s.text
            segments_detail.append(entry)

        variables_detail = {}
        for k, v in self.variables.items():
            variables_detail[k] = {
                "text": v.text,
                "token_count": v.token_count,
                "stop_reason": v.stop_reason,
                "stop_sequence": v.stop_sequence,
            }

        mask = self.loss_mask
        # Exact tokens used for backprop — get all tokens and their loss_mask
        loss_mask = self.loss_mask
        all_tokens = [t for s in self.segments for t in s.tokens]
        trainable_tokens = [all_tokens[i] for i, w in enumerate(loss_mask) if w > 0]

        # Decode text_used_for_loss: exact tokens where loss_mask > 0
        if decode_fn is not None and trainable_tokens:
            text_used_for_loss = decode_fn(trainable_tokens)
        else:
            # Fallback: Reconstruct text from trainable tokens only.
            # This respects token-level masking (mask_system_prompt, mask_user_prompt)
            # by including only segments/tokens where loss_mask > 0.
            # For prompts split into system/user parts, this means we include
            # only the parts that are actually trainable (not masked).
            text_parts = []
            token_idx = 0
            for s in self.segments:
                segment_tokens = s.tokens
                segment_mask = loss_mask[token_idx : token_idx + len(segment_tokens)]
                token_idx += len(segment_tokens)

                # For segments with token-level masking, reconstruct only trainable parts
                trainable_segment_tokens = [
                    segment_tokens[i] for i in range(len(segment_tokens)) if segment_mask[i] > 0
                ]

                if trainable_segment_tokens:
                    # Decode just the trainable tokens from this segment
                    if decode_fn is not None:
                        segment_text = decode_fn(trainable_segment_tokens)
                    else:
                        # No decoder available — use full segment text as approximation
                        # This is imperfect but better than nothing
                        segment_text = s.text
                    text_parts.append(segment_text)

            text_used_for_loss = "".join(text_parts)

        base: dict[str, Any] = {
            "type": self.type_name,
            "system_message": self.system_message,
            "prompt_text": self.prompt_text,
            "generated_completion": self.completion_text,
            "generated_answer": self.extracted_answer,
            "text_used_for_loss": text_used_for_loss,
            "segments": segments_detail,
            "variables": variables_detail,
            "token_counts": {
                "prompt": self.prompt_token_count,
                "generated": self.generated_token_count,
                "injected": self.injected_token_count,
                "total_completion": len(self.completion_tokens),
                "trainable": sum(mask),
                "masked": len(mask) - sum(mask),
            },
            "recovered": self.recovered,
            "metadata": {k: v for k, v in self.metadata.items() if k not in self._LOG_EXCLUDE_META},
        }
        if self.gar_activation_drift is not None:
            base["gar_mode"] = True
            base["gar_activation_drift"] = self.gar_activation_drift
        else:
            base["gar_mode"] = False
        return base
