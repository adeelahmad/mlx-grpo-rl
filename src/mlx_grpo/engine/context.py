"""Rollout context — the API surface for rollout templates.

Every rollout function receives a ``RolloutContext``. It provides:
- ``ctx.generate()``  — ask the model to produce tokens
- ``ctx.inject()``    — force literal text into the completion
- ``ctx.prompt``      — the input prompt
- ``ctx.answer``      — ground truth (if available)
- ``ctx.batch``       — training loop state (iteration, progress, etc.)
- ``ctx.callbacks``   — Python helper functions

The context accumulates Segments internally and builds a RolloutResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.base import GenerateParams, GenerationBackend
from ..scaffold import ScaffoldConfig
from .callbacks import CallbackRegistry
from .result import GenerateResult, RolloutResult, Segment, SegmentSource


@dataclass(slots=True, frozen=True)
class BatchContext:
    """Read-only training loop state, available as ``ctx.batch``."""

    iteration: int = 0
    epoch: int = 0
    total_iters: int = 1
    global_step: int = 0
    group_index: int = 0
    group_size: int = 1
    batch_index: int = 0
    dataset_name: str = ""
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    custom: dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Training progress: 0.0 → 1.0."""
        return self.iteration / max(self.total_iters, 1)

    @property
    def is_first_group(self) -> bool:
        return self.group_index == 0

    @property
    def scaffold_ratio(self) -> float:
        """Current scaffold ratio — delegates to ScaffoldConfig."""
        return self.scaffold.effective_ratio(self.progress, self.group_index)

    @property
    def scaffold_thinking_ratio(self) -> float:
        """Scaffold ratio for thinking segments (0.0 if target is answer-only)."""
        return self.scaffold_ratio * self.scaffold.thinking_ratio_factor

    @property
    def scaffold_answer_ratio(self) -> float:
        """Scaffold ratio for answer segments (0.0 if target is thinking-only)."""
        return self.scaffold_ratio * self.scaffold.answer_ratio_factor

    @property
    def is_last_group(self) -> bool:
        return self.group_index == self.group_size - 1


class RolloutContext:
    """Mutable execution context for a single rollout.

    Template functions call methods on this to build the completion::

        def my_rollout(ctx: RolloutContext) -> None:
            thinking = ctx.generate(max_tokens=512, stop=["</think>"])
            if thinking.stop_reason == "max_tokens":
                ctx.inject("</think>", masked=True)
    """

    __slots__ = (
        "prompt",
        "answer",
        "type_name",
        "type_config",
        "batch",
        "callbacks",
        "sample",
        "short_circuited",
        "_backend",
        "_prefix_tokens",
        "_segments",
        "_variables",
        "_metadata",
        "_original_prompt_tokens",
        "_prompt_cache",
        "_images",
        "recovered",
        "recovery_enabled",
        "max_recovery_attempts",
        "max_completion_tokens",
        "_gen_defaults",
        "_batch_collector",
    )

    def __init__(
        self,
        backend: GenerationBackend,
        prompt: str,
        prompt_tokens: list[int],
        answer: str | None = None,
        type_name: str = "",
        type_config: dict[str, Any] | None = None,
        batch: BatchContext | None = None,
        callbacks: CallbackRegistry | None = None,
        sample: dict[str, Any] | None = None,
        recovery_enabled: bool = True,
        max_recovery_attempts: int = 1,
        max_completion_tokens: int = 0,
        prompt_cache: Any = None,
        gen_defaults: dict[str, Any] | None = None,
        *,
        images: list[str | bytes] | None = None,
    ):
        self.prompt = prompt
        self.answer = answer
        self.type_name = type_name
        self.type_config = type_config or {}
        self.batch = batch or BatchContext()
        self.callbacks = callbacks or CallbackRegistry()
        self.sample = sample or {}
        self.short_circuited: bool = False
        self._backend = backend
        self._prefix_tokens = list(prompt_tokens)
        self._original_prompt_tokens = list(prompt_tokens)
        self._segments: list[Segment] = []
        self._variables: dict[str, GenerateResult] = {}
        self._metadata: dict[str, Any] = {}
        self._prompt_cache = prompt_cache
        self._images: list[str | bytes] | None = images
        self.recovered: bool = False
        self.recovery_enabled: bool = recovery_enabled
        self.max_recovery_attempts: int = max_recovery_attempts
        self.max_completion_tokens: int = max_completion_tokens
        self._gen_defaults: dict[str, Any] = gen_defaults or {}
        self._batch_collector: Any = None  # Optional BatchCollector for batched gen

    # ── Core API ──

    def generate(
        self,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        repetition_context_size: int | None = None,
        logit_bias: dict[int, float] | None = None,
        xtc_probability: float | None = None,
        xtc_threshold: float | None = None,
        xtc_special_tokens: set[int] | None = None,
        min_tokens_to_keep: int | None = None,
        tag: str = "",
        masked: bool = False,
        on_token: Any = None,
        *,
        images: list[str | bytes] | None = None,
    ) -> GenerateResult:
        """Generate tokens from the model. Returns a GenerateResult.

        When recovery is enabled and a 'thinking' segment hits max_tokens
        without producing '</think>', the result is marked as needing
        recovery. The calling rollout template can then inject the closing
        tag and generate an answer segment.
        """
        if self.short_circuited:
            empty = GenerateResult(
                text="",
                tokens=[],
                stop_reason="short_circuited",
            )
            self._add_segment([], "", SegmentSource.GENERATED, masked=True, tag=tag)
            return empty

        # Resolve None params: gen_defaults (from GenerationConfig) → hardcoded fallback.
        # Explicit caller args (non-None) win; unset args fall through to generation config.
        _gd = self._gen_defaults
        temperature = _gd.get("temperature", 0.7) if temperature is None else temperature
        top_p = _gd.get("top_p", 0.95) if top_p is None else top_p
        top_k = _gd.get("top_k", 20) if top_k is None else top_k
        min_p = _gd.get("min_p", 0.0) if min_p is None else min_p
        repetition_penalty = (
            _gd.get("repetition_penalty", 1.0) if repetition_penalty is None else repetition_penalty
        )
        repetition_context_size = (
            _gd.get("repetition_context_size", 20)
            if repetition_context_size is None
            else repetition_context_size
        )
        xtc_probability = (
            _gd.get("xtc_probability", 0.0) if xtc_probability is None else xtc_probability
        )
        xtc_threshold = _gd.get("xtc_threshold", 0.1) if xtc_threshold is None else xtc_threshold
        min_tokens_to_keep = (
            _gd.get("min_tokens_to_keep", 1) if min_tokens_to_keep is None else min_tokens_to_keep
        )

        # Auto-merge segment-level overrides from type_config["segments"][tag]
        if tag:
            seg_cfg = self.type_config.get("segments", {}).get(tag, {})
            if seg_cfg:
                max_tokens = seg_cfg.get("max_tokens", max_tokens)
                stop = seg_cfg.get("stop", stop)
                temperature = seg_cfg.get("temperature", temperature)
                top_p = seg_cfg.get("top_p", top_p)
                top_k = seg_cfg.get("top_k", top_k)
                min_p = seg_cfg.get("min_p", min_p)
                repetition_penalty = seg_cfg.get("repetition_penalty", repetition_penalty)
                repetition_context_size = seg_cfg.get(
                    "repetition_context_size",
                    repetition_context_size,
                )
                xtc_probability = seg_cfg.get("xtc_probability", xtc_probability)
                xtc_threshold = seg_cfg.get("xtc_threshold", xtc_threshold)
                min_tokens_to_keep = seg_cfg.get("min_tokens_to_keep", min_tokens_to_keep)
                if "logit_bias" in seg_cfg:
                    logit_bias = seg_cfg["logit_bias"]
                if "xtc_special_tokens" in seg_cfg:
                    xtc_special_tokens = set(seg_cfg["xtc_special_tokens"])

        # Cap at global max_completion_tokens if set (prevents Metal timeout
        # from types requesting more tokens than the hardware can handle).
        # Only count GENERATED tokens — injected scaffold/prefill doesn't
        # consume the generation budget.
        if self.max_completion_tokens > 0:
            already_generated = sum(
                len(s.tokens) for s in self._segments if s.source == SegmentSource.GENERATED
            )
            remaining = max(1, self.max_completion_tokens - already_generated)
            max_tokens = min(max_tokens, remaining)

        # Clone prompt cache for this generate call (original stays immutable)
        cloned_cache = None
        if self._prompt_cache is not None:
            from ..backends.cache import clone_prompt_cache

            cloned_cache = clone_prompt_cache(self._prompt_cache)

        effective_images = images if images is not None else self._images
        # Safety: always stop on chat template special tokens to prevent
        # base-model completion mode (hallucinated multi-turn conversations).
        # "<|" catches ALL special tokens: <|im_end|>, <|im_start|>, <|system|>,
        # <|end_of_text|>, <|eot_id|>, etc.
        _stop = list(stop) if stop else []
        for _ss in ("<|",):
            if _ss not in _stop:
                _stop.append(_ss)
        base_params = GenerateParams(
            max_tokens=max_tokens,
            stop=_stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            logit_bias=logit_bias,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            xtc_special_tokens=xtc_special_tokens,
            min_tokens_to_keep=min_tokens_to_keep,
            prompt_cache=cloned_cache,
            images=effective_images,
        )

        # Chunked generation: respect generation.max_tokens as a per-call Metal-safe
        # cap. Loop until the caller's budget is exhausted or a stop sequence fires.
        # Each chunk is at most chunk_size tokens → no single Metal command exceeds
        # the watchdog limit regardless of how large thinking_budget is.
        chunk_size = _gd.get("max_tokens", 0)
        if chunk_size > 0 and chunk_size < max_tokens:
            from dataclasses import replace as _dc_replace  # noqa: PLC0415

            from ..backends.base import GenerateOutput as _GenOut  # noqa: PLC0415,N814

            acc_tokens: list[int] = []
            acc_text = ""
            final_stop_reason = "max_tokens"
            final_stop_seq: str | None = None
            remaining = max_tokens
            first = True
            while remaining > 0:
                chunk_params = _dc_replace(
                    base_params,
                    max_tokens=min(chunk_size, remaining),
                    prompt_cache=cloned_cache if first else None,
                )
                prefix = self._prefix_tokens + acc_tokens
                chunk_out = self._backend.generate(prefix, chunk_params)
                acc_tokens.extend(chunk_out.tokens)
                acc_text += chunk_out.text
                remaining -= len(chunk_out.tokens)
                first = False
                if chunk_out.stop_reason != "max_tokens" or not chunk_out.tokens:
                    final_stop_reason = chunk_out.stop_reason
                    final_stop_seq = chunk_out.stop_sequence
                    break
            output: _GenOut = _GenOut(
                text=acc_text,
                tokens=acc_tokens,
                stop_reason=final_stop_reason,
                stop_sequence=final_stop_seq,
                logprobs=None,
            )
        elif self._batch_collector is not None:
            output = self._batch_collector.submit(
                self.batch.group_index, self._prefix_tokens, base_params
            )
        else:
            output = self._backend.generate(self._prefix_tokens, base_params)

        # Post-generation safety: truncate at "<|" if the stop check missed it.
        # The model may generate partial special tokens between check intervals.
        _out_text = output.text
        _out_tokens = output.tokens
        _pipe_idx = _out_text.find("<|")
        if _pipe_idx >= 0:
            _out_text = _out_text[:_pipe_idx]
            _out_tokens = self._backend.encode(_out_text) if _out_text else []
            output = type(output)(
                text=_out_text,
                tokens=_out_tokens,
                stop_reason="stop_sequence",
                stop_sequence="<|",
                logprobs=output.logprobs,
            )

        result = GenerateResult(
            text=output.text,
            tokens=output.tokens,
            stop_reason=output.stop_reason,
            stop_sequence=output.stop_sequence,
            logprobs=output.logprobs if output.logprobs is not None else [],
        )

        # Post-hoc token callback — check generated output for degenerate patterns
        if on_token is not None and result.tokens:
            stop_signal = on_token(result.tokens, result.logprobs)
            if stop_signal is not None:
                result.stop_reason = stop_signal
                self._metadata["on_token_triggered"] = stop_signal

        self._add_segment(
            output.tokens,
            output.text,
            SegmentSource.GENERATED,
            masked=masked,
            tag=tag,
        )

        # Two-phase recovery detection: if a thinking segment hit max_tokens
        # without the model producing '</think>', flag for recovery
        if (
            self.recovery_enabled
            and tag == "thinking"
            and output.stop_reason == "max_tokens"
            and "</think>" not in output.text
        ):
            self._metadata["needs_recovery"] = True

        return result

    def inject(
        self, text: str, masked: bool = True, tag: str = "", loss_weight: float = 1.0
    ) -> None:
        """Inject literal text into the completion.

        loss_weight: per-segment loss multiplier applied when masked=False.
        Values > 1.0 upweight these tokens in the loss (e.g. 1.1 for anchor).
        Has no effect when masked=True (weight is always 0 for masked tokens).

        If recovery is pending (thinking hit max_tokens without '</think>'),
        injecting text marks this rollout as recovered.
        """
        if self._metadata.get("needs_recovery"):
            self.recovered = True
            self._metadata["needs_recovery"] = False

        tokens = self._backend.encode(text)
        self._add_segment(
            tokens,
            text,
            SegmentSource.INJECTED,
            masked=masked,
            tag=tag,
            loss_weight=loss_weight,
        )

    def encode(self, text: str) -> list[int]:
        """Encode text to tokens (delegates to backend)."""
        return self._backend.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text (delegates to backend)."""
        return self._backend.decode(tokens)

    def mark_vision_tokens(self, n: int) -> None:
        """Mark the last n tokens of the last segment as vision modality.

        Vision tokens (from image encoder) should not receive gradients.
        This is typically called after a generation that includes image encoding.

        Args:
            n: Number of tokens from the end of the last segment to mark as vision.
        """
        if not self._segments or n <= 0:
            return
        last_seg = self._segments[-1]
        if len(last_seg.tokens) <= n:
            # Entire segment is vision — replace modality
            self._segments[-1] = Segment(
                tokens=last_seg.tokens,
                text=last_seg.text,
                source=last_seg.source,
                masked=last_seg.masked,
                tag=last_seg.tag,
                metadata=last_seg.metadata,
                loss_weight=last_seg.loss_weight,
                modality="vision",
            )
        else:
            # Split segment: text tokens stay, vision tokens tagged separately
            text_tokens = last_seg.tokens[:-n]
            vision_tokens = last_seg.tokens[-n:]
            text_text = self._backend.decode(text_tokens)
            vision_text = self._backend.decode(vision_tokens)
            # Replace with text-only segment
            self._segments[-1] = Segment(
                tokens=text_tokens,
                text=text_text,
                source=last_seg.source,
                masked=last_seg.masked,
                tag=last_seg.tag,
                metadata=last_seg.metadata,
                loss_weight=last_seg.loss_weight,
                modality=None,
            )
            # Append vision segment
            self._segments.append(
                Segment(
                    tokens=vision_tokens,
                    text=vision_text,
                    source=last_seg.source,
                    masked=last_seg.masked,
                    tag=last_seg.tag,
                    metadata={},
                    loss_weight=last_seg.loss_weight,
                    modality="vision",
                )
            )

    def generate_until(
        self,
        endswith: str,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> GenerateResult:
        """Generate until output ends with a pattern."""
        return self.generate(max_tokens=max_tokens, stop=[endswith], **kwargs)

    def generate_if(
        self,
        validator: Any,
        max_retries: int = 3,
        max_tokens: int = 128,
        **kwargs: Any,
    ) -> GenerateResult:
        """Generate with validation + retry."""
        for attempt in range(max_retries):
            result = self.generate(max_tokens=max_tokens, **kwargs)
            if validator(result.text):
                result.valid = True
                return result
            if attempt < max_retries - 1:
                self.inject(f"\n# Retry {attempt + 2}:\n", masked=True)
        result.valid = False
        return result

    # ── Variable capture ──

    def capture(self, name: str, result: GenerateResult) -> None:
        """Store a named variable (accessible later in the rollout)."""
        self._variables[name] = result

    def var(self, name: str) -> GenerateResult | None:
        """Retrieve a previously captured variable."""
        return self._variables.get(name)

    # ── Segment config ──

    def seg(self, tag: str) -> dict[str, Any]:
        """Return type_config merged with segment-level overrides for *tag*.

        Precedence: ``type_config`` flat keys → ``segments.{tag}`` overrides.
        Builtins can use ``cfg = ctx.seg("thinking")`` instead of
        ``cfg = ctx.type_config`` to pick up per-segment config from YAML.
        """
        base = {k: v for k, v in self.type_config.items() if k != "segments"}
        seg_cfg = self.type_config.get("segments", {}).get(tag, {})
        return {**base, **seg_cfg} if seg_cfg else base

    # ── Metadata ──

    def set_meta(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    # ── Build final result ──

    def build_result(self) -> RolloutResult:
        if self.short_circuited:
            self._metadata["short_circuited"] = True
        prompt_mask_strategy = self.type_config.get("prompt_mask_strategy", "mask")
        # Prepend a PROMPT segment so loss_mask can include/exclude prompt tokens.
        # Without this, prompt tokens exist in prompt_tokens but are invisible to loss.
        prompt_toks = list(self._original_prompt_tokens)
        prompt_text = self._backend.decode(prompt_toks) if prompt_toks else ""
        prompt_seg = Segment(
            tokens=prompt_toks,
            text=prompt_text,
            source=SegmentSource.PROMPT,
            masked=False,  # loss_mask handles masking via prompt_mask_strategy
            tag="prompt",
        )
        all_segments = [prompt_seg] + list(self._segments)

        # Optional EOS injection: append end-of-sequence token so model learns
        # when to stop generating. Controlled by type_config["inject_eos"].
        if self.type_config.get("inject_eos", False):
            eos_id = getattr(self._backend, "eos_token_id", None)
            if eos_id is not None:
                eos_text = (
                    self._backend.decode([eos_id]) if hasattr(self._backend, "decode") else ""
                )
                eos_masked = self.type_config.get("inject_eos_masked", False)
                all_segments.append(
                    Segment(
                        tokens=[eos_id],
                        text=eos_text,
                        source=SegmentSource.INJECTED,
                        masked=eos_masked,
                        tag="eos",
                    )
                )

        return RolloutResult(
            segments=all_segments,
            variables=dict(self._variables),
            type_name=self.type_name,
            metadata=dict(self._metadata),
            prompt_tokens=prompt_toks,
            recovered=self.recovered,
            prompt_mask_strategy=prompt_mask_strategy,
        )

    # ── Internal ──

    def _add_segment(
        self,
        tokens: list[int],
        text: str,
        source: SegmentSource,
        masked: bool,
        tag: str = "",
        loss_weight: float = 1.0,
        modality: str | None = None,
    ) -> None:
        self._segments.append(
            Segment(
                tokens=tokens,
                text=text,
                source=source,
                masked=masked,
                tag=tag,
                loss_weight=loss_weight,
                modality=modality,
            )
        )
        self._prefix_tokens.extend(tokens)


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: None
# Usage: Provided to rollout functions as ctx; call ctx.generate(), ctx.inject()
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
