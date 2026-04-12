"""MLX generation backend for Apple Silicon.

Wraps mlx-lm's generate primitives behind the GenerationBackend protocol.
Supports sampler caching, stop sequence detection, logits processors,
quantized KV cache, and speculative decoding.
"""

from __future__ import annotations

from typing import Any

from .base import BackendRegistry, GenerateOutput, GenerateParams


def _check_stop(
    tokens: list[int],
    text: str,
    stop_seqs: list[str],
    tokenizer: Any,
) -> tuple[str, str | None, str, list[int]]:
    """Return (reason, seq, trimmed_text, trimmed_tokens) or ("", None, text, tokens)."""
    for stop in stop_seqs:
        if text.endswith(stop):
            trimmed = text[: -len(stop)]
            enc = tokenizer.encode(trimmed, add_special_tokens=False)
            return "stop_sequence", stop, trimmed, enc
    return "", None, text, tokens


class MLXBackend(BackendRegistry, name="mlx"):
    """MLX backend with quantized KV cache, logits processors, and speculative decoding."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        max_kv_size: int | None = None,
        draft_model: Any = None,
        prefill_step_size: int = 512,
        quantized_kv_start: int = 0,
        **kwargs: Any,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._sampler_cache: dict[tuple, Any] = {}
        self._kv_bits = kv_bits
        self._kv_group_size = kv_group_size
        self._max_kv_size = max_kv_size
        self._draft_model = draft_model
        self._prefill_step_size = prefill_step_size
        self._quantized_kv_start = quantized_kv_start
        self._stream_cb: Any = None  # Optional: (text_delta: str) -> None

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self._tokenizer, "eos_token_id", None)

    def _get_sampler(self, params: GenerateParams) -> Any:
        key = (
            params.temperature,
            params.top_p,
            params.min_p,
            params.top_k,
            params.xtc_probability,
            params.xtc_threshold,
            frozenset(params.xtc_special_tokens) if params.xtc_special_tokens else None,
            params.min_tokens_to_keep,
        )
        if key not in self._sampler_cache:
            try:
                from mlx_lm.sample_utils import make_sampler
            except ImportError:
                from mlx_lm.utils import make_sampler
            kw: dict[str, Any] = {
                "top_p": params.top_p,
                "min_p": params.min_p,
                "top_k": params.top_k,
                "min_tokens_to_keep": params.min_tokens_to_keep,
            }
            if params.xtc_probability > 0:
                kw["xtc_probability"] = params.xtc_probability
                kw["xtc_threshold"] = params.xtc_threshold
                if params.xtc_special_tokens is not None:
                    kw["xtc_special_tokens"] = params.xtc_special_tokens
            self._sampler_cache[key] = make_sampler(params.temperature, **kw)
        return self._sampler_cache[key]

    def _get_logits_processors(self, params: GenerateParams) -> Any:
        """Build logits processors for repetition penalty / logit bias. None if no-op."""
        has_rep = params.repetition_penalty > 1.0
        has_bias = params.logit_bias is not None
        if not has_rep and not has_bias:
            return None
        try:
            from mlx_lm.sample_utils import make_logits_processors
        except ImportError:
            return None
        kw: dict[str, Any] = {}
        if has_rep:
            kw["repetition_penalty"] = params.repetition_penalty
            kw["repetition_context_size"] = params.repetition_context_size
        if has_bias:
            kw["logit_bias"] = params.logit_bias
        return make_logits_processors(**kw)

    def generate(self, prefix_tokens: list[int], params: GenerateParams) -> GenerateOutput:
        if self._draft_model is not None:
            return self._generate_speculative(prefix_tokens, params)
        return self._generate_standard(prefix_tokens, params)

    def _generate_standard(
        self,
        prefix_tokens: list[int],
        params: GenerateParams,
    ) -> GenerateOutput:
        import mlx.core as mx

        try:
            from mlx_lm.generate import generate_step
        except ImportError:
            from mlx_lm.utils import generate_step

        sampler = self._get_sampler(params)
        lp = self._get_logits_processors(params)

        step_kw: dict[str, Any] = {"sampler": sampler, "max_tokens": params.max_tokens}
        if lp is not None:
            step_kw["logits_processors"] = lp
        if self._kv_bits is not None:
            step_kw["kv_bits"] = self._kv_bits
            step_kw["kv_group_size"] = self._kv_group_size
            if self._quantized_kv_start > 0:
                step_kw["quantized_kv_start"] = self._quantized_kv_start
        if self._max_kv_size is not None:
            step_kw["max_kv_size"] = self._max_kv_size
        if self._prefill_step_size != 512:
            step_kw["prefill_step_size"] = self._prefill_step_size

        # Use pre-populated prompt cache if available — skip cached tokens
        if params.prompt_cache is not None:
            cache_offset = params.prompt_cache[0].offset
            prompt = mx.array(prefix_tokens[cache_offset:])
            step_kw["prompt_cache"] = params.prompt_cache
        else:
            prompt = mx.array(prefix_tokens)

        gen_tokens: list[int] = []
        gen_logprobs: list[float] = []
        gen_text = ""
        stop_reason, stop_seq = "eos", None
        has_stops = bool(params.stop)
        chk = 0
        _do_decode = has_stops or self._stream_cb is not None
        _stream_prev = ""

        gen = generate_step(prompt, self._model, **step_kw)
        for token, token_lp in gen:
            tid = token if isinstance(token, int) else token.item()
            if tid == self._tokenizer.eos_token_id:
                break
            gen_tokens.append(tid)
            if token_lp is not None:
                gen_logprobs.append(float(token_lp[tid]))
            chk += 1
            if _do_decode and chk >= 4:
                chk = 0
                gen_text = self._tokenizer.decode(gen_tokens)
                if has_stops:
                    stop_reason, stop_seq, gen_text, gen_tokens = _check_stop(
                        gen_tokens,
                        gen_text,
                        params.stop,
                        self._tokenizer,
                    )
                    if stop_reason == "stop_sequence":
                        gen_logprobs = gen_logprobs[: len(gen_tokens)]
                        break
                if self._stream_cb and gen_text != _stream_prev:
                    self._stream_cb(gen_text[len(_stream_prev) :])
                    _stream_prev = gen_text
            if len(gen_tokens) >= params.max_tokens:
                stop_reason = "max_tokens"
                break

        # Close generator to release KV cache, then free Metal buffers
        del gen
        mx.clear_cache()

        # Post-loop stop check: catches sequences missed by the 4-token interval.
        # Covers three cases: (a) stop_reason="" — generator exhausted with stop
        # sequence inside text but not at end; (b) stop_reason="max_tokens" —
        # stop sequence was generated before the token limit but interval timing
        # missed it (e.g. </think> followed by more tokens in the same batch);
        # (c) stop_reason="eos" — generation completed naturally, but stop sequence
        # appears in the generated text.
        if has_stops and stop_reason in ("max_tokens", "", "eos"):
            _final_text = self._tokenizer.decode(gen_tokens)
            for _stop in params.stop:
                _idx = _final_text.find(_stop)
                if _idx != -1:
                    _trimmed = _final_text[:_idx]
                    gen_tokens = self._tokenizer.encode(_trimmed, add_special_tokens=False)
                    gen_logprobs = gen_logprobs[: len(gen_tokens)]
                    gen_text = _trimmed
                    stop_reason = "stop_sequence"
                    stop_seq = _stop
                    break

        if stop_reason != "stop_sequence":
            gen_text = self._tokenizer.decode(gen_tokens)
        return GenerateOutput(
            text=gen_text,
            tokens=gen_tokens,
            stop_reason=stop_reason,
            stop_sequence=stop_seq,
            logprobs=gen_logprobs or None,
        )

    def _generate_speculative(
        self,
        prefix_tokens: list[int],
        params: GenerateParams,
    ) -> GenerateOutput:
        import mlx.core as mx
        from mlx_lm import stream_generate

        sampler = self._get_sampler(params)
        lp = self._get_logits_processors(params)
        prompt_text = self._tokenizer.decode(prefix_tokens)

        gen_kw: dict[str, Any] = {
            "model": self._model,
            "tokenizer": self._tokenizer,
            "prompt": prompt_text,
            "max_tokens": params.max_tokens,
            "sampler": sampler,
            "draft_model": self._draft_model,
        }
        if lp is not None:
            gen_kw["logits_processors"] = lp

        gen_tokens: list[int] = []
        gen_text = ""
        stop_reason, stop_seq = "eos", None
        has_stops = bool(params.stop)
        chk = 0
        _do_decode = has_stops or self._stream_cb is not None
        _stream_prev = ""

        for response in stream_generate(**gen_kw):
            tid = response.token
            if tid == self._tokenizer.eos_token_id:
                break
            gen_tokens.append(tid)
            chk += 1
            if _do_decode and chk >= 4:
                chk = 0
                gen_text = self._tokenizer.decode(gen_tokens)
                if has_stops:
                    stop_reason, stop_seq, gen_text, gen_tokens = _check_stop(
                        gen_tokens,
                        gen_text,
                        params.stop,
                        self._tokenizer,
                    )
                    if stop_reason == "stop_sequence":
                        break
                if self._stream_cb and gen_text != _stream_prev:
                    self._stream_cb(gen_text[len(_stream_prev) :])
                    _stream_prev = gen_text
            if len(gen_tokens) >= params.max_tokens:
                stop_reason = "max_tokens"
                break

        # Post-loop stop check: same interval-miss fix as _generate_standard.
        # Also check "eos" case where generation completed naturally.
        if has_stops and stop_reason in ("max_tokens", "", "eos"):
            _final_text = self._tokenizer.decode(gen_tokens)
            for _stop in params.stop:
                _idx = _final_text.find(_stop)
                if _idx != -1:
                    _trimmed = _final_text[:_idx]
                    gen_tokens = self._tokenizer.encode(_trimmed, add_special_tokens=False)
                    gen_text = _trimmed
                    stop_reason = "stop_sequence"
                    stop_seq = _stop
                    break

        if stop_reason != "stop_sequence":
            gen_text = self._tokenizer.decode(gen_tokens)
        mx.clear_cache()
        return GenerateOutput(
            text=gen_text,
            tokens=gen_tokens,
            stop_reason=stop_reason,
            stop_sequence=stop_seq,
        )

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
            **{k: v for k, v in kwargs.items() if k != "add_generation_prompt"},
        )
