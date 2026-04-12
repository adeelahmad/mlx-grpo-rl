"""MLX batch_generate backend for high-throughput generation.

Uses mlx_lm's BatchGenerator directly for efficient batched inference with
proper token-level stop sequence support via Aho-Corasick trie.

Limitations vs mlx_backend:
  - Single temperature/top_k/top_p across entire batch
  - No speculative decoding, logit processors, KV quantization
  - Best when all sequences have identical generation parameters

Best for: Evaluation, synthetic data generation (uniform parameters)
"""

from __future__ import annotations

from typing import Any

from .base import BackendRegistry, GenerateOutput, GenerateParams


def _build_sampler(params: GenerateParams) -> Any:
    """Build mlx_lm sampler from GenerateParams using make_sampler."""
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        return None
    return make_sampler(
        temp=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        min_p=params.min_p,
        min_tokens_to_keep=1,
    )


class MLXBatchGenerateBackend(BackendRegistry, name="mlx_batch_generate"):
    """MLX backend using BatchGenerator for parallel generation with stop support.

    Unlike the convenience batch_generate() wrapper, this uses BatchGenerator
    directly to support custom stop sequences (tokenized) alongside eos tokens.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        verbose: bool = False,
        **kwargs: Any,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._verbose = verbose

    def _ensure_tokenizer(self) -> None:
        """Raise helpful error if tokenizer is missing."""
        if self._tokenizer is None:
            raise RuntimeError(
                "MLXBatchGenerateBackend requires tokenizer. "
                "Pass model= and tokenizer= at init, or let training "
                "framework inject via backend._tokenizer = tokenizer."
            )

    @property
    def name(self) -> str:
        return "mlx_batch_generate"

    @property
    def eos_token_id(self) -> int | None:
        """EOS token ID from tokenizer, used for optional EOS injection."""
        return getattr(self._tokenizer, "eos_token_id", None)

    def _tokenize_stops(self, params: GenerateParams) -> list[list[int]]:
        """Tokenize stop sequences + eos tokens → token ID sequences.

        Returns list of token sequences for BatchGenerator's SequenceStateMachine.
        Sequences that fail to tokenize are skipped (string fallback catches them).
        """
        stop_tokens: list[list[int]] = []
        # Add eos tokens first
        eos_ids = getattr(self._tokenizer, "eos_token_ids", [])
        if not eos_ids:
            eid = getattr(self._tokenizer, "eos_token_id", None)
            eos_ids = [eid] if eid is not None else []
        for eid in eos_ids:
            stop_tokens.append([eid])
        # Tokenize string stop sequences
        for seq in params.stop:
            if not seq:
                continue
            try:
                toks = self._tokenizer.encode(seq, add_special_tokens=False)
                if toks:
                    stop_tokens.append(toks)
            except Exception:
                pass  # String-based post-loop fallback will catch these
        return stop_tokens

    @staticmethod
    def _make_generator(model: Any, **kwargs: Any) -> Any:
        """Create a BatchGenerator. Separate method for test mockability."""
        from mlx_lm.generate import BatchGenerator

        return BatchGenerator(model, **kwargs)

    @staticmethod
    def _clear_cache() -> None:
        """Clear MLX cache. Separate method for test mockability."""
        import mlx.core as mx

        mx.clear_cache()

    def generate(self, prefix_tokens: list[int], params: GenerateParams) -> GenerateOutput:
        """Generate using BatchGenerator with single-element batch."""
        results = self.generate_batch([prefix_tokens], params)
        return results[0] if results else GenerateOutput(text="", tokens=[])

    def generate_batch(
        self,
        prefix_tokens_batch: list[list[int]],
        params: GenerateParams,
    ) -> list[GenerateOutput]:
        """Generate from multiple prefixes using BatchGenerator.

        Uses BatchGenerator directly (not batch_generate wrapper) so we can
        pass custom stop token sequences alongside eos tokens.
        """
        self._ensure_tokenizer()
        if not prefix_tokens_batch:
            return []

        sampler = _build_sampler(params)
        stop_tokens = self._tokenize_stops(params)

        gen = self._make_generator(
            self._model,
            max_tokens=params.max_tokens,
            stop_tokens=stop_tokens if stop_tokens else None,
            sampler=sampler,
        )
        max_toks = [params.max_tokens] * len(prefix_tokens_batch)
        uids = gen.insert(prefix_tokens_batch, max_tokens=max_toks)

        collected: dict[int, list[int]] = {uid: [] for uid in uids}
        finish_info: dict[int, Any] = {}

        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    finish_info[r.uid] = r
                # On "stop", the matching token is NOT appended (already excluded)
                if r.finish_reason != "stop":
                    collected[r.uid].append(r.token)
        gen.close()

        # Build eos set for distinguishing eos vs custom stop
        eos_set = set(getattr(self._tokenizer, "eos_token_ids", []))
        has_str_stops = bool(params.stop)
        outputs: list[GenerateOutput] = []

        for uid in uids:
            tokens = collected[uid]
            text = self._tokenizer.decode(tokens) if tokens else ""

            info = finish_info.get(uid)
            if info and info.finish_reason == "stop" and info.match_sequence:
                # Distinguish eos match vs custom stop sequence
                if len(info.match_sequence) == 1 and info.match_sequence[0] in eos_set:
                    stop_reason, stop_seq = "eos", None
                else:
                    stop_text = self._tokenizer.decode(list(info.match_sequence))
                    stop_reason, stop_seq = "stop_sequence", stop_text
            elif info and info.finish_reason == "length":
                stop_reason, stop_seq = "max_tokens", None
            else:
                stop_reason, stop_seq = "eos", None

            # Post-loop string-based fallback: catches stops that failed to
            # tokenize (e.g. "<|" partial prefix). Same pattern as mlx_backend.
            if has_str_stops and stop_reason != "stop_sequence":
                for stop in params.stop:
                    idx = text.find(stop)
                    if idx != -1:
                        text = text[:idx]
                        tokens = (
                            self._tokenizer.encode(text, add_special_tokens=False) if text else []
                        )
                        stop_reason, stop_seq = "stop_sequence", stop
                        break

            outputs.append(
                GenerateOutput(
                    text=text,
                    tokens=tokens,
                    stop_reason=stop_reason,
                    stop_sequence=stop_seq,
                )
            )

        self._clear_cache()
        return outputs

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        self._ensure_tokenizer()
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        self._ensure_tokenizer()
        return self._tokenizer.decode(tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> list[int]:
        """Apply chat template and return token IDs."""
        self._ensure_tokenizer()
        chat_str = self._tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        return self.encode(chat_str)
