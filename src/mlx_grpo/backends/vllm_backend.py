"""vLLM generation backend — high-throughput GPU inference.

Uses vLLM's continuous batching for maximum throughput when
generating multiple rollouts in parallel. Ideal for large-scale
GRPO training with multiple GPUs.

Usage:
    backend = BackendRegistry.get("vllm", model_path="meta-llama/Llama-3.2-3B")
"""

from __future__ import annotations

from typing import Any

from .base import BackendRegistry, GenerateOutput, GenerateParams

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


class VLLMBackend(BackendRegistry, name="vllm"):
    """vLLM backend for high-throughput generation.

    Leverages continuous batching and PagedAttention for
    efficient parallel rollout generation.
    """

    def __init__(
        self,
        model_path: str = "",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        **kwargs: Any,
    ):
        if not HAS_VLLM:
            raise ImportError("pip install vllm")

        self._model_path = model_path
        self._llm = None
        self._tokenizer = None

        if model_path:
            self._llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    @property
    def name(self) -> str:
        return "vllm"

    def generate(self, prefix_tokens: list[int], params: GenerateParams) -> GenerateOutput:
        assert self._llm is not None and self._tokenizer is not None

        prompt = self._tokenizer.decode(prefix_tokens)
        sampling_kw: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "temperature": max(params.temperature, 0.01),
            "top_p": params.top_p,
            "top_k": params.top_k if params.top_k > 0 else -1,
            "stop": params.stop or None,
        }
        if params.repetition_penalty > 1.0:
            sampling_kw["repetition_penalty"] = params.repetition_penalty
        if params.logit_bias:
            sampling_kw["logit_bias"] = params.logit_bias
        sampling = SamplingParams(**sampling_kw)

        # VLM: wrap prompt in multimodal dict when images are provided
        if params.images:
            generate_input = [{"prompt": prompt, "multi_modal_data": {"image": params.images}}]
        else:
            generate_input = [prompt]
        outputs = self._llm.generate(generate_input, sampling)
        output = outputs[0].outputs[0]

        text = output.text
        tokens = list(output.token_ids)
        stop_reason = output.finish_reason or "eos"

        stop_seq = None
        if stop_reason == "stop":
            stop_reason = "stop_sequence"
            for s in params.stop:
                if text.endswith(s) or s in text:
                    stop_seq = s
                    break

        return GenerateOutput(
            text=text,
            tokens=tokens,
            stop_reason=stop_reason,
            stop_sequence=stop_seq,
        )

    def generate_batch(
        self,
        prefixes: list[list[int]],
        params: GenerateParams,
    ) -> list[GenerateOutput]:
        """Batch generation — vLLM's strength.

        Generates multiple rollouts in a single batched call
        for maximum throughput.
        """
        assert self._llm is not None and self._tokenizer is not None

        prompts = [self._tokenizer.decode(p) for p in prefixes]
        sampling = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=max(params.temperature, 0.01),
            top_p=params.top_p,
            top_k=params.top_k if params.top_k > 0 else -1,
            stop=params.stop or None,
        )

        outputs = self._llm.generate(prompts, sampling)
        results = []
        for out in outputs:
            o = out.outputs[0]
            results.append(
                GenerateOutput(
                    text=o.text,
                    tokens=list(o.token_ids),
                    stop_reason=o.finish_reason or "eos",
                )
            )
        return results

    def encode(self, text: str) -> list[int]:
        assert self._tokenizer is not None
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        assert self._tokenizer is not None
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        return self.encode(text)
