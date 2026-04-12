"""llama.cpp generation backend — CPU/GGUF quantized inference.

Wraps llama-cpp-python for efficient CPU inference with quantized
models. Useful for testing on machines without GPUs and for
running smaller quantized models.

Usage:
    backend = BackendRegistry.get("llamacpp", model_path="model.gguf")
"""

from __future__ import annotations

from typing import Any

from .base import BackendRegistry, GenerateOutput, GenerateParams

try:
    from llama_cpp import Llama

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False


class LlamaCppBackend(BackendRegistry, name="llamacpp"):
    """llama.cpp backend via llama-cpp-python.

    Supports GGUF quantized models for efficient CPU inference.
    Can also use GPU offloading via n_gpu_layers parameter.
    """

    def __init__(
        self,
        model_path: str = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **kwargs: Any,
    ):
        if not HAS_LLAMACPP:
            raise ImportError("pip install llama-cpp-python")

        self._model = None
        if model_path:
            self._model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

    @property
    def name(self) -> str:
        return "llamacpp"

    def generate(self, prefix_tokens: list[int], params: GenerateParams) -> GenerateOutput:
        assert self._model is not None

        prompt = self.decode(prefix_tokens)
        call_kw: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "temperature": max(params.temperature, 0.01),
            "top_p": params.top_p,
            "top_k": params.top_k if params.top_k > 0 else 40,
            "stop": params.stop or None,
        }
        if params.repetition_penalty > 1.0:
            call_kw["repeat_penalty"] = params.repetition_penalty
        if params.logit_bias:
            call_kw["logit_bias"] = params.logit_bias
        # VLM: pass image(s) as base64 image_url when available
        # Multi-image support: if multiple images, pass first image (llama.cpp limit)
        if params.images:
            import base64

            img = params.images[0]
            if isinstance(img, bytes):
                b64 = base64.b64encode(img).decode()
                call_kw["image_url"] = f"data:image/jpeg;base64,{b64}"
            else:
                call_kw["image_url"] = img
        output = self._model(prompt, **call_kw)

        text = output["choices"][0]["text"]
        finish = output["choices"][0].get("finish_reason", "stop")
        tokens = self.encode(text)

        stop_reason = "eos"
        stop_seq = None
        if finish == "length":
            stop_reason = "max_tokens"
        elif finish == "stop":
            for s in params.stop:
                if s in text or text.endswith(s):
                    stop_reason = "stop_sequence"
                    stop_seq = s
                    break

        return GenerateOutput(
            text=text,
            tokens=tokens,
            stop_reason=stop_reason,
            stop_sequence=stop_seq,
        )

    def encode(self, text: str) -> list[int]:
        assert self._model is not None
        return self._model.tokenize(text.encode("utf-8"), add_bos=False)

    def decode(self, tokens: list[int]) -> str:
        assert self._model is not None
        return self._model.detokenize(tokens).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        """Apply chat template. Uses ChatML format as default."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return self.encode("\n".join(parts))
