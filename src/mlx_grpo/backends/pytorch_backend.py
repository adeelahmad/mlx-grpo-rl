"""PyTorch generation backend — for CUDA/ROCm GPUs.

Wraps HuggingFace transformers models for use with the framework.
Enables training on NVIDIA/AMD hardware while keeping the same
template-driven rollout architecture.

Usage:
    backend = BackendRegistry.get("pytorch", model_path="meta-llama/Llama-3.2-3B")
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BackendRegistry, GenerateOutput, GenerateParams

_log = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PyTorchBackend(BackendRegistry, name="pytorch"):
    """PyTorch/HuggingFace Transformers generation backend.

    Supports CUDA, MPS, and CPU inference. Loads models via
    HuggingFace AutoModel with configurable precision.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "auto",
        dtype: str = "bfloat16",
        model: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
        **kwargs: Any,
    ):
        if not HAS_TORCH:
            raise ImportError("pip install torch transformers")

        self._model_path = model_path
        self._dtype = getattr(torch, dtype, torch.bfloat16)
        self._model = None
        self._tokenizer = None
        self._processor = processor

        # Accept pre-loaded model (e.g., from TorchModelLoader)
        if model is not None:
            self._model = model
            self._tokenizer = tokenizer
            self._processor = processor
            self._device = str(next(model.parameters()).device)
        else:
            self._device = self._resolve_device(device)
            if model_path:
                self._load(model_path)

    @property
    def name(self) -> str:
        return "pytorch"

    def _load(self, model_path: str) -> None:
        """Load model and tokenizer.

        Handles MLX-specific quantization config that PyTorch doesn't understand.
        """
        import json
        from pathlib import Path

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check for MLX quantization config and temporarily remove it
        config_path = Path(model_path) / "config.json"
        config_backup = None

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_dict = json.load(f)

                # Check if quantization_config exists and lacks quant_method
                if "quantization_config" in config_dict:
                    quant_cfg = config_dict["quantization_config"]
                    if isinstance(quant_cfg, dict) and "quant_method" not in quant_cfg:
                        config_backup = config_dict.copy()
                        del config_dict["quantization_config"]
                        with open(config_path, "w") as f:
                            json.dump(config_dict, f, indent=2)
            except Exception:
                pass

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self._dtype,
                device_map=self._device,
                ignore_mismatched_sizes=True,
            )
        finally:
            # Restore config.json if we modified it
            if config_backup is not None and config_path.exists():
                try:
                    with open(config_path, "w") as f:
                        json.dump(config_backup, f, indent=2)
                except Exception:
                    pass

        self._model.eval()

    def generate(
        self,
        prefix_tokens: list[int],
        params: GenerateParams,
        images: Any = None,
    ) -> GenerateOutput:
        """Generate tokens using HuggingFace generate().

        For VLM models, optionally pass preprocessed images.
        """
        assert self._model is not None, "Model not loaded"

        # Log adapter usage if present
        try:
            from peft import PeftModel

            if isinstance(self._model, PeftModel):
                _log.info("Generating with adapter enabled")
        except ImportError:
            pass

        # Set seed for reproducibility if provided
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(params.seed)

        input_ids = torch.tensor([prefix_tokens], device=self._device)
        # Create attention_mask (all 1s for non-padded tokens)
        attention_mask = torch.ones_like(input_ids)

        # Build stop criteria with StoppingCriteria
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StopOnTokens(StoppingCriteria):
            """Stop generation when any stop token sequence is encountered."""

            def __init__(self, stop_ids: list[list[int]], tokenizer: Any):
                self.stop_ids = stop_ids
                self.tokenizer = tokenizer

            def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
                # Check if any stop sequence has been generated
                for stop_seq in self.stop_ids:
                    if len(stop_seq) <= 0:
                        continue
                    # Check if last N tokens match stop sequence
                    if input_ids.shape[-1] >= len(stop_seq):
                        last_tokens = input_ids[0, -len(stop_seq) :].tolist()
                        if last_tokens == stop_seq:
                            return True
                return False

        stop_ids = []
        if params.stop and self._tokenizer:
            for s in params.stop:
                ids = self._tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids)

        # Build EOS token list (default + extra)
        eos_token_id = self._model.config.eos_token_id
        eos_token_ids = [eos_token_id] if eos_token_id is not None else []
        if params.extra_eos_tokens:
            eos_token_ids.extend(params.extra_eos_tokens)
        eos_token_ids = list(set(eos_token_ids)) if eos_token_ids else None

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_tokens,
            "temperature": max(params.temperature, 0.01),
            "top_p": params.top_p,
            "top_k": params.top_k,  # Pass directly, matching mlx-lm behavior
            "do_sample": params.temperature > 0.01,
            "attention_mask": attention_mask,
            "eos_token_id": eos_token_ids,  # Stop at EOS tokens
        }
        # MLX sampler supports min_p, min_tokens_to_keep; PyTorch/transformers only supports min_p
        if params.min_p > 0:
            gen_kwargs["min_p"] = params.min_p
        # Note: min_tokens_to_keep not supported by transformers.PreTrainedModel.generate()
        if params.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = params.repetition_penalty

        # XTC sampling: supported via repetition penalty + future processor
        # TODO: add XTC processor when available in transformers

        # Add images for VLM models
        if images is not None:
            gen_kwargs["images"] = images

        # Logit processors: stop criteria + logit bias
        processors = None
        if stop_ids or params.logit_bias:
            from transformers import LogitsProcessor, LogitsProcessorList

            class _BiasProc(LogitsProcessor):
                def __init__(self, bias: dict[int, float]):
                    self._bias = bias

                def __call__(self, input_ids, scores):
                    for tid, val in self._bias.items():
                        scores[:, tid] += val
                    return scores

            processor_list = []
            if params.logit_bias:
                processor_list.append(_BiasProc(params.logit_bias))
            processors = LogitsProcessorList(processor_list)
            gen_kwargs["logits_processor"] = processors

        # Add stopping criteria
        if stop_ids:
            stopping_criteria = StoppingCriteriaList([_StopOnTokens(stop_ids, self._tokenizer)])
            gen_kwargs["stopping_criteria"] = stopping_criteria

        # Run on a per-call CUDA stream so multiple rollout threads
        # generate in true parallel (default stream serializes everything).
        try:
            if self._device.startswith("cuda"):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream), torch.no_grad():
                    output = self._model.generate(input_ids, **gen_kwargs)
                stream.synchronize()
            else:
                with torch.no_grad():
                    output = self._model.generate(input_ids, **gen_kwargs)
        except ValueError as e:
            # If unsupported parameter error, retry without problematic params
            if "not used by the model" in str(e):
                unsupported = set()
                for word in str(e).split():
                    if word.startswith("'") and word.endswith("'"):
                        unsupported.add(word.strip("'"))
                for key in unsupported:
                    gen_kwargs.pop(key, None)
                if self._device.startswith("cuda"):
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream), torch.no_grad():
                        output = self._model.generate(input_ids, **gen_kwargs)
                    stream.synchronize()
                else:
                    with torch.no_grad():
                        output = self._model.generate(input_ids, **gen_kwargs)
            else:
                raise

        new_tokens = output[0][len(prefix_tokens) :].tolist()
        text = (
            self._tokenizer.decode(new_tokens, skip_special_tokens=True) if self._tokenizer else ""
        )

        # Determine stop reason: check for stop sequences
        stop_reason, stop_seq = "eos", None

        # Check if any stop sequence appears in the text
        for stop in params.stop:
            if stop in text:
                # Truncate at first occurrence of stop sequence
                text = text[: text.index(stop)]
                new_tokens = self.encode(text) if self._tokenizer else []
                stop_seq = stop
                stop_reason = "stop_sequence"
                break

        # If no stop sequence detected, check if we hit max tokens
        if stop_reason == "eos" and len(new_tokens) >= params.max_tokens:
            stop_reason = "max_tokens"

        return GenerateOutput(
            text=text,
            tokens=new_tokens,
            stop_reason=stop_reason,
            stop_sequence=stop_seq,
        )

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

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
