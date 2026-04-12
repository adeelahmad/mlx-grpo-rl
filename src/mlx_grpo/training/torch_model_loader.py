"""PyTorch model loading — HuggingFace + PEFT for CUDA/MPS/CPU training.

Parallel to model_loader.py but using HuggingFace transformers and PEFT
instead of MLX. Supports LoRA, DoRA, QLoRA (CUDA only), and full fine-tuning.

Usage:
    loader = TorchModelLoader(ModelConfig(model_path="meta-llama/Llama-3.2-3B"))
    model, ref_model, tokenizer = loader.load()
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from mlx_grpo.config.vision import VisionConfig
from mlx_grpo.training.model_loader import AdapterType, ModelConfig
from mlx_grpo.training.vlm_loader import freeze_vision_layers, get_vlm_lora_targets, load_processor

_log = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from peft import LoraConfig, PeftModel, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class TorchModelLoader:
    """Loads and configures PyTorch models for GRPO training.

    Modes:
      - LoRA:  PEFT LoraConfig with adapter training
      - DoRA:  PEFT LoraConfig(use_dora=True)
      - QLoRA: BitsAndBytes 4-bit + LoRA (CUDA only)
      - Full:  All parameters trainable, deepcopy for reference
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.adapter_type = AdapterType.from_str(config.adapter_type)

    def load(self) -> tuple[Any, Any, Any, Any | None]:
        """Load model, reference model, tokenizer, and optional processor.

        Returns:
            (model, ref_model, tokenizer, processor)
        """
        if not HAS_TORCH:
            raise ImportError("pip install torch transformers")

        _log.info(
            "Loading %s (mode: %s, backend: pytorch)",
            self.config.model_path,
            self.adapter_type.value,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path or self.config.model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        import json
        from pathlib import Path

        dtype = _resolve_dtype(self.config.dtype)

        # Device mapping strategy:
        # - "gpu" (default): Keep model on GPU for training speed (fails if OOM)
        # - "auto": Allow CPU offloading on OOM (slow but fits on 8GB constraints)
        # - "cpu": CPU-only inference mode
        device_map_strategy = getattr(self.config, "pytorch_device_map", "gpu")
        if device_map_strategy == "gpu":
            device_map = None  # None = GPU only, explicit
            _log.info("Device mapping: GPU-only (training mode, max speed)")
        elif device_map_strategy == "auto":
            device_map = "auto"  # CPU offloading allowed (transformers device_map)
            _log.info(
                "Device mapping: auto (CPU offloading allowed on OOM, slow but constrained VRAM ok)"
            )
        elif device_map_strategy == "cpu":
            device_map = "cpu"
            _log.info("Device mapping: CPU-only (inference mode)")
        else:
            device_map = None  # Default to GPU-only for safety
            _log.warning(
                f"Unknown pytorch_device_map={device_map_strategy!r}, defaulting to GPU-only"
            )

        # QLoRA: 4-bit quantization (CUDA only)
        quantization_config = None
        if self.adapter_type in (AdapterType.LORA, AdapterType.DORA):
            use_qlora = getattr(self.config, "qlora", False)
            if use_qlora:
                quantization_config = _get_bnb_config()

        # Handle MLX quantization config (remove if present, restore after load)
        config_path = Path(self.config.model_path) / "config.json"
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
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
            # If device_map=None (GPU-only mode), explicitly move to GPU
            # transformers doesn't auto-move to GPU when device_map=None
            if device_map is None:
                import torch

                if torch.cuda.is_available():
                    model = model.to("cuda")
                    _log.info("Model moved to CUDA device")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    model = model.to("mps")
                    _log.info("Model moved to MPS device")
        finally:
            # Restore config.json if we modified it
            if config_backup is not None and config_path.exists():
                try:
                    with open(config_path, "w") as f:
                        json.dump(config_backup, f, indent=2)
                except Exception:
                    pass

        # Handle VLM configuration if provided
        vision_cfg = self.config.vision or VisionConfig()
        freeze_vision_layers(model, vision_cfg)
        processor = load_processor(vision_cfg)

        if self.adapter_type == AdapterType.FULL:
            return self._setup_full(model, tokenizer, vision_cfg, processor)
        return self._setup_adapter(model, tokenizer, vision_cfg, processor)

    def _setup_adapter(
        self,
        model: Any,
        tokenizer: Any,
        vision_cfg: VisionConfig,
        processor: Any | None,
    ) -> tuple[Any, Any, Any, Any | None]:
        """Apply LoRA/DoRA via PEFT."""
        if not HAS_PEFT:
            raise ImportError("pip install peft>=0.8")

        cfg = self.config.adapter
        use_dora = self.adapter_type == AdapterType.DORA

        # Convert MLX-style target module names to HuggingFace style
        target_modules = _convert_target_modules(cfg.target_modules)

        # Add VLM-specific targets for projector layers
        target_modules = get_vlm_lora_targets(vision_cfg, target_modules)

        lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=target_modules,
            use_dora=use_dora,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Load existing adapter or create new
        if self.config.adapter_path:
            model = PeftModel.from_pretrained(model, self.config.adapter_path)
            _log.info("Loaded PEFT adapter from %s", self.config.adapter_path)
        else:
            model = get_peft_model(model, lora_config)

        model.train()

        if self.config.grad_checkpoint:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        ref_model = _PeftReferenceModel(model)

        total, trainable = count_params(model)
        label = "DoRA" if use_dora else "LoRA"
        _log.info(
            "%s: %s trainable / %s total (%.2f%%)",
            label,
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )

        return model, ref_model, tokenizer, processor

    def _setup_full(
        self,
        model: Any,
        tokenizer: Any,
        vision_cfg: VisionConfig,
        processor: Any | None,
    ) -> tuple[Any, Any, Any, Any | None]:
        """Full model fine-tuning with deepcopy reference."""
        _log.info("Full fine-tuning: cloning reference model (higher memory)")

        model.train()
        if self.config.grad_checkpoint:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        ref_model = _TorchSnapshotReferenceModel(model)

        total, _ = count_params(model)
        _log.info("Full: %s parameters (all trainable)", f"{total:,}")

        return model, ref_model, tokenizer, processor


class _PeftReferenceModel:
    """Reference forward with PEFT adapter explicitly disabled.

    Uses PEFT's disable_adapter() context manager which correctly
    calls disable_adapter_layers() on enter and enable_adapter_layers()
    on exit.  The previous code called disable_adapter() as a bare
    function — a no-op because it's a @contextmanager (returns an
    unentered generator).
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        self._model.eval()
        try:
            # disable_adapter() is a context manager in PEFT ≥0.8:
            # it calls disable_adapter_layers() on enter and
            # enable_adapter_layers() on exit, giving base-model logits.
            with self._model.disable_adapter(), torch.no_grad():
                output = self._model(input_ids, **kwargs)
        finally:
            self._model.train()
        return output


class _TorchSnapshotReferenceModel:
    """Full fine-tune: deepcopy of initial weights as frozen reference."""

    def __init__(self, model: Any) -> None:
        self._ref = copy.deepcopy(model)
        self._ref.eval()
        for p in self._ref.parameters():
            p.requires_grad = False

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return self._ref(input_ids, **kwargs)


def count_params(model: Any) -> tuple[int, int]:
    """Count (total, trainable) parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _convert_target_modules(modules: list[str]) -> list[str]:
    """Convert MLX-style target module names to HuggingFace PEFT style.

    MLX uses full path: 'self_attn.q_proj'
    PEFT matches by suffix: 'q_proj'
    """
    converted = []
    for m in modules:
        # Take the last component (e.g., 'self_attn.q_proj' -> 'q_proj')
        parts = m.split(".")
        converted.append(parts[-1])
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in converted:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _resolve_dtype(dtype_str: str) -> Any:
    """Resolve dtype string to torch dtype."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def _get_bnb_config() -> Any:
    """Create BitsAndBytes 4-bit quantization config for QLoRA."""
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA — not available on MPS/CPU")
    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        raise ImportError("pip install bitsandbytes (CUDA only)") from None
