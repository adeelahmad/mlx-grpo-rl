"""PyTorch model conversion and quantization — mlx_lm.convert equivalent.

Provides CLI utilities for:
  - Downloading and converting HuggingFace models to local format
  - Quantizing models (4-bit QLoRA, 8-bit)
  - Merging adapters into base weights
  - Inspecting model configs
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def convert_model(
    model_id: str,
    output_dir: str | Path,
    trust_remote_code: bool = False,
    quantize_bits: int | None = None,
    merge_adapters: list[str] | None = None,
) -> int:
    """Download and optionally quantize a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-3B')
        output_dir: Local directory to save model files
        trust_remote_code: Allow remote code execution
        quantize_bits: Quantization bits (4 or 8); None = no quantization
        merge_adapters: Optional list of adapter paths to merge

    Returns:
        0 on success, 1 on error
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        _log.error("pip install torch transformers")
        return 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info(f"Converting {model_id} to {output_dir}")

    try:
        # Download tokenizer
        _log.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(output_dir)
        _log.info(f"✓ Tokenizer saved to {output_dir}")

        # Download model with optional quantization
        quantization_config = None
        if quantize_bits == 4:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                _log.info("4-bit quantization enabled (QLoRA)")
            except ImportError:
                _log.error("pip install bitsandbytes (requires CUDA)")
                return 1
        elif quantize_bits == 8:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                _log.info("8-bit quantization enabled")
            except ImportError:
                _log.error("pip install bitsandbytes (requires CUDA)")
                return 1

        # Download model
        _log.info("Downloading model...")
        device_map = "auto" if quantize_bits else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if not quantize_bits else None,
        )

        # Save model
        _log.info("Saving model...")
        model.save_pretrained(output_dir)
        _log.info(f"✓ Model saved to {output_dir}")

        # Merge adapters if provided
        if merge_adapters:
            from mlx_grpo.training.pytorch_adapter_migration import merge_adapter_into_base

            for adapter_path in merge_adapters:
                _log.info(f"Merging adapter: {adapter_path}")
                n_fused = merge_adapter_into_base(model, adapter_path)
                if n_fused > 0:
                    model.save_pretrained(output_dir)
                    _log.info(f"✓ Merged {n_fused} layers, saved to {output_dir}")

        _log.info("✓ Conversion complete")
        return 0

    except Exception as e:
        _log.error(f"Conversion failed: {e}")
        return 1


def quantize_model(
    model_dir: str | Path,
    output_dir: str | Path,
    bits: int = 4,
) -> int:
    """Quantize an existing model in-place or to a new directory.

    Args:
        model_dir: Path to local model directory
        output_dir: Output directory for quantized model
        bits: Quantization bits (4 or 8)

    Returns:
        0 on success, 1 on error
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        _log.error("pip install torch transformers")
        return 1

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (model_dir / "config.json").exists():
        _log.error(f"No config.json found in {model_dir}")
        return 1

    _log.info(f"Quantizing {model_dir} to {bits}-bit")

    try:
        # Load quantization config
        if bits == 4:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                _log.info("4-bit quantization (QLoRA)")
            except ImportError:
                _log.error("pip install bitsandbytes (requires CUDA)")
                return 1
        elif bits == 8:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                _log.info("8-bit quantization")
            except ImportError:
                _log.error("pip install bitsandbytes (requires CUDA)")
                return 1
        else:
            _log.error(f"Unsupported quantization bits: {bits}")
            return 1

        # Load tokenizer (no quantization needed)
        _log.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.save_pretrained(output_dir)

        # Load and quantize model
        _log.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # Save quantized model
        _log.info("Saving quantized model...")
        model.save_pretrained(output_dir)
        _log.info(f"✓ Quantized model saved to {output_dir}")

        return 0

    except Exception as e:
        _log.error(f"Quantization failed: {e}")
        return 1


def inspect_model(model_dir: str | Path) -> int:
    """Print model config and parameter counts.

    Args:
        model_dir: Path to local model directory

    Returns:
        0 on success, 1 on error
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        _log.error("pip install transformers")
        return 1

    model_dir = Path(model_dir)

    if not (model_dir / "config.json").exists():
        _log.error(f"No config.json found in {model_dir}")
        return 1

    try:
        # Load config
        config = AutoConfig.from_pretrained(model_dir)
        print(f"\n{'=' * 60}")
        print(f"Model: {config.model_type}")
        print(f"{'=' * 60}")
        print(f"  Hidden size:     {config.hidden_size}")
        print(f"  Num layers:      {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Vocab size:      {config.vocab_size}")
        if hasattr(config, "max_position_embeddings"):
            print(f"  Max seq length:  {config.max_position_embeddings}")

        # Load model to count parameters
        print("\nLoading model for parameter count...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cpu",
        )
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params:    {total:,}")
        print(f"  Trainable:       {trainable:,}")
        print(f"  Trainable %:     {100 * trainable / total:.1f}%")
        print(f"{'=' * 60}\n")

        return 0

    except Exception as e:
        _log.error(f"Inspection failed: {e}")
        return 1


# ── End-of-File ──────────────────────────────────────────────────────
# Usage:
#   mlx-grpo torch-convert meta-llama/Llama-3.2-3B ./models/llama
#   mlx-grpo torch-quantize ./models/llama ./models/llama-q4 --bits 4
#   mlx-grpo torch-inspect ./models/llama
# ─────────────────────────────────────────────────────────────────────
