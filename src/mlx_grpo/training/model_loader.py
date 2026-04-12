"""Model loading — loads models, applies adapters, creates reference copies.

Supports three training modes:
  - LoRA:  Low-Rank Adaptation (default, memory efficient)
  - DoRA:  Weight-Decomposed Low-Rank Adaptation (better convergence than LoRA)
  - Full:  Full model fine-tuning (all parameters trainable, highest memory)

Each mode produces the same (model, ref_model, tokenizer) triple.

Usage:
    loader = ModelLoader(ModelConfig(model_path="Qwen/Qwen3-4B", adapter_type="dora"))
    model, ref_model, tokenizer = loader.load()
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Stub for type checking / import when MLX not installed
    mx = None  # type: ignore[assignment]


# ── Adapter Type ─────────────────────────────────────────────────────


class AdapterType(enum.Enum):
    """Training mode / adapter type."""

    LORA = "lora"
    DORA = "dora"
    FULL = "full"
    LORA_MOE = "lora_moe"

    @classmethod
    def from_str(cls, s: str) -> AdapterType:
        """Parse from config string, case-insensitive."""
        try:
            return cls(s.lower().strip())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown adapter type '{s}'. Valid: {valid}") from None


# ── DoRA Layer ───────────────────────────────────────────────────────

# DoRALinear requires MLX nn.Module. Define conditionally to allow
# importing the module (for AdapterType, configs, etc.) without MLX.

if HAS_MLX:

    class DoRALinear(nn.Module):
        """Weight-Decomposed Low-Rank Adaptation (DoRA) layer.

        DoRA decomposes the pre-trained weight W into magnitude (m) and
        direction (V/||V||), then applies LoRA to the direction component:

            W' = m * (V + B·A) / ||V + B·A||

        This preserves the magnitude/direction decomposition during training,
        which empirically gives better convergence than standard LoRA.

        Reference: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
            bias: bool = False,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.scale = alpha / rank

            # Frozen base weight (direction component)
            self.weight = mx.zeros((out_features, in_features))

            # Trainable magnitude vector (per output neuron)
            self.magnitude = mx.ones((out_features,))

            # LoRA low-rank factors (applied to direction)
            self.lora_a = mx.random.normal((rank, in_features)) * (0.01 / rank)
            self.lora_b = mx.zeros((out_features, rank))

            self.dropout = dropout
            self.has_bias = bias
            if bias:
                self.bias = mx.zeros((out_features,))

        @classmethod
        def from_linear(
            cls,
            linear: Any,
            r: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
        ) -> DoRALinear:
            """Create DoRA layer from an existing nn.Linear."""
            has_bias = hasattr(linear, "bias") and linear.bias is not None
            weight = linear.weight

            layer = cls(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                rank=r,
                alpha=alpha,
                dropout=dropout,
                bias=has_bias,
            )

            # Decompose W into magnitude and direction
            # m = ||W_row|| for each output neuron
            col_norms = mx.sqrt(mx.sum(weight * weight, axis=1, keepdims=False))
            layer.magnitude = col_norms
            layer.weight = weight  # frozen base direction

            if has_bias:
                layer.bias = linear.bias

            return layer

        def __call__(self, x: Any) -> Any:
            """Forward pass: m * (V + scale*B·A)x / ||V + scale*B·A||."""
            # Direction = base weight + scaled LoRA update
            lora_update = self.lora_b @ self.lora_a  # (out, in)
            adapted_weight = self.weight + self.scale * lora_update

            # Normalize direction per output neuron
            direction_norm = mx.sqrt(mx.sum(adapted_weight * adapted_weight, axis=1, keepdims=True))
            normalized = adapted_weight / mx.maximum(direction_norm, mx.array(1e-8))

            # Apply magnitude: W' = diag(m) @ normalized
            weighted = self.magnitude[:, None] * normalized

            # Standard linear: output = x @ W'.T
            result = x @ weighted.T

            if self.has_bias:
                result = result + self.bias

            return result

        @property
        def is_dora(self) -> bool:
            """Marker property for adapter detection."""
            return True
else:
    # Stub so the name exists for isinstance checks / type hints
    class DoRALinear:  # type: ignore[no-redef]
        """DoRA requires MLX. Install with: pip install mlx"""

        is_dora = True

        @classmethod
        def from_linear(cls, *args: Any, **kwargs: Any) -> DoRALinear:
            raise ImportError("DoRA requires MLX: pip install mlx")


# ── Config ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class AdapterConfig:
    """Adapter configuration for LoRA / DoRA."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.v_proj",
        ]
    )
    layers: int | None = None  # None = all layers


@dataclass(slots=True)
class ModelConfig:
    """Model loading configuration."""

    model_path: str = ""
    adapter_path: str | None = None
    tokenizer_path: str | None = None
    adapter_type: str = "lora"  # lora | dora | full | lora_moe
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    freeze_base: bool = True  # ignored for full fine-tuning
    grad_checkpoint: bool = True  # gradient checkpointing on transformer layers
    dtype: str = "float16"
    moe_cfg: Any = None  # MoEConfig | None — avoids circular import
    vision: Any = None  # VisionConfig | None — avoids circular import


# ── Model Loader ─────────────────────────────────────────────────────


class ModelLoader:
    """Loads and configures models for GRPO training.

    Supports three modes:
      - LoRA:  Applies LoRA adapters to target modules. Reference model
               uses the same model with adapters disabled (scale=0).
      - DoRA:  Like LoRA but decomposes weight into magnitude+direction.
               Same reference model strategy (adapter scale=0).
      - Full:  All parameters trainable. Reference model is a frozen
               weight snapshot (higher memory cost).

    Produces:
      - model:     Trainable model
      - ref_model:  Frozen reference for KL computation
      - tokenizer:  HuggingFace tokenizer
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.adapter_type = AdapterType.from_str(config.adapter_type)

    def load(self) -> tuple[Any, Any, Any]:
        """Load model, reference model, and tokenizer.

        Returns:
            (model, ref_model, tokenizer)
        """
        if not HAS_MLX:
            raise ImportError("pip install mlx mlx-lm")

        from mlx_lm import load as mlx_load

        _log.info(
            "Loading model %s (mode: %s)",
            self.config.model_path,
            self.adapter_type.value,
        )

        model, tokenizer = mlx_load(
            self.config.model_path,
            tokenizer_config={"trust_remote_code": True},
        )

        # Verify quantization is intact after loading
        is_q, bits, n_q = _check_quantization(model)
        if is_q:
            _log.info("✓ Model loaded as %d-bit quantized (%d QuantizedLinear layers)", bits, n_q)
            print(f"[model] ✓ {bits}-bit quantized ({n_q} layers) — uint{bits} base weights")
        else:
            _log.warning(
                "Model loaded as FULL PRECISION — no QuantizedLinear layers detected in %s",
                self.config.model_path,
            )
            print("[model] ⚠ FULL PRECISION — no quantized layers found. Expected 4-bit model.")

        if self.adapter_type == AdapterType.FULL:
            return self._setup_full(model, tokenizer)
        elif self.adapter_type == AdapterType.DORA:
            return self._setup_dora(model, tokenizer)
        elif self.adapter_type == AdapterType.LORA_MOE:
            return self._setup_moe(model, tokenizer)
        else:
            return self._setup_lora(model, tokenizer)

    # ── LoRA / DoRA Setup ─────────────────────────────────────────

    def _setup_lora(self, model: Any, tokenizer: Any) -> tuple[Any, Any, Any]:
        """Apply LoRA adapters using mlx-lm's battle-tested utility.

        Pattern (same as mlx-lm's own training):
        1. Freeze entire model
        2. linear_to_lora_layers() converts target layers — LoRA params auto-unfrozen
        3. Load checkpoint weights if resuming
        """
        return self._setup_adapter(model, tokenizer, use_dora=False)

    def _setup_dora(self, model: Any, tokenizer: Any) -> tuple[Any, Any, Any]:
        """Apply DoRA adapters using mlx-lm's utility."""
        return self._setup_adapter(model, tokenizer, use_dora=True)

    def _setup_adapter(self, model: Any, tokenizer: Any, use_dora: bool) -> tuple[Any, Any, Any]:
        """Shared LoRA/DoRA setup using mlx-lm's linear_to_lora_layers."""
        label = "DoRA" if use_dora else "LoRA"
        cfg = self.config.adapter

        # Step 1: Freeze entire model (required before linear_to_lora_layers)
        model.freeze()

        # Step 2: Apply LoRA/DoRA via mlx-lm's official utility
        num_layers = cfg.layers if cfg.layers else -1  # -1 = all layers
        lora_config = {
            "rank": cfg.rank,
            "scale": cfg.alpha / max(cfg.rank, 1),
            "dropout": cfg.dropout,
            "keys": cfg.target_modules,
        }
        _apply_lora_layers(model, num_layers, lora_config, use_dora=use_dora)

        # Step 3 (DoRA only): Snapshot pre-trained magnitude vectors BEFORE loading any
        # checkpoint. These are needed to correctly compute reference-model outputs.
        # After a checkpoint load, m holds trained values; the reference model must use
        # the pre-trained norms so that: m_init * V / ||V|| == V (the original weight).
        initial_magnitudes: dict[int, Any] = {}
        if use_dora and HAS_MLX:
            for _, module in model.named_modules():
                m = getattr(module, "magnitude", None)
                if m is not None:
                    initial_magnitudes[id(module)] = mx.array(m)

        # Step 4: Gradient checkpointing (breaks backward into per-layer
        # Metal command buffers, preventing macOS GPU timeout)
        if self.config.grad_checkpoint:
            _enable_grad_checkpointing(model)

        # Step 5: Load checkpoint weights if resuming
        self._maybe_load_adapters(model)

        # Step 6: Verify quantization survived LoRA wrapping
        is_q, bits, _ = _check_quantization(model)
        if is_q:
            _log.info("%s: %d-bit base weights preserved inside LoRA wrappers", label, bits)
        else:
            _log.warning(
                "%s: quantization NOT detected after LoRA setup — running full precision", label
            )

        # Step 7: Reference model (same model, adapters disabled via scale=0).
        # For DoRA, also pass initial magnitudes so the ref forward restores m to its
        # pre-trained values — without this, scale=0 gives m_trained*V/||V|| ≠ V.
        ref_model = _AdapterReferenceModel(model, initial_magnitudes=initial_magnitudes)

        total, trainable = count_params(model, self.adapter_type)
        _log.info(
            "%s: %s trainable / %s total (%.2f%%)",
            label,
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )
        return model, ref_model, tokenizer

    # ── LoRA-MoE Setup ────────────────────────────────────────────

    def _setup_moe(self, model: Any, tokenizer: Any) -> tuple[Any, Any, Any]:
        """Apply LoRA-MoE adapters and create MoE wrapper model."""
        moe_cfg = self.config.moe_cfg
        if moe_cfg is None:
            raise ValueError("adapter_type=lora_moe requires moe_cfg in ModelConfig")

        from ..model.lora_moe import apply_moe_lora_to_model
        from ..model.moe_model import LoRAMoEModel

        # Step 1: freeze base
        model.freeze()

        # Step 2: build per-expert rank/alpha lists respecting overrides
        expert_ranks = [moe_cfg.lora_rank] * moe_cfg.num_experts
        expert_alphas = [moe_cfg.lora_alpha] * moe_cfg.num_experts
        for et in moe_cfg.expert_types:
            if 0 <= et.expert_index < moe_cfg.num_experts:
                if et.lora_rank is not None:
                    expert_ranks[et.expert_index] = et.lora_rank
                if et.lora_alpha is not None:
                    expert_alphas[et.expert_index] = et.lora_alpha

        # Step 3: replace target nn.Linear layers with LoRAMoELinear
        apply_moe_lora_to_model(
            model,
            num_experts=moe_cfg.num_experts,
            target_modules=moe_cfg.lora_targets,
            num_layers=moe_cfg.lora_layers,
            expert_ranks=expert_ranks,
            expert_alphas=expert_alphas,
            dropout=moe_cfg.lora_dropout,
        )

        # Step 4: gradient checkpointing
        if self.config.grad_checkpoint:
            _enable_grad_checkpointing(model)

        # Step 5: wrap in LoRAMoEModel
        moe_model = LoRAMoEModel(model, moe_cfg)
        moe_model.freeze_base()

        # Step 6: load checkpoint weights if resuming
        if self.config.adapter_path:
            _load_moe_weights(moe_model, self.config.adapter_path)

        # Step 7: reference model (same model, scale toggled to 0)
        ref_model = _MoEAdapterReferenceModel(moe_model)

        _log.info(
            "LoRA-MoE: %d experts, rank=%s, targets=%s",
            moe_cfg.num_experts,
            expert_ranks,
            list(moe_cfg.lora_targets),
        )
        return moe_model, ref_model, tokenizer

    # ── Full Fine-Tuning Setup ───────────────────────────────────

    def _setup_full(self, model: Any, tokenizer: Any) -> tuple[Any, Any, Any]:
        """Full model fine-tuning. Reference model is a weight snapshot."""
        _log.info("Full fine-tuning: snapshotting reference weights (higher memory)")
        ref_weights = _snapshot_weights(model)
        ref_model = _SnapshotReferenceModel(model, ref_weights)

        if self.config.grad_checkpoint:
            _enable_grad_checkpointing(model)

        # Load checkpoint weights if resuming
        if self.config.adapter_path:
            _load_full_checkpoint(model, self.config.adapter_path)

        total, _ = count_params(model, self.adapter_type)
        _log.info("Full: %s parameters (all trainable)", f"{total:,}")
        return model, ref_model, tokenizer

    def _maybe_load_adapters(self, model: Any) -> Any:
        """Load adapter weights if resuming from checkpoint."""
        if self.config.adapter_path:
            model = _load_adapter_weights(model, self.config.adapter_path)
        return model


# ── Standalone Utilities ─────────────────────────────────────────────


def _check_quantization(model: Any) -> tuple[bool, int, int]:
    """Check if model has QuantizedLinear layers (survives LoRA wrapping via nested modules).

    Returns: (is_quantized, bits, n_quant_layers)
    """
    try:
        n_quant = 0
        bits = 0
        for _, module in model.named_modules():
            if type(module).__name__ == "QuantizedLinear":
                n_quant += 1
                if bits == 0:
                    bits = getattr(module, "bits", 4)
        return n_quant > 0, bits, n_quant
    except Exception:
        return False, 0, 0


def _enable_grad_checkpointing(model: Any) -> int:
    """Enable gradient checkpointing on all transformer layers.

    Trades compute for memory by not storing intermediate activations
    during the forward pass — they are recomputed during backward.
    This breaks the backward pass into per-layer Metal command buffers,
    preventing macOS interactivity timeouts on large models.

    Must be called AFTER LoRA/DoRA layers are applied.
    """
    layers = getattr(getattr(model, "model", None), "layers", [])
    if not layers:
        return 0

    # Wrap the layer type's __call__ to use mx.checkpoint
    original_fn = type(layers[0]).__call__

    def checkpointed_fn(layer_self: Any, *args: Any, **kwargs: Any) -> Any:
        def inner_fn(params: dict, *args: Any, **kwargs: Any) -> Any:
            layer_self.update(params)
            return original_fn(layer_self, *args, **kwargs)

        return mx.checkpoint(inner_fn)(layer_self.trainable_parameters(), *args, **kwargs)

    type(layers[0]).__call__ = checkpointed_fn
    count = len(layers)
    _log.info("Gradient checkpointing: %d layers", count)
    return count


def _apply_lora_layers(
    model: Any,
    num_layers: int,
    config: dict,
    *,
    use_dora: bool = False,
) -> None:
    """Apply LoRA/DoRA using mlx-lm's linear_to_lora_layers.

    This is the same utility mlx-lm uses for its own LoRA training.
    It correctly handles QuantizedLinear, freeze/unfreeze, and
    all layer types. Model MUST be frozen before calling.
    """
    try:
        from mlx_lm.tuner import linear_to_lora_layers
    except ImportError:
        try:
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except ImportError:
            raise ImportError(
                "mlx-lm tuner not found. Install mlx-lm>=0.20: pip install mlx-lm"
            ) from None

    try:
        linear_to_lora_layers(model, num_layers, config, use_dora=use_dora)
    except Exception as exc:
        if use_dora and _check_quantization(model)[0]:
            raise RuntimeError(
                "DoRA failed on a quantized model. mlx-lm's linear_to_lora_layers requires "
                "access to dequantized float weights to initialize magnitude vectors, but your "
                "mlx-lm version may not support this. "
                "Try: pip install -U mlx-lm  (>=0.21 is required for DoRA + 4-bit). "
                "Alternatively, use adapter_type: lora which works with all quantized models. "
                f"Original error: {exc}"
            ) from exc
        raise

    # Count what was applied
    from mlx.utils import tree_flatten

    lora_params = [k for k, _ in tree_flatten(model.trainable_parameters())]
    _log.info(
        "Applied %s via mlx-lm (%d trainable parameter groups)",
        "DoRA" if use_dora else "LoRA",
        len(lora_params),
    )


def _load_adapter_weights(model: Any, path: str) -> Any:
    """Load LoRA/DoRA adapter weights from checkpoint."""
    adapter_path = Path(path)
    for filename in ("adapters.safetensors", "adapters.npz"):
        fpath = adapter_path / filename
        if fpath.exists():
            weights = dict(mx.load(str(fpath)))
            model.load_weights(list(weights.items()), strict=False)
            _log.info("Loaded adapter weights from %s", fpath)
            return model
    _log.warning("No adapter weights found in %s", adapter_path)
    return model


def _load_full_checkpoint(model: Any, path: str) -> None:
    """Load full model weights from checkpoint.

    Supports both flat layout (path/model.safetensors) and the newer
    directory-based layout (path/latest/model.safetensors).
    """
    ckpt_path = Path(path)
    # Prefer latest/ subdir (new layout), fall back to flat path (old layout).
    search_dirs = []
    latest = ckpt_path / "latest"
    if latest.exists():  # resolves symlink
        search_dirs.append(latest.resolve())
    search_dirs.append(ckpt_path)
    for base in search_dirs:
        for filename in ("adapters.safetensors", "model.safetensors", "model.npz", "weights.npz"):
            fpath = base / filename
            if fpath.exists():
                weights = dict(mx.load(str(fpath)))
                model.load_weights(list(weights.items()))
                _log.info("Loaded full model weights from %s", fpath)
                return
    _log.warning("No full model checkpoint found in %s", ckpt_path)


def get_trainable_params(model: Any, adapter_type: AdapterType) -> dict[str, Any]:
    """Get trainable parameters based on training mode."""
    from mlx.utils import tree_flatten

    if adapter_type == AdapterType.FULL:
        return dict(tree_flatten(model.parameters()))

    trainable = {}
    for name, param in tree_flatten(model.parameters()):
        name_lower = name.lower()
        if "lora" in name_lower or "magnitude" in name_lower or "dora" in name_lower:
            trainable[name] = param
    return trainable


def count_params(model: Any, adapter_type: AdapterType | None = None) -> tuple[int, int]:
    """Count (total, trainable) parameters.

    For adapter modes, trainable = adapter params only.
    For full mode, trainable = total.
    """
    from mlx.utils import tree_flatten

    total = 0
    trainable = 0
    for name, param in tree_flatten(model.parameters()):
        size = param.size if hasattr(param, "size") else 0
        total += size
        name_lower = name.lower()
        if "lora" in name_lower or "magnitude" in name_lower or "dora" in name_lower:
            trainable += size

    if adapter_type == AdapterType.FULL or trainable == 0:
        trainable = total

    return total, trainable


# ── Reference Model Implementations ─────────────────────────────────


class _AdapterReferenceModel:
    """Reference model for LoRA/DoRA training modes.

    Temporarily disables adapter layers (scale=0) to produce original
    base model outputs. Zero-copy and safe because:
    - Base weights are frozen
    - Scale and DoRA magnitudes are always restored in finally block
    - LoRA modules are cached on init (no per-forward scan)

    DoRA note: setting scale=0 alone is insufficient after any gradient
    step because the magnitude vector m has been updated:
        scale=0 → W_ref = m_trained * V / ||V||  ≠  V
    Fix: also restore m to its initial (pre-trained) value so that:
        m_init = ||V_col||  →  m_init * V / ||V|| = V  ✓
    The initial_magnitudes dict is populated in _setup_adapter BEFORE
    any checkpoint is loaded, capturing the true pre-trained norms.
    """

    def __init__(self, model: Any, initial_magnitudes: dict[int, Any] | None = None) -> None:
        self._model = model
        self._initial_magnitudes = initial_magnitudes or {}
        # Cache LoRA/DoRA modules once — avoids named_modules() scan per forward
        self._lora_modules: list[Any] = []
        for name, module in model.named_modules():
            if not hasattr(module, "scale"):
                continue
            cls_name = type(module).__name__.lower()
            name_lower = name.lower()
            if (
                "lora" in name_lower
                or "dora" in name_lower
                or "lora" in cls_name
                or "dora" in cls_name
                or hasattr(module, "is_dora")
            ):
                self._lora_modules.append(module)

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        """Forward pass with adapters disabled.

        For DoRA: also resets magnitude vectors to pre-trained values so the
        effective weight recovers the original base model weight exactly.
        """
        saved_scales = [(m, m.scale) for m in self._lora_modules]
        # Save current trained magnitudes so we can restore them after ref forward
        saved_magnitudes = {
            id(m): mx.array(m.magnitude) for m in self._lora_modules if hasattr(m, "magnitude")
        }
        try:
            for m, _ in saved_scales:
                m.scale = 0.0
            # Reset magnitudes to pre-trained values: m_init * V/||V|| == V
            for m in self._lora_modules:
                init_m = self._initial_magnitudes.get(id(m))
                if init_m is not None and hasattr(m, "magnitude"):
                    m.magnitude = init_m
            output = self._model(input_ids, **kwargs)
            return mx.stop_gradient(output)
        finally:
            for m, original_scale in saved_scales:
                m.scale = original_scale
            # Restore trained magnitudes
            for m in self._lora_modules:
                saved_m = saved_magnitudes.get(id(m))
                if saved_m is not None:
                    m.magnitude = saved_m


class _SnapshotReferenceModel:
    """Reference model for full fine-tuning mode.

    Holds a frozen snapshot of the original weights. Swaps them in for
    each forward pass. More expensive than adapter ref but necessary
    when there are no adapter scales to toggle.
    """

    def __init__(self, model: Any, frozen_weights: dict[str, Any]) -> None:
        self._model = model
        self._frozen = frozen_weights
        # Pre-compute items list — avoids dict.items() + list() per forward
        self._frozen_items = list(frozen_weights.items())

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        """Forward pass with frozen snapshot weights."""
        current = _snapshot_weights(self._model)
        try:
            self._model.load_weights(self._frozen_items)
            output = self._model(input_ids, **kwargs)
            return mx.stop_gradient(output)
        finally:
            self._model.load_weights(list(current.items()))


class _MoEAdapterReferenceModel:
    """Reference model for LoRA-MoE training.

    Toggles all LoRAMoELinear layers to scale=0 (base output only)
    then restores scale=1 after the forward pass. Zero-copy and safe
    because base weights are frozen and scale is always restored.
    """

    def __init__(self, moe_model: Any) -> None:
        self._moe = moe_model

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        """Forward pass with LoRA experts disabled (scale=0)."""
        self._moe.set_scale(0.0)
        try:
            output = self._moe(input_ids, **kwargs)
            return mx.stop_gradient(output)
        finally:
            self._moe.set_scale(1.0)


def _load_moe_weights(moe_model: Any, path: str) -> None:
    """Load MoE adapter weights (experts + gate) from checkpoint."""
    adapter_path = Path(path)
    for filename in ("adapters.safetensors", "adapters.npz"):
        fpath = adapter_path / filename
        if not fpath.exists():
            continue
        weights = dict(mx.load(str(fpath)))
        # Detect MoE checkpoint by gate key prefix
        is_moe = any(k.startswith("_gate.") for k in weights)
        if is_moe:
            moe_model.load_weights(list(weights.items()), strict=False)
            _log.info("Loaded MoE adapter weights from %s (%d keys)", fpath, len(weights))
        else:
            _log.warning("Checkpoint %s has no _gate.* keys — not a MoE checkpoint", fpath)
        return
    _log.warning("No adapter weights found in %s", adapter_path)


def _snapshot_weights(model: Any) -> dict[str, Any]:
    """Deep copy all model weights for reference snapshot."""
    from mlx.utils import tree_flatten

    snapshot: dict[str, Any] = {}
    for k, v in tree_flatten(model.parameters()):
        snapshot[k] = mx.array(v)
    return snapshot


# ── Corruption Detection ────────────────────────────────────────────


class CorruptionError(RuntimeError):
    """Raised when model weights contain NaN/Inf values."""

    EXIT_CODE = 42


def validate_weights(model: Any, label: str = "model") -> None:
    """Check all model weights for NaN/Inf corruption.

    Should be called after loading weights and after each optimizer step
    (optionally). Raises CorruptionError if any parameter contains
    non-finite values.

    Args:
        model: MLX model to validate
        label: Human-readable label for error messages
    """
    if not HAS_MLX:
        return

    from mlx.utils import tree_flatten

    for name, param in tree_flatten(model.parameters()):
        if not hasattr(param, "shape"):
            continue
        has_nan = mx.any(mx.isnan(param)).item()
        has_inf = mx.any(mx.isinf(param)).item()
        if has_nan or has_inf:
            kind = "NaN" if has_nan else "Inf"
            raise CorruptionError(
                f"{label} weight corruption: {kind} in '{name}' "
                f"(shape={param.shape}, dtype={param.dtype})"
            )


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: pip install mlx mlx-lm (required for this module)
# Usage:
#   LoRA:  ModelLoader(ModelConfig(model_path="...", adapter_type="lora")).load()
#   DoRA:  ModelLoader(ModelConfig(model_path="...", adapter_type="dora")).load()
#   Full:  ModelLoader(ModelConfig(model_path="...", adapter_type="full")).load()
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
