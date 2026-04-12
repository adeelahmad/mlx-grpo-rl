"""ActivationCollector — MLX boundary.

All MLX imports are isolated here. Everything else in the package is pure-numpy.
This module is excluded from coverage (requires real GPU + model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


# ── Model architecture helpers (mirrors activ4.py) ──────────────────────────


def _get_base_model(model: Any) -> Any:
    """Return the inner model that has .layers (handles model.model pattern)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model


def _get_embed_func(base_model: Any) -> Any | None:
    """Find the embedding function across common attribute names."""
    for attr in ("embed_tokens", "tok_embeddings", "wte", "embedding"):
        obj = getattr(base_model, attr, None)
        if obj is not None and callable(obj):
            return obj
    return None


def _get_attn_module(layer: Any) -> Any | None:
    return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)


def _get_mlp_module(layer: Any) -> Any | None:
    return getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)


def _get_norm1(layer: Any) -> Any | None:
    return getattr(layer, "input_layernorm", None) or getattr(layer, "ln_1", None)


def _get_norm2(layer: Any) -> Any | None:
    return getattr(layer, "post_attention_layernorm", None) or getattr(layer, "ln_2", None)


def _layer_forward(layer: Any, h: Any) -> Any:
    """Call layer with optional mask/cache kwargs based on its signature."""
    import inspect as _inspect  # noqa: PLC0415

    try:
        sig = _inspect.signature(layer)
        kwargs = {k: None for k in ("mask", "cache") if k in sig.parameters}
        out = layer(h, **kwargs)
        return out[0] if isinstance(out, tuple) else out
    except Exception:  # noqa: BLE001
        return h


def _format_with_template(tokenizer: Any, prompt: str) -> str:
    """Apply chat template if available, else return prompt as-is."""
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            out = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if out:
                return str(out)
    except Exception:  # noqa: BLE001
        pass
    return prompt


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class LayerActivations:
    """Activations captured at a single transformer layer."""

    key: str
    hidden: Any  # mx.array (1, seq, hidden_dim)
    attn_out: Any | None = None  # (seq, hidden) numpy, when capture_sublayers=True
    mlp_out: Any | None = None  # (seq, hidden) numpy, when capture_sublayers=True
    full_out: Any | None = None  # (seq, hidden) numpy full-layer output


@dataclass(slots=True, frozen=True)
class CollectionResult:
    """Result of collecting activations for a single prompt."""

    layer_acts: dict[str, LayerActivations]
    sketches: dict[str, np.ndarray]  # (seq, proj_dim) per layer, for CKA
    raw_np: dict[str, np.ndarray]  # (seq, hidden) full-sequence, for RSA/neuron metrics
    prompt_ids: list[int]
    track_pos: int


def _get_projection_matrix(hidden_dim: int, proj_dim: int, seed: int) -> np.ndarray:
    """Gaussian random projection matrix, shape (hidden_dim, proj_dim)."""
    # Match activ4.py _get_projection_matrix seed formula
    rs = np.random.RandomState(seed + hidden_dim * 7 + proj_dim * 13)
    return rs.normal(0.0, 1.0 / np.sqrt(proj_dim), (hidden_dim, proj_dim)).astype(np.float32)


def load_model_for_inspect(model_path: str) -> tuple[Any, Any]:
    """Load an MLX-LM model+tokenizer from a path."""
    from mlx_lm.utils import load  # noqa: PLC0415

    return load(model_path)  # type: ignore[return-value]


class ActivationCollector:
    """Collects per-layer hidden-state activations from an MLX transformer.

    Usage::
        collector = ActivationCollector(model, tokenizer, cfg)
        result = collector.collect("What is 2+2?")

    raw_np stores full (seq, hidden) arrays — matching activ4.py run_and_track_enhanced.
    sketches stores (seq, proj_dim) random projections for CKA comparison.
    """

    def __init__(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._proj_matrices: dict[int, np.ndarray] = {}

    def _get_proj(self, hidden_dim: int) -> np.ndarray:
        if hidden_dim not in self._proj_matrices:
            self._proj_matrices[hidden_dim] = _get_projection_matrix(
                hidden_dim, self._cfg.proj_dim, self._cfg.sketch_seed
            )
        return self._proj_matrices[hidden_dim]

    def collect(self, prompt: str) -> CollectionResult:
        """Run a forward pass and collect activations per layer."""
        import mlx.core as mx  # noqa: PLC0415

        tokenizer = self._tokenizer
        model = self._model

        # Apply chat template (matches activ4.py _format_with_template)
        fmt_prompt = _format_with_template(tokenizer, prompt)
        tokens = tokenizer.encode(fmt_prompt)
        prompt_ids = list(tokens)

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            # Last non-pad position
            track_pos = len(prompt_ids) - 1
            for i in range(len(prompt_ids) - 1, -1, -1):
                if prompt_ids[i] != pad_id:
                    track_pos = i
                    break
        else:
            track_pos = len(prompt_ids) - 1

        x = mx.array([prompt_ids])
        base_model = _get_base_model(model)
        embed_fn = _get_embed_func(base_model)
        if embed_fn is None:
            return CollectionResult(
                layer_acts={},
                sketches={},
                raw_np={},
                prompt_ids=prompt_ids,
                track_pos=track_pos,
            )

        h = embed_fn(x)
        mx.eval(h)

        layer_acts: dict[str, LayerActivations] = {}
        sketches: dict[str, np.ndarray] = {}
        raw_np: dict[str, np.ndarray] = {}

        # Store embed activations
        arr_embed = np.array(h[0].tolist(), dtype=np.float32)  # (seq, hidden)
        hidden_dim = arr_embed.shape[-1]
        proj = self._get_proj(hidden_dim)
        sketches["embed"] = (arr_embed @ proj).astype(np.float32)
        raw_np["embed"] = arr_embed

        layers = base_model.layers
        for i, layer in enumerate(layers):
            key = f"layer_{i}"
            attn_out_np: np.ndarray | None = None
            mlp_out_np: np.ndarray | None = None

            if self._cfg.capture_sublayers:
                # Sublayer capture: attn, mlp, full (matching _extract_sublayer_outputs)
                norm1 = _get_norm1(layer)
                norm2 = _get_norm2(layer)
                attn_mod = _get_attn_module(layer)
                mlp_mod = _get_mlp_module(layer)

                attn_out_mx = None
                if attn_mod is not None and norm1 is not None:
                    try:
                        h_normed = norm1(h)
                        raw_attn = attn_mod(h_normed)
                        attn_out_mx = raw_attn[0] if isinstance(raw_attn, tuple) else raw_attn
                        mx.eval(attn_out_mx)
                        attn_out_np = np.array(attn_out_mx[0].tolist(), dtype=np.float32)
                    except Exception:  # noqa: BLE001
                        attn_out_mx = None

                if mlp_mod is not None and norm2 is not None:
                    try:
                        h_after_attn = (h + attn_out_mx) if attn_out_mx is not None else h
                        h_normed2 = norm2(h_after_attn)
                        mlp_out_mx = mlp_mod(h_normed2)
                        mx.eval(mlp_out_mx)
                        mlp_out_np = np.array(mlp_out_mx[0].tolist(), dtype=np.float32)
                    except Exception:  # noqa: BLE001
                        pass

            h = _layer_forward(layer, h)
            mx.eval(h)

            # Full sequence (seq, hidden) stored in raw_np
            arr = np.array(h[0].tolist(), dtype=np.float32)
            sketch = (arr @ proj).astype(np.float32)

            la = LayerActivations(
                key=key,
                hidden=h,
                attn_out=attn_out_np,
                mlp_out=mlp_out_np,
                full_out=arr,
            )
            layer_acts[key] = la
            sketches[key] = sketch
            raw_np[key] = arr

        # Final norm
        fn = getattr(base_model, "norm", None)
        if callable(fn):
            try:
                h_normed = fn(h)
                mx.eval(h_normed)
                arr_fn = np.array(h_normed[0].tolist(), dtype=np.float32)
                sketches["final_norm"] = (arr_fn @ proj).astype(np.float32)
                raw_np["final_norm"] = arr_fn
            except Exception:  # noqa: BLE001
                pass

        return CollectionResult(
            layer_acts=layer_acts,
            sketches=sketches,
            raw_np=raw_np,
            prompt_ids=prompt_ids,
            track_pos=track_pos,
        )
