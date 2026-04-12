"""Circuit analysis — head importance tracing.

MLX-dependent: excluded from coverage.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class HeadImportanceTrace:
    """Attention head importance at a single layer."""

    layer_idx: int
    head_importances: np.ndarray  # (n_heads,) normalized V-proj norms
    head_norms: np.ndarray
    attn_entropy: np.ndarray  # (n_heads,) entropy of Q·K^T weights


@dataclass(slots=True)
class CircuitTrace:
    """Full circuit trace for a prompt + optional generation steps."""

    model_name: str
    prompt_text: str
    generated_text: str
    prompt_traces: dict[int, HeadImportanceTrace]
    step_traces: dict[int, dict[int, HeadImportanceTrace]] = field(default_factory=dict)
    layer_norms: dict[int, float] = field(default_factory=dict)
    layer_entropy: dict[int, float] = field(default_factory=dict)


def compute_head_importance(
    attn_mod: Any,
    h_normed: Any,
    n_heads: int,
    head_dim: int,
) -> HeadImportanceTrace:  # pragma: no cover
    """Compute per-head importance from V-projection norms and Q·K entropy."""
    import mlx.core as mx  # noqa: PLC0415

    layer_idx = 0
    with contextlib.suppress(Exception):
        layer_idx = getattr(attn_mod, "_layer_idx", 0)

    # V-projection norms as proxy for head importance
    v_proj = getattr(attn_mod, "v_proj", None)
    if v_proj is not None and hasattr(v_proj, "weight"):
        w = np.array(v_proj.weight.tolist(), dtype=np.float32)
        # shape (n_heads * head_dim, hidden) → (n_heads, head_dim, hidden)
        w3d = w.reshape(n_heads, head_dim, -1)
        head_norms = np.linalg.norm(w3d, axis=(1, 2))
        total = head_norms.sum()
        head_importances = head_norms / (total + 1e-12)
    else:
        head_norms = np.ones(n_heads, dtype=np.float32)
        head_importances = head_norms / n_heads

    # Attempt to compute Q·K entropy
    attn_entropy = np.zeros(n_heads, dtype=np.float32)
    q_proj = getattr(attn_mod, "q_proj", None)
    k_proj = getattr(attn_mod, "k_proj", None)
    if q_proj is not None and k_proj is not None:
        try:
            q = q_proj(h_normed)  # (1, seq, n_heads*head_dim)
            k = k_proj(h_normed)
            seq = q.shape[1]
            q_r = q.reshape(1, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            k_r = k.reshape(1, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            scale = head_dim**-0.5
            scores = (q_r @ k_r.transpose(0, 1, 3, 2)) * scale  # (1,H,S,S)
            mx.eval(scores)
            scores_np = np.array(scores[0].tolist(), dtype=np.float32)
            # Softmax per head
            scores_np = scores_np - scores_np.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores_np)
            probs = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-12)
            # Entropy per head (mean over query positions)
            ent = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
            attn_entropy = ent.mean(axis=-1)  # (n_heads,)
        except Exception:  # noqa: BLE001
            pass

    return HeadImportanceTrace(
        layer_idx=layer_idx,
        head_importances=head_importances,
        head_norms=head_norms,
        attn_entropy=attn_entropy,
    )


def run_circuit_forward(
    model: Any,
    h: Any,
    circuit_layers: tuple[int, ...] | None = None,
) -> dict[int, HeadImportanceTrace]:  # pragma: no cover
    """Run a forward pass collecting circuit info at specified layers."""
    layers = getattr(model, "layers", getattr(model.model, "layers", []))
    traces: dict[int, HeadImportanceTrace] = {}

    for i, layer in enumerate(layers):
        if circuit_layers and i not in circuit_layers:
            h = layer(h)
            continue

        attn_mod = getattr(layer, "self_attn", None)
        normed = getattr(layer, "input_layernorm", None)
        h_normed = normed(h) if normed is not None else h

        n_heads = getattr(attn_mod, "n_heads", 16) if attn_mod is not None else 16
        head_dim = getattr(attn_mod, "head_dim", 64) if attn_mod is not None else 64

        if attn_mod is not None:
            trace = compute_head_importance(attn_mod, h_normed, n_heads, head_dim)
            trace.layer_idx = i
            traces[i] = trace

        h = layer(h)

    return traces


def trace_model_circuits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    model_name: str,
    cfg: Any,
) -> CircuitTrace:  # pragma: no cover
    """Full circuit trace for a prompt, optionally tracing generation steps."""
    import mlx.core as mx  # noqa: PLC0415

    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])

    embed = model.model.embed_tokens
    h = embed(x)

    circuit_layers = cfg.circuit_layers if cfg.circuit_layers else None
    prompt_traces = run_circuit_forward(model, h, circuit_layers)

    # Compute layer-wise output norms and mean entropy
    layer_norms: dict[int, float] = {}
    layer_entropy: dict[int, float] = {}
    for i, trace in prompt_traces.items():
        layer_norms[i] = float(trace.head_norms.mean())
        layer_entropy[i] = float(trace.attn_entropy.mean())

    # Generation steps (optional)
    step_traces: dict[int, dict[int, HeadImportanceTrace]] = {}
    generated_text = ""

    if cfg.trace_generation and cfg.max_new_tokens > 0:
        from mlx_lm import generate  # noqa: PLC0415

        out = generate(
            model, tokenizer, prompt=prompt, max_tokens=cfg.max_new_tokens, verbose=False
        )
        generated_text = out

    return CircuitTrace(
        model_name=model_name,
        prompt_text=prompt,
        generated_text=generated_text,
        prompt_traces=prompt_traces,
        step_traces=step_traces,
        layer_norms=layer_norms,
        layer_entropy=layer_entropy,
    )
