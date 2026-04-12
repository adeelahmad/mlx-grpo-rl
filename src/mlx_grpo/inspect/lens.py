"""Logit lens and token trajectory analysis.

logit_lens() and plot_token_trajectory() are MLX/matplotlib-dependent — excluded
from coverage. compute_token_trajectory() is pure-numpy and fully testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class LogitLensRow:
    """Top-k token predictions at a layer position."""

    layer_key: str
    position: int
    top_tokens: list[tuple[str, float, int]]  # (text, prob, id)


def logit_lens(  # pragma: no cover
    model: Any,
    base_model: Any,
    tokenizer: Any,
    layer_hiddens: dict[str, Any],
    track_pos: int,
    top_k: int = 5,
) -> list[LogitLensRow]:
    """Project each layer's hidden state through the LM head to get token probs.

    Uses the final LayerNorm + lm_head from base_model (or model if no base_model).
    """
    import mlx.core as mx  # noqa: PLC0415

    lm = base_model if base_model is not None else model
    norm = getattr(lm.model, "norm", None)
    lm_head = getattr(lm, "lm_head", None)

    rows: list[LogitLensRow] = []

    for key, la in layer_hiddens.items():
        h = la.hidden  # (1, seq, hidden)
        try:
            h_norm = norm(h) if norm is not None else h
            logits = lm_head(h_norm) if lm_head is not None else None
            if logits is None:
                continue
            mx.eval(logits)
            logits_np = np.array(logits[0, track_pos].tolist(), dtype=np.float32)
            # Softmax
            logits_np -= logits_np.max()
            exp_l = np.exp(logits_np)
            probs = exp_l / (exp_l.sum() + 1e-12)
            top_ids = np.argsort(probs)[::-1][:top_k]
            top_tokens = []
            for tid in top_ids:
                text = tokenizer.decode([int(tid)])
                top_tokens.append((text, float(probs[tid]), int(tid)))
            rows.append(LogitLensRow(layer_key=key, position=track_pos, top_tokens=top_tokens))
        except Exception:  # noqa: BLE001
            pass

    return rows


def save_logit_lens_csv(rows: list[LogitLensRow], path: Path) -> None:
    """Save logit lens rows to a CSV file."""
    import csv  # noqa: PLC0415

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "position", "rank", "token", "prob", "token_id"])
        for row in rows:
            for rank, (text, prob, tid) in enumerate(row.top_tokens):
                writer.writerow([row.layer_key, row.position, rank, text, f"{prob:.6f}", tid])


def compute_token_trajectory(
    layer_vecs: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """PCA-reduce a sequence of layer vectors to 2D for trajectory visualization.

    Returns (coords, explained_variance_ratio) where coords is (n_layers, 2).
    Pure-numpy — fully testable.
    """
    if len(layer_vecs) < 2:
        coords = np.zeros((len(layer_vecs), 2), dtype=np.float32)
        return coords, 0.0

    mat = np.stack([v.ravel().astype(np.float64) for v in layer_vecs])
    mat -= mat.mean(axis=0, keepdims=True)

    # SVD-based PCA
    _, s, vt = np.linalg.svd(mat, full_matrices=False)
    coords = (mat @ vt[:2].T).astype(np.float32)

    total_var = float((s**2).sum())
    explained = float((s[:2] ** 2).sum() / max(total_var, 1e-12))

    return coords, explained


def plot_token_trajectory(  # pragma: no cover
    coords: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Plot 2D PCA trajectory across layers."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8, 6))
    n = coords.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    for i in range(n - 1):
        ax.annotate(
            "",
            xy=coords[i + 1],
            xytext=coords[i],
            arrowprops={"arrowstyle": "->", "color": colors[i]},
        )

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(n), cmap="viridis", zorder=3)
    plt.colorbar(sc, ax=ax, label="Layer")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
