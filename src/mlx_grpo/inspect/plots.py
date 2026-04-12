"""Matplotlib plots for inspect subcommand.

All functions are matplotlib-dependent — excluded from coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def plot_shaman_validation(  # pragma: no cover
    similarity: dict[str, dict[str, float]],
    rsa_scores: dict[str, float],
    sublayer_sim: dict[str, Any],
    neuron_metrics: dict[str, dict[str, float]],
    name_a: str,
    name_b: str,
    out_path: Path,
    num_layers: int = 36,
) -> None:
    """4-panel SHAMAN validation plot."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    keys = sorted(similarity.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0)
    layer_idxs = [int(k.split("_")[-1]) if "_" in k else i for i, k in enumerate(keys)]

    cosines = [similarity[k].get("cosine", 0.0) for k in keys]
    ckas = [similarity[k].get("cka", 0.0) for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"SHAMAN Analysis: {name_a} vs {name_b}", fontsize=13)

    # Panel 1: Cosine + CKA
    ax = axes[0, 0]
    ax.plot(layer_idxs, cosines, "b-o", markersize=3, label="Cosine")
    ax.plot(layer_idxs, ckas, "r-s", markersize=3, label="CKA")
    ax.set_title("Layer Similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Similarity")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: RSA Spearman
    ax = axes[0, 1]
    rsa_keys = sorted(rsa_scores.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0)
    rsa_vals = [rsa_scores[k] for k in rsa_keys]
    rsa_layer_idxs = [int(k.split("_")[-1]) if "_" in k else i for i, k in enumerate(rsa_keys)]
    valid = [(x, y) for x, y in zip(rsa_layer_idxs, rsa_vals, strict=False) if y == y]
    if valid:
        rx, ry = zip(*valid, strict=False)
        ax.plot(rx, ry, "g-^", markersize=3)
    ax.set_title("RSA Spearman")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman r")
    ax.grid(alpha=0.3)

    # Panel 3: Neuron Jaccard
    ax = axes[1, 0]
    nm_keys = sorted(neuron_metrics.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0)
    jac_vals = [neuron_metrics[k].get("jaccard", 0.0) for k in nm_keys]
    nm_idxs = [int(k.split("_")[-1]) if "_" in k else i for i, k in enumerate(nm_keys)]
    ax.bar(nm_idxs, jac_vals, color="purple", alpha=0.6)
    ax.set_title("Neuron Jaccard Similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard")
    ax.grid(alpha=0.3)

    # Panel 4: Sublayer attn vs MLP
    ax = axes[1, 1]
    sl_keys = sorted(sublayer_sim.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0)
    attn_cos_vals = [sublayer_sim[k].get("attn_cos") for k in sl_keys]
    mlp_cos_vals = [sublayer_sim[k].get("mlp_cos") for k in sl_keys]
    sl_idxs = [int(k.split("_")[-1]) if "_" in k else i for i, k in enumerate(sl_keys)]

    valid_attn = [(x, y) for x, y in zip(sl_idxs, attn_cos_vals, strict=False) if y is not None]
    valid_mlp = [(x, y) for x, y in zip(sl_idxs, mlp_cos_vals, strict=False) if y is not None]
    if valid_attn:
        ax.plot(*zip(*valid_attn, strict=False), "b-o", markersize=3, label="Attn")
    if valid_mlp:
        ax.plot(*zip(*valid_mlp, strict=False), "r-s", markersize=3, label="MLP")
    ax.set_title("Sublayer Similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_circuit_comparison(  # pragma: no cover
    trace_a: Any,
    trace_b: Any,
    out_path: Path,
) -> None:
    """4-panel circuit comparison between two models."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Circuit Comparison: {trace_a.model_name} vs {trace_b.model_name}", fontsize=13)

    for col, (trace, name) in enumerate(
        [(trace_a, trace_a.model_name), (trace_b, trace_b.model_name)]
    ):
        layers = sorted(trace.prompt_traces.keys())
        imp_means = [trace.prompt_traces[li].head_importances.mean() for li in layers]
        ent_means = [trace.prompt_traces[li].attn_entropy.mean() for li in layers]

        ax = axes[0, col]
        ax.plot(layers, imp_means, "b-o", markersize=3)
        ax.set_title(f"{name}: Head Importance")
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(layers, ent_means, "r-s", markersize=3)
        ax.set_title(f"{name}: Attn Entropy")
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_comparison(  # pragma: no cover
    similarity: dict[str, dict[str, float]],
    name_a: str,
    name_b: str,
    out_path: Path,
) -> None:
    """Simple cosine+CKA comparison plot."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    keys = sorted(similarity.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0)
    layer_idxs = [int(k.split("_")[-1]) if "_" in k else i for i, k in enumerate(keys)]
    cosines = [similarity[k].get("cosine", 0.0) for k in keys]
    ckas = [similarity[k].get("cka", 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layer_idxs, cosines, "b-o", markersize=3, label="Cosine")
    ax.plot(layer_idxs, ckas, "r-s", markersize=3, label="CKA")
    ax.set_title(f"Layer Similarity: {name_a} vs {name_b}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Similarity")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cka_trajectory(  # pragma: no cover
    checkpoint_labels: list[str],
    layer_cka_matrix: np.ndarray,
    out_path: Path,
) -> None:
    """Heatmap of CKA drift per checkpoint × layer (Feature 1)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(layer_cka_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
    plt.colorbar(im, ax=ax, label="CKA")
    ax.set_yticks(range(len(checkpoint_labels)))
    ax.set_yticklabels(checkpoint_labels, fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Checkpoint")
    ax.set_title("CKA Drift Across Checkpoints")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_stratified(  # pragma: no cover
    high_sim: dict[str, dict[str, float]],
    low_sim: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Two-line CKA plot: high vs low reward rollouts (Feature 2)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    def _extract(sim: dict[str, dict[str, float]], metric: str) -> tuple[list[int], list[float]]:
        keys = sorted(sim.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0)
        idxs = [int(k.split("_")[-1]) if "_" in k else i for i, k in enumerate(keys)]
        vals = [sim[k].get(metric, 0.0) for k in keys]
        return idxs, vals

    fig, ax = plt.subplots(figsize=(10, 5))
    hi_x, hi_y = _extract(high_sim, "cka")
    lo_x, lo_y = _extract(low_sim, "cka")
    ax.plot(hi_x, hi_y, "g-o", markersize=3, label="High reward")
    ax.plot(lo_x, lo_y, "r-s", markersize=3, label="Low reward")
    ax.set_title("CKA by Reward Stratum")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
