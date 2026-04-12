"""Pure-numpy similarity and neuron metrics — no MLX dependency.

All functions accept numpy arrays. scipy is optional (guarded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

STAT_FIELDS = (
    "mean",
    "median",
    "std",
    "min",
    "max",
    "sparsity",
    "q01",
    "q25",
    "q75",
    "q99",
    "l1_norm",
    "l2_norm",
)


@dataclass(slots=True, frozen=True)
class LayerStats:
    layer_key: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    sparsity: float
    q01: float
    q25: float
    q75: float
    q99: float
    l1_norm: float
    l2_norm: float
    error: str | None = None


def compute_layer_stats(arr: npt.NDArray[Any], layer_key: str) -> LayerStats:
    """Compute descriptive statistics for a layer activation array."""
    flat = arr.ravel().astype(np.float64)
    if flat.size == 0:
        return LayerStats(
            layer_key=layer_key,
            mean=0.0,
            median=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            sparsity=0.0,
            q01=0.0,
            q25=0.0,
            q75=0.0,
            q99=0.0,
            l1_norm=0.0,
            l2_norm=0.0,
            error="empty array",
        )
    return LayerStats(
        layer_key=layer_key,
        mean=float(np.mean(flat)),
        median=float(np.median(flat)),
        std=float(np.std(flat)),
        min=float(np.min(flat)),
        max=float(np.max(flat)),
        # exact zeros (matches activ4.py _calculate_stats: np.mean(flat == 0))
        sparsity=float(np.mean(flat == 0.0)),
        q01=float(np.percentile(flat, 1)),
        q25=float(np.percentile(flat, 25)),
        q75=float(np.percentile(flat, 75)),
        q99=float(np.percentile(flat, 99)),
        l1_norm=float(np.sum(np.abs(flat))),
        l2_norm=float(np.linalg.norm(flat)),
    )


def cosine_similarity(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
    """Cosine similarity between two vectors."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def linear_cka(x_mat: npt.NDArray[Any], y_mat: npt.NDArray[Any]) -> float:
    """Centered Kernel Alignment (linear kernel) between representation matrices.

    x_mat, y_mat: (n_samples, d) — rows are samples, columns are features.
    Uses the efficient ||X^T Y||_F^2 formulation. Returns CKA in [0, 1].
    """
    x_mat = x_mat.astype(np.float64)
    y_mat = y_mat.astype(np.float64)
    n = x_mat.shape[0]
    if n < 2:
        return 1.0

    # Center
    x_mat = x_mat - x_mat.mean(axis=0, keepdims=True)
    y_mat = y_mat - y_mat.mean(axis=0, keepdims=True)

    # HSIC via ||X^T Y||_F^2 (efficient for many features, few samples)
    hsic = float(np.linalg.norm(x_mat.T @ y_mat, "fro") ** 2)
    var1 = float(np.linalg.norm(x_mat.T @ x_mat, "fro") ** 2)
    var2 = float(np.linalg.norm(y_mat.T @ y_mat, "fro") ** 2)

    import math  # noqa: PLC0415

    denom = math.sqrt(var1) * math.sqrt(var2) + 1e-12
    return float(hsic / denom)


def compute_layerwise_similarity(
    sketches_a: dict[str, npt.NDArray[Any]],
    sketches_b: dict[str, npt.NDArray[Any]],
) -> dict[str, dict[str, float]]:
    """Per-layer cosine and CKA similarity between two sketch dicts.

    Each sketch is shape (seq, proj_dim) or (proj_dim,) for single token.
    Returns {layer_key: {"cosine": float, "cka": float}}.
    """
    result: dict[str, dict[str, float]] = {}
    for key in sketches_a:
        if key not in sketches_b:
            continue
        a = np.atleast_2d(sketches_a[key].astype(np.float64))
        b = np.atleast_2d(sketches_b[key].astype(np.float64))
        # Align feature dims (may differ if hidden dims differ)
        if a.shape[1] != b.shape[1]:
            d = min(a.shape[1], b.shape[1])
            a, b = a[:, :d], b[:, :d]
        cos = cosine_similarity(a.ravel(), b.ravel())
        cka = linear_cka(a, b)
        result[key] = {"cosine": cos, "cka": cka}
    return result


def rsa_matrix(acts: npt.NDArray[Any], n_subsample: int = 200) -> npt.NDArray[np.float32]:
    """Representational Similarity Analysis — returns flat upper-triangle vector.

    acts: (n_prompts, d). Returns 1D cosine-dissimilarity upper-triangle.
    """
    acts = acts.astype(np.float64)
    n = acts.shape[0]
    if n > n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, n_subsample, replace=False)
        acts = acts[idx]
    norms = np.linalg.norm(acts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = acts / norms
    dist = 1.0 - (normed @ normed.T)
    iu = np.triu_indices(dist.shape[0], k=1)
    return np.asarray(dist[iu], dtype=np.float32)


def rsa_spearman(va: npt.NDArray[Any], vb: npt.NDArray[Any]) -> float:
    """Spearman correlation between two flat upper-triangle dissimilarity vectors.

    Returns nan if scipy unavailable or arrays too small.
    """
    try:
        from scipy.stats import spearmanr  # noqa: PLC0415
    except ImportError:
        return float("nan")
    if va.size < 3 or vb.size < 3:
        return float("nan")
    r, _ = spearmanr(va, vb)
    return float(r)


def compute_rsa_layerwise(
    raw_a: dict[str, npt.NDArray[Any]],
    raw_b: dict[str, npt.NDArray[Any]],
) -> dict[str, float]:
    """Per-layer RSA Spearman scores. raw arrays shape (n_prompts, hidden) or (hidden,)."""
    result: dict[str, float] = {}
    for key in raw_a:
        if key not in raw_b:
            continue
        a = np.atleast_2d(raw_a[key].astype(np.float64))
        b = np.atleast_2d(raw_b[key].astype(np.float64))
        try:
            va = rsa_matrix(a)
            vb = rsa_matrix(b)
            result[key] = rsa_spearman(va, vb)
        except Exception:  # noqa: BLE001
            result[key] = float("nan")
    return result


def neuron_jaccard(a: npt.NDArray[Any], b: npt.NDArray[Any], threshold: float = 0.01) -> float:
    """Fraction of neurons active in both / active in either.

    For 2D inputs (seq, hidden), computes mean activation magnitude per neuron
    across sequence positions first (matching activ4.py behaviour).
    """
    # mean over sequence axis if 2D
    ma = np.abs(a).mean(axis=0) if a.ndim > 1 else np.abs(a)
    mb = np.abs(b).mean(axis=0) if b.ndim > 1 else np.abs(b)
    # align dims
    n = min(ma.size, mb.size)
    ma, mb = ma.ravel()[:n], mb.ravel()[:n]
    active_a = ma > threshold
    active_b = mb > threshold
    union = int(np.sum(active_a | active_b))
    if union == 0:
        return 1.0
    return float(np.sum(active_a & active_b) / union)


def neuron_sign_consistency(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
    """Fraction of neurons with the same sign.

    For 2D inputs (seq, hidden), takes mean over axis=0 first.
    """
    ma = a.mean(axis=0) if a.ndim > 1 else a
    mb = b.mean(axis=0) if b.ndim > 1 else b
    n = min(ma.size, mb.size)
    return float(np.mean(np.sign(ma.ravel()[:n]) == np.sign(mb.ravel()[:n])))


def l2_norm_diff_pct(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
    """Relative RMS norm difference as percentage.

    Returns abs(rms_b - rms_a) / rms_a * 100. Matches activ4.py _l2_norm_diff_pct.
    """
    rms_a = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    rms_b = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
    if rms_a < 1e-9:
        return 0.0
    return abs(rms_b - rms_a) / rms_a * 100.0


def compute_neuron_metrics(
    raw_a: dict[str, npt.NDArray[Any]],
    raw_b: dict[str, npt.NDArray[Any]],
) -> dict[str, dict[str, float]]:
    """Per-layer Jaccard, sign consistency, L2 diff, and sparsity diff.

    sparsity_diff_pct = abs(sparsity_b - sparsity_a) / (sparsity_a + 1e-12) * 100
    where sparsity is the fraction of exact-zero values (matching activ4.py).
    """
    result: dict[str, dict[str, float]] = {}
    for key in raw_a:
        if key not in raw_b:
            continue
        a = raw_a[key].astype(np.float32)
        b = raw_b[key].astype(np.float32)
        # sparsity: fraction of exact zeros (as in activ4.py _calculate_stats)
        sp_a = float(np.mean(a.ravel() == 0.0))
        sp_b = float(np.mean(b.ravel() == 0.0))
        sp_diff = abs(sp_b - sp_a) / (sp_a + 1e-12) * 100.0
        result[key] = {
            "jaccard": neuron_jaccard(a, b),
            "sign_consistency": neuron_sign_consistency(a, b),
            "l2_diff_pct": l2_norm_diff_pct(a, b),
            "sparsity_diff_pct": sp_diff,
        }
    return result


def compute_sublayer_similarity(
    acts_a: dict[str, Any],
    acts_b: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    """Per-layer attn_out, mlp_out, and full-layer cosine similarity.

    Matches activ4.py sublayer_sim format with attn_cos, mlp_cos, full_cos keys.
    """
    result: dict[str, dict[str, float | None]] = {}
    for key in acts_a:
        if key not in acts_b:
            continue
        la = acts_a[key]
        lb = acts_b[key]

        attn_cos: float | None = None
        mlp_cos: float | None = None
        full_cos: float | None = None

        if la.attn_out is not None and lb.attn_out is not None:
            va = np.asarray(la.attn_out).ravel().astype(np.float32)
            vb = np.asarray(lb.attn_out).ravel().astype(np.float32)
            # mean over seq if 2D
            if la.attn_out.ndim > 1 if hasattr(la.attn_out, "ndim") else False:
                va = np.asarray(la.attn_out).mean(axis=0).ravel().astype(np.float32)
                vb = np.asarray(lb.attn_out).mean(axis=0).ravel().astype(np.float32)
            attn_cos = cosine_similarity(va, vb)

        if la.mlp_out is not None and lb.mlp_out is not None:
            va = np.asarray(la.mlp_out).ravel().astype(np.float32)
            vb = np.asarray(lb.mlp_out).ravel().astype(np.float32)
            if la.mlp_out.ndim > 1 if hasattr(la.mlp_out, "ndim") else False:
                va = np.asarray(la.mlp_out).mean(axis=0).ravel().astype(np.float32)
                vb = np.asarray(lb.mlp_out).mean(axis=0).ravel().astype(np.float32)
            mlp_cos = cosine_similarity(va, vb)

        # full-layer cosine from hidden state (stored in raw_np equivalent)
        if (
            hasattr(la, "full_out")
            and la.full_out is not None
            and hasattr(lb, "full_out")
            and lb.full_out is not None
        ):
            fa = np.asarray(la.full_out).ravel().astype(np.float32)
            fb = np.asarray(lb.full_out).ravel().astype(np.float32)
            full_cos = cosine_similarity(fa, fb)

        result[key] = {"attn_cos": attn_cos, "mlp_cos": mlp_cos, "full_cos": full_cos}
    return result
