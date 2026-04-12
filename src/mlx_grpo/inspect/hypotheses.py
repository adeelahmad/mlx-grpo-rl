"""SHAMAN hypothesis evaluator — pure-python, no MLX dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SHAMAN_HYPOTHESES: dict[str, str] = {
    "H1_early_preservation": "Early layers (0-9): cos>=0.9999, CKA>=0.9999",
    "H2_mid_rewiring": "Mid layers (10-22): sparsity diff 7-22%, L2 diff <5%",
    "H3_deep_geometric": "Deep layers (23-35): CKA drops faster than cosine",
    "H4_task_specificity": "Control prompt: cos>=0.9998, CKA>=0.9998 everywhere",
    "H5_sublayer_attn_vs_mlp": "Mid layers: attn vs MLP sublayer divergence quantified",
}


@dataclass(slots=True, frozen=True)
class HypothesisResult:
    passed: bool | None  # None = not applicable
    description: str
    evidence: str
    metrics: dict[str, float] = field(default_factory=dict)


def _layer_range(
    similarity: dict[str, dict[str, float]], lo: int, hi: int
) -> list[dict[str, float]]:
    """Return similarity dicts for layers in [lo, hi] range."""
    rows = []
    for key, vals in similarity.items():
        try:
            idx = int(key.split("_")[-1]) if "_" in key else int(key)
        except ValueError:
            continue
        if lo <= idx <= hi:
            rows.append(vals)
    return rows


def _mean_metric(rows: list[dict[str, float]], metric: str) -> float:
    vals = [r[metric] for r in rows if metric in r]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def evaluate_hypotheses(
    similarity: dict[str, dict[str, float]],
    neuron_metrics: dict[str, dict[str, float]],
    sublayer_sim: dict[str, Any],
    control_sim: dict[str, dict[str, float]] | None,
    num_layers: int,
) -> dict[str, HypothesisResult]:
    """Evaluate all five SHAMAN hypotheses.

    Returns dict keyed by hypothesis name → HypothesisResult.
    """
    results: dict[str, HypothesisResult] = {}

    # H1: Early layers (0–9) preservation
    early = _layer_range(similarity, 0, 9)
    if early:
        mean_cos = _mean_metric(early, "cosine")
        mean_cka = _mean_metric(early, "cka")
        passed_h1 = mean_cos >= 0.9999 and mean_cka >= 0.9999
        results["H1_early_preservation"] = HypothesisResult(
            passed=passed_h1,
            description=SHAMAN_HYPOTHESES["H1_early_preservation"],
            evidence=(
                f"Early layers mean cosine={mean_cos:.6f}, mean CKA={mean_cka:.6f}. "
                f"Threshold: both ≥ 0.9999."
            ),
            metrics={"mean_cosine": mean_cos, "mean_cka": mean_cka},
        )
    else:
        results["H1_early_preservation"] = HypothesisResult(
            passed=None,
            description=SHAMAN_HYPOTHESES["H1_early_preservation"],
            evidence="No layers in range 0–9 found in similarity dict.",
            metrics={},
        )

    # H2: Mid layers (10–22) rewiring — uses sparsity_diff_pct from neuron_metrics
    mid_neuro: list[dict[str, float]] = []
    for key, vals in neuron_metrics.items():
        try:
            idx = int(key.split("_")[-1]) if "_" in key else int(key)
        except ValueError:
            continue
        if 10 <= idx <= 22:
            mid_neuro.append(vals)

    if mid_neuro:
        sp_diffs = [v["sparsity_diff_pct"] for v in mid_neuro if "sparsity_diff_pct" in v]
        l2_diffs = [v["l2_diff_pct"] for v in mid_neuro if "l2_diff_pct" in v]

        if sp_diffs and l2_diffs:
            mean_sp = sum(sp_diffs) / len(sp_diffs)
            mean_l2 = sum(l2_diffs) / len(l2_diffs)
            # Match activ4.py: sp_ok = all(7 <= s <= 30 for s in sp_diffs), l2_ok = mean < 5%
            sp_ok = all(7.0 <= s <= 30.0 for s in sp_diffs)
            l2_ok = mean_l2 < 5.0
            passed_h2 = sp_ok and l2_ok
            results["H2_mid_rewiring"] = HypothesisResult(
                passed=passed_h2,
                description=SHAMAN_HYPOTHESES["H2_mid_rewiring"],
                evidence=(
                    f"Mid layers mean sparsity_diff={mean_sp:.1f}% "
                    f"(7-30% expected), mean L2_diff={mean_l2:.2f}% (<5% expected)."
                ),
                metrics={"sparsity_diff_pct": mean_sp, "l2_diff_pct": mean_l2},
            )
        else:
            results["H2_mid_rewiring"] = HypothesisResult(
                passed=None,
                description=SHAMAN_HYPOTHESES["H2_mid_rewiring"],
                evidence="sparsity_diff_pct or l2_diff_pct missing from neuron_metrics.",
                metrics={},
            )
    else:
        results["H2_mid_rewiring"] = HypothesisResult(
            passed=None,
            description=SHAMAN_HYPOTHESES["H2_mid_rewiring"],
            evidence="Insufficient data for mid layers 10–22.",
            metrics={},
        )

    # H3: Deep layers (23–35) CKA drops faster than cosine
    deep = _layer_range(similarity, 23, 35)
    if deep:
        cka_vals = [r["cka"] for r in deep if "cka" in r]
        cos_vals = [r["cosine"] for r in deep if "cosine" in r]
        if cka_vals and cos_vals:
            med_cka = sorted(cka_vals)[len(cka_vals) // 2]
            med_cos = sorted(cos_vals)[len(cos_vals) // 2]
            passed_h3 = med_cka < med_cos
            results["H3_deep_geometric"] = HypothesisResult(
                passed=passed_h3,
                description=SHAMAN_HYPOTHESES["H3_deep_geometric"],
                evidence=(
                    f"Deep layers median CKA={med_cka:.6f}, median cosine={med_cos:.6f}. "
                    f"CKA < cosine: {passed_h3}."
                ),
                metrics={"median_cka": med_cka, "median_cosine": med_cos},
            )
        else:
            results["H3_deep_geometric"] = HypothesisResult(
                passed=None,
                description=SHAMAN_HYPOTHESES["H3_deep_geometric"],
                evidence="CKA or cosine values missing for deep layers.",
                metrics={},
            )
    else:
        results["H3_deep_geometric"] = HypothesisResult(
            passed=None,
            description=SHAMAN_HYPOTHESES["H3_deep_geometric"],
            evidence="No layers in range 23–35 found.",
            metrics={},
        )

    # H4: Task specificity via control prompt
    if control_sim is not None:
        all_rows = list(control_sim.values())
        mean_cos_ctrl = _mean_metric(all_rows, "cosine")
        mean_cka_ctrl = _mean_metric(all_rows, "cka")
        passed_h4 = mean_cos_ctrl >= 0.9998 and mean_cka_ctrl >= 0.9998
        results["H4_task_specificity"] = HypothesisResult(
            passed=passed_h4,
            description=SHAMAN_HYPOTHESES["H4_task_specificity"],
            evidence=(
                f"Control prompt all-layer mean cosine={mean_cos_ctrl:.6f}, "
                f"CKA={mean_cka_ctrl:.6f}. Threshold: both ≥ 0.9998."
            ),
            metrics={"mean_cosine": mean_cos_ctrl, "mean_cka": mean_cka_ctrl},
        )
    else:
        results["H4_task_specificity"] = HypothesisResult(
            passed=None,
            description=SHAMAN_HYPOTHESES["H4_task_specificity"],
            evidence="No control prompt provided — H4 not applicable.",
            metrics={},
        )

    # H5: Sublayer attn vs MLP divergence
    mid_sublayer: list[dict[str, Any]] = []
    for key, vals in sublayer_sim.items():
        try:
            idx = int(key.split("_")[-1]) if "_" in key else int(key)
        except ValueError:
            continue
        if 10 <= idx <= 22:
            mid_sublayer.append(vals)

    if mid_sublayer:
        attn_vals: list[float] = [
            float(v["attn_cos"]) for v in mid_sublayer if v.get("attn_cos") is not None
        ]
        mlp_vals: list[float] = [
            float(v["mlp_cos"]) for v in mid_sublayer if v.get("mlp_cos") is not None
        ]

        if attn_vals and mlp_vals:
            mean_attn = sum(attn_vals) / len(attn_vals)
            mean_mlp = sum(mlp_vals) / len(mlp_vals)
            # Match activ4.py: H5 always passes when sublayer data is present
            # (quantifies divergence, doesn't fail on a threshold)
            attn_more_different = mean_attn < mean_mlp
            results["H5_sublayer_attn_vs_mlp"] = HypothesisResult(
                passed=True,
                description=SHAMAN_HYPOTHESES["H5_sublayer_attn_vs_mlp"],
                evidence=(
                    f"Mid layer mean attn_cos={mean_attn:.6f}, mlp_cos={mean_mlp:.6f}. "
                    f"{'Attention' if attn_more_different else 'MLP'} sublayer changed more."
                ),
                metrics={
                    "mean_attn_cos": mean_attn,
                    "mean_mlp_cos": mean_mlp,
                    "attn_more_different": float(attn_more_different),
                },
            )
        else:
            results["H5_sublayer_attn_vs_mlp"] = HypothesisResult(
                passed=None,
                description=SHAMAN_HYPOTHESES["H5_sublayer_attn_vs_mlp"],
                evidence="Sublayer activations missing — capture_sublayers=False?",
                metrics={},
            )
    else:
        results["H5_sublayer_attn_vs_mlp"] = HypothesisResult(
            passed=None,
            description=SHAMAN_HYPOTHESES["H5_sublayer_attn_vs_mlp"],
            evidence="Not tested — pass --capture_sublayers to validate.",
            metrics={},
        )

    return results


def format_shaman_report(
    hyp_results: dict[str, HypothesisResult],
    rsa_scores: dict[str, float],
    neuron_metrics: dict[str, dict[str, float]],
    sublayer_sim: dict[str, Any],
    name_a: str,
    name_b: str,
) -> str:
    """Format a human-readable SHAMAN analysis report."""
    lines: list[str] = [
        "=" * 70,
        "SHAMAN Representational Analysis Report",
        f"Model A: {name_a}",
        f"Model B: {name_b}",
        "=" * 70,
        "",
    ]

    for key, hr in hyp_results.items():
        status = "PASS" if hr.passed else ("N/A" if hr.passed is None else "FAIL")
        lines.append(f"[{status}] {key}")
        lines.append(f"  {hr.description}")
        lines.append(f"  Evidence: {hr.evidence}")
        if hr.metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in hr.metrics.items())
            lines.append(f"  Metrics: {metric_str}")
        lines.append("")

    # RSA summary
    if rsa_scores:
        valid = {k: v for k, v in rsa_scores.items() if v == v}  # filter nan
        if valid:
            mean_rsa = sum(valid.values()) / len(valid)
            lines.append(f"RSA Spearman (mean across layers): {mean_rsa:.4f}")

    # Neuron summary
    if neuron_metrics:
        all_jac = [v["jaccard"] for v in neuron_metrics.values() if "jaccard" in v]
        if all_jac:
            mean_jac = sum(all_jac) / len(all_jac)
            lines.append(f"Mean neuron Jaccard similarity: {mean_jac:.4f}")

    lines.append("=" * 70)
    return "\n".join(lines)
