"""Tests for mlx_grpo.inspect.hypotheses."""

from __future__ import annotations

import pytest

from mlx_grpo.inspect.hypotheses import (
    SHAMAN_HYPOTHESES,
    HypothesisResult,
    evaluate_hypotheses,
    format_shaman_report,
)


def _make_similarity(
    n_layers: int,
    cos_val: float = 0.9999,
    cka_val: float = 0.9999,
) -> dict[str, dict[str, float]]:
    return {f"layer_{i}": {"cosine": cos_val, "cka": cka_val} for i in range(n_layers)}


def _make_neuron_metrics(
    n_layers: int,
    jaccard: float = 0.85,
    sign_consistency: float = 0.9,
    l2_diff_pct: float = 3.0,  # percentage (0-100 scale)
    sparsity_diff_pct: float = 15.0,  # percentage (7-30% satisfies H2)
) -> dict[str, dict[str, float]]:
    return {
        f"layer_{i}": {
            "jaccard": jaccard,
            "sign_consistency": sign_consistency,
            "l2_diff_pct": l2_diff_pct,
            "sparsity_diff_pct": sparsity_diff_pct,
        }
        for i in range(n_layers)
    }


def _make_sublayer_sim(
    n_layers: int,
    attn_cos: float | None = 0.99,
    mlp_cos: float | None = 0.95,
) -> dict[str, dict[str, float | None]]:
    return {f"layer_{i}": {"attn_cos": attn_cos, "mlp_cos": mlp_cos} for i in range(n_layers)}


class TestShamanHypotheses:
    def test_hypothesis_keys_exist(self) -> None:
        assert "H1_early_preservation" in SHAMAN_HYPOTHESES
        assert "H2_mid_rewiring" in SHAMAN_HYPOTHESES
        assert "H3_deep_geometric" in SHAMAN_HYPOTHESES
        assert "H4_task_specificity" in SHAMAN_HYPOTHESES
        assert "H5_sublayer_attn_vs_mlp" in SHAMAN_HYPOTHESES


class TestH1EarlyPreservation:
    def test_passes_with_high_similarity(self) -> None:
        sim = _make_similarity(36, cos_val=0.99995, cka_val=0.99995)
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H1_early_preservation"].passed is True

    def test_fails_with_low_cosine(self) -> None:
        sim = _make_similarity(36, cos_val=0.99, cka_val=0.9999)
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H1_early_preservation"].passed is False

    def test_fails_with_low_cka(self) -> None:
        sim = _make_similarity(36, cos_val=0.99995, cka_val=0.99)
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H1_early_preservation"].passed is False

    def test_not_applicable_when_no_early_layers(self) -> None:
        sim = {f"layer_{i}": {"cosine": 0.9999, "cka": 0.9999} for i in range(25, 36)}
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H1_early_preservation"].passed is None


class TestH2MidRewiring:
    def test_passes_with_expected_values(self) -> None:
        # sparsity_diff=15% (in 7-30%), l2_diff=3% (<5%) → passes
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36, sparsity_diff_pct=15.0, l2_diff_pct=3.0)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H2_mid_rewiring"].passed is True

    def test_fails_low_sparsity_diff(self) -> None:
        # sparsity_diff=1% below the 7% floor
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36, sparsity_diff_pct=1.0, l2_diff_pct=3.0)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H2_mid_rewiring"].passed is False

    def test_fails_high_l2_diff(self) -> None:
        # sparsity_diff=15% (ok), l2_diff=10% above the 5% ceiling
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36, sparsity_diff_pct=15.0, l2_diff_pct=10.0)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H2_mid_rewiring"].passed is False

    def test_not_applicable_when_no_mid_layers(self) -> None:
        # Only layers 0-9 — no mid-range layers exist
        sim = {f"layer_{i}": {"cosine": 0.9999, "cka": 0.9999} for i in range(10)}
        nm = {
            f"layer_{i}": {"jaccard": 0.85, "l2_diff_pct": 3.0, "sparsity_diff_pct": 15.0}
            for i in range(10)
        }
        results = evaluate_hypotheses(sim, nm, {}, None, 10)
        assert results["H2_mid_rewiring"].passed is None


class TestH3DeepGeometric:
    def test_passes_when_cka_lt_cosine(self) -> None:
        # CKA < cosine in deep layers
        sim = {f"layer_{i}": {"cosine": 0.99, "cka": 0.90} for i in range(36)}
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H3_deep_geometric"].passed is True

    def test_fails_when_cka_ge_cosine(self) -> None:
        # CKA >= cosine in deep layers
        sim = {f"layer_{i}": {"cosine": 0.90, "cka": 0.99} for i in range(36)}
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H3_deep_geometric"].passed is False

    def test_not_applicable_when_no_deep_layers(self) -> None:
        sim = {f"layer_{i}": {"cosine": 0.99, "cka": 0.90} for i in range(10)}
        nm = _make_neuron_metrics(10)
        results = evaluate_hypotheses(sim, nm, {}, None, 10)
        assert results["H3_deep_geometric"].passed is None


class TestH4TaskSpecificity:
    def test_passes_with_high_control_similarity(self) -> None:
        sim = _make_similarity(36, cos_val=0.9999, cka_val=0.9999)
        nm = _make_neuron_metrics(36)
        control_sim = _make_similarity(36, cos_val=0.99990, cka_val=0.99990)
        results = evaluate_hypotheses(sim, nm, {}, control_sim, 36)
        assert results["H4_task_specificity"].passed is True

    def test_fails_with_low_control_similarity(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        control_sim = _make_similarity(36, cos_val=0.95, cka_val=0.95)
        results = evaluate_hypotheses(sim, nm, {}, control_sim, 36)
        assert results["H4_task_specificity"].passed is False

    def test_not_applicable_when_no_control(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H4_task_specificity"].passed is None


class TestH5SublayerAttnVsMLP:
    def test_passes_with_divergence(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        sublayer = _make_sublayer_sim(36, attn_cos=0.99, mlp_cos=0.80)
        results = evaluate_hypotheses(sim, nm, sublayer, None, 36)
        assert results["H5_sublayer_attn_vs_mlp"].passed is True

    def test_passes_with_same_values_but_present(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        sublayer = _make_sublayer_sim(36, attn_cos=0.95, mlp_cos=0.95)
        results = evaluate_hypotheses(sim, nm, sublayer, None, 36)
        # H5 always passes when sublayer data is present (quantifies divergence, never fails)
        assert results["H5_sublayer_attn_vs_mlp"].passed is True

    def test_not_applicable_when_no_sublayers(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        # Empty sublayer sim
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        assert results["H5_sublayer_attn_vs_mlp"].passed is None

    def test_not_applicable_when_none_values(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        sublayer = _make_sublayer_sim(36, attn_cos=None, mlp_cos=None)
        results = evaluate_hypotheses(sim, nm, sublayer, None, 36)
        assert results["H5_sublayer_attn_vs_mlp"].passed is None


class TestHypothesisResult:
    def test_frozen(self) -> None:
        hr = HypothesisResult(passed=True, description="test", evidence="ev")
        with pytest.raises((AttributeError, TypeError)):
            hr.passed = False  # type: ignore[misc]

    def test_default_metrics(self) -> None:
        hr = HypothesisResult(passed=None, description="test", evidence="ev")
        assert hr.metrics == {}


class TestFormatShamanReport:
    def test_format_basic(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        rsa = {"layer_0": 0.85, "layer_1": 0.90}
        report = format_shaman_report(results, rsa, nm, {}, "model_a", "model_b")
        assert "SHAMAN Representational Analysis Report" in report
        assert "model_a" in report
        assert "model_b" in report
        assert "H1_early_preservation" in report
        assert "H4_task_specificity" in report

    def test_format_includes_pass_fail(self) -> None:
        sim = _make_similarity(36, cos_val=0.9999, cka_val=0.9999)
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        report = format_shaman_report(results, {}, nm, {}, "a", "b")
        assert "PASS" in report or "FAIL" in report or "N/A" in report

    def test_format_nan_rsa_filtered(self) -> None:
        sim = _make_similarity(36)
        nm = _make_neuron_metrics(36)
        results = evaluate_hypotheses(sim, nm, {}, None, 36)
        # All nan RSA scores should not cause errors
        rsa = {"layer_0": float("nan"), "layer_1": float("nan")}
        report = format_shaman_report(results, rsa, nm, {}, "a", "b")
        assert "SHAMAN" in report
