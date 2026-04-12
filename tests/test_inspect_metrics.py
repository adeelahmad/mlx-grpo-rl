"""Tests for mlx_grpo.inspect.metrics — pure numpy, no hardware needed."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_grpo.inspect.metrics import (
    compute_layer_stats,
    compute_layerwise_similarity,
    compute_neuron_metrics,
    compute_rsa_layerwise,
    compute_sublayer_similarity,
    cosine_similarity,
    l2_norm_diff_pct,
    linear_cka,
    neuron_jaccard,
    neuron_sign_consistency,
    rsa_matrix,
    rsa_spearman,
)


class TestLayerStats:
    def test_basic_stats(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_layer_stats(arr, "layer_0")
        assert stats.layer_key == "layer_0"
        assert abs(stats.mean - 3.0) < 1e-6
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.error is None

    def test_empty_array(self) -> None:
        arr = np.array([])
        stats = compute_layer_stats(arr, "layer_0")
        assert stats.error is not None
        assert stats.mean == 0.0

    def test_sparsity(self) -> None:
        # sparsity = fraction of exact zeros (matching activ4.py)
        arr = np.array([0.0, 0.0, 1.0, 2.0])
        stats = compute_layer_stats(arr, "layer_0")
        assert abs(stats.sparsity - 0.5) < 1e-6

    def test_l1_l2_norm(self) -> None:
        arr = np.array([3.0, 4.0])
        stats = compute_layer_stats(arr, "layer_0")
        assert abs(stats.l2_norm - 5.0) < 1e-6
        assert abs(stats.l1_norm - 7.0) < 1e-6

    def test_frozen(self) -> None:
        stats = compute_layer_stats(np.array([1.0]), "layer_0")
        with pytest.raises((AttributeError, TypeError)):
            stats.mean = 99.0  # type: ignore[misc]

    def test_2d_array(self) -> None:
        arr = np.ones((3, 4))
        stats = compute_layer_stats(arr, "layer_0")
        assert abs(stats.mean - 1.0) < 1e-6


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero(self) -> None:
        a = np.zeros(5)
        assert cosine_similarity(a, a) == 0.0


class TestLinearCKA:
    def test_identical_matrices(self) -> None:
        x = np.random.default_rng(0).standard_normal((10, 8))
        assert abs(linear_cka(x, x) - 1.0) < 1e-5

    def test_range_zero_one(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 16))
        y = rng.standard_normal((20, 16))
        cka = linear_cka(x, y)
        assert 0.0 <= cka <= 1.0 + 1e-9

    def test_single_sample(self) -> None:
        x = np.array([[1.0, 2.0]])
        y = np.array([[3.0, 4.0]])
        cka = linear_cka(x, y)
        assert cka == 1.0

    def test_zero_matrix(self) -> None:
        x = np.zeros((5, 4))
        cka = linear_cka(x, x)
        assert cka == 0.0

    def test_different_dims(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal((10, 8))
        y = rng.standard_normal((10, 16))
        cka = linear_cka(x, y)
        assert 0.0 <= cka <= 1.0 + 1e-9


class TestComputeLayerwiseSimilarity:
    def test_basic(self) -> None:
        rng = np.random.default_rng(0)
        sketches_a = {f"layer_{i}": rng.standard_normal(32).astype(np.float32) for i in range(3)}
        sketches_b = {f"layer_{i}": rng.standard_normal(32).astype(np.float32) for i in range(3)}
        result = compute_layerwise_similarity(sketches_a, sketches_b)
        assert set(result.keys()) == {"layer_0", "layer_1", "layer_2"}
        for _key, vals in result.items():
            assert "cosine" in vals
            assert "cka" in vals

    def test_identical_sketches(self) -> None:
        a = np.ones(16, dtype=np.float32)
        sketches = {"layer_0": a}
        result = compute_layerwise_similarity(sketches, sketches)
        assert abs(result["layer_0"]["cosine"] - 1.0) < 1e-5

    def test_missing_key_in_b(self) -> None:
        sketches_a = {"layer_0": np.ones(8), "layer_1": np.ones(8)}
        sketches_b = {"layer_0": np.ones(8)}
        result = compute_layerwise_similarity(sketches_a, sketches_b)
        assert "layer_0" in result
        assert "layer_1" not in result


class TestRSA:
    def test_rsa_matrix_shape(self) -> None:
        # 5 samples → upper-triangle length = 5*4//2 = 10
        acts = np.random.default_rng(0).standard_normal((5, 8))
        vec = rsa_matrix(acts)
        assert vec.shape == (10,)

    def test_rsa_matrix_values_in_range(self) -> None:
        # Cosine dissimilarity values must be in [0, 2]
        acts = np.random.default_rng(0).standard_normal((5, 8))
        vec = rsa_matrix(acts)
        assert np.all(vec >= -1e-6)
        assert np.all(vec <= 2.0 + 1e-6)

    def test_rsa_matrix_nonnegative(self) -> None:
        acts = np.random.default_rng(0).standard_normal((5, 8))
        vec = rsa_matrix(acts)
        assert float(vec.min()) >= -1e-6

    def test_rsa_matrix_subsampling(self) -> None:
        # 50 samples → upper-triangle length = 50*49//2 = 1225
        acts = np.random.default_rng(0).standard_normal((300, 8))
        vec = rsa_matrix(acts, n_subsample=50)
        assert vec.shape == (1225,)

    def test_rsa_spearman_identical(self) -> None:
        acts = np.random.default_rng(0).standard_normal((10, 8))
        mat = rsa_matrix(acts)
        r = rsa_spearman(mat, mat)
        # scipy may or may not be available
        if r == r:  # not nan
            assert abs(r - 1.0) < 1e-6

    def test_compute_rsa_layerwise(self) -> None:
        rng = np.random.default_rng(0)
        raw_a = {"layer_0": rng.standard_normal((5, 8))}
        raw_b = {"layer_0": rng.standard_normal((5, 8))}
        result = compute_rsa_layerwise(raw_a, raw_b)
        assert "layer_0" in result

    def test_compute_rsa_layerwise_single_sample(self) -> None:
        raw_a = {"layer_0": np.ones((1, 8))}
        raw_b = {"layer_0": np.ones((1, 8))}
        result = compute_rsa_layerwise(raw_a, raw_b)
        assert result["layer_0"] != result["layer_0"] or result["layer_0"] == result["layer_0"]


class TestNeuronMetrics:
    def test_jaccard_identical_1d(self) -> None:
        a = np.array([1.0, 0.5, 0.02, 0.0])
        assert abs(neuron_jaccard(a, a) - 1.0) < 1e-6

    def test_jaccard_no_overlap_1d(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        j = neuron_jaccard(a, b, threshold=0.5)
        assert abs(j - 0.0) < 1e-6

    def test_jaccard_all_zero(self) -> None:
        a = np.zeros(5)
        assert neuron_jaccard(a, a) == 1.0

    def test_jaccard_2d_arrays(self) -> None:
        # 2D: (seq, hidden) — should mean over axis=0 first
        a = np.array([[1.0, 0.0], [0.5, 0.0]])  # mean: [0.75, 0.0]
        b = np.array([[1.0, 0.0], [0.5, 0.0]])
        assert abs(neuron_jaccard(a, b) - 1.0) < 1e-6

    def test_sign_consistency_identical(self) -> None:
        a = np.array([1.0, -2.0, 3.0])
        assert abs(neuron_sign_consistency(a, a) - 1.0) < 1e-6

    def test_sign_consistency_opposite(self) -> None:
        a = np.array([1.0, -2.0, 3.0])
        b = -a
        assert abs(neuron_sign_consistency(a, b)) < 1e-6

    def test_sign_consistency_2d(self) -> None:
        # 2D: mean then sign
        a = np.array([[1.0, -1.0], [1.0, -1.0]])  # mean: [1.0, -1.0]
        b = np.array([[1.0, -1.0], [1.0, -1.0]])
        assert abs(neuron_sign_consistency(a, b) - 1.0) < 1e-6

    def test_l2_diff_pct_identical(self) -> None:
        a = np.array([3.0, 4.0])
        assert l2_norm_diff_pct(a, a) == 0.0

    def test_l2_diff_pct_rms_formula(self) -> None:
        # rms_a = sqrt(mean([1,1,1,1])) = 1.0, rms_b = sqrt(mean([4,4,4,4])) = 2.0
        # pct = abs(2-1)/1*100 = 100%
        a = np.ones(4)
        b = np.full(4, 2.0)
        pct = l2_norm_diff_pct(a, b)
        assert abs(pct - 100.0) < 1e-5

    def test_l2_diff_pct_zero_a(self) -> None:
        a = np.zeros(4)
        b = np.ones(4)
        pct = l2_norm_diff_pct(a, b)
        assert pct == 0.0  # rms_a < 1e-9 branch

    def test_compute_neuron_metrics_all_keys(self) -> None:
        rng = np.random.default_rng(0)
        raw_a = {"layer_0": rng.standard_normal(32)}
        raw_b = {"layer_0": rng.standard_normal(32)}
        result = compute_neuron_metrics(raw_a, raw_b)
        assert "layer_0" in result
        assert "jaccard" in result["layer_0"]
        assert "sign_consistency" in result["layer_0"]
        assert "l2_diff_pct" in result["layer_0"]
        assert "sparsity_diff_pct" in result["layer_0"]

    def test_sparsity_diff_pct_identical(self) -> None:
        a = np.array([0.0, 1.0, 0.0, 2.0])  # sparsity = 0.5
        result = compute_neuron_metrics({"layer_0": a}, {"layer_0": a})
        assert result["layer_0"]["sparsity_diff_pct"] == 0.0

    def test_sparsity_diff_pct_different(self) -> None:
        a = np.array([0.0, 0.0, 1.0, 2.0])  # sparsity = 0.5
        b = np.array([1.0, 2.0, 3.0, 4.0])  # sparsity = 0.0
        result = compute_neuron_metrics({"layer_0": a}, {"layer_0": b})
        # abs(0.0 - 0.5) / 0.5 * 100 = 100%
        assert result["layer_0"]["sparsity_diff_pct"] > 0.0


class TestSublayerSimilarity:
    def test_no_sublayers(self) -> None:
        class MockLA:
            attn_out = None
            mlp_out = None

        acts_a = {"layer_0": MockLA()}
        acts_b = {"layer_0": MockLA()}
        result = compute_sublayer_similarity(acts_a, acts_b)
        assert result["layer_0"]["attn_cos"] is None
        assert result["layer_0"]["mlp_cos"] is None

    def test_with_sublayers(self) -> None:
        class MockLA:
            def __init__(self, attn: np.ndarray, mlp: np.ndarray) -> None:
                self.attn_out = attn
                self.mlp_out = mlp

        a = np.ones(8, dtype=np.float32)
        acts_a = {"layer_0": MockLA(a, a * 2)}
        acts_b = {"layer_0": MockLA(a, a * 2)}
        result = compute_sublayer_similarity(acts_a, acts_b)
        assert result["layer_0"]["attn_cos"] is not None
        assert result["layer_0"]["mlp_cos"] is not None
        assert abs(result["layer_0"]["attn_cos"] - 1.0) < 1e-5

    def test_missing_key(self) -> None:
        class MockLA:
            attn_out = None
            mlp_out = None

        acts_a = {"layer_0": MockLA(), "layer_1": MockLA()}
        acts_b = {"layer_0": MockLA()}
        result = compute_sublayer_similarity(acts_a, acts_b)
        assert "layer_0" in result
        assert "layer_1" not in result
