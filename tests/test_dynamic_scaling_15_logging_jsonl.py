"""Tests for dynamic_scaling_logger.multipliers_to_jsonl_dict."""

from __future__ import annotations

import pytest

from mlx_grpo.training.dynamic_scaling_logger import multipliers_to_jsonl_dict


class TestMultipliersToJSONLDict:
    """Test JSONL dict generation for rollouts.jsonl."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        assert isinstance(result, dict)

    def test_has_gradient_multipliers_key(self):
        """Test that dict has 'gradient_multipliers' key."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        assert "gradient_multipliers" in result

    def test_has_max_multiplier_key(self):
        """Test that dict has 'max_multiplier_used' key."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        assert "max_multiplier_used" in result

    def test_total_keys_two(self):
        """Test that dict has exactly 2 top-level keys."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        assert len(result) == 2

    def test_gradient_multipliers_is_dict(self):
        """Test that gradient_multipliers is a nested dict."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        assert isinstance(result["gradient_multipliers"], dict)

    def test_gradient_multipliers_has_16_entries(self):
        """Test that gradient_multipliers has 16 layer entries."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        assert len(result["gradient_multipliers"]) == 16

    def test_gradient_multipliers_per_layer_format(self):
        """Test that gradient_multipliers uses layer names as keys."""
        multipliers = [
            2.1,
            0.5,
            3.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            1.2,
            1.8,
            2.8,
            3.2,
            0.8,
            1.6,
            2.4,
            3.6,
        ]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 3.6)

        gm = result["gradient_multipliers"]

        assert gm["L20"] == 2.1
        assert gm["L21"] == 0.5
        assert gm["L22"] == 3.5
        assert gm["L35"] == 3.6

    def test_max_multiplier_used_value(self):
        """Test that max_multiplier_used matches passed value."""
        multipliers = [2.0, 0.5, 1.5, 1.0] * 4
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 3.7)

        assert result["max_multiplier_used"] == 3.7

    def test_max_multiplier_various_values(self):
        """Test max_multiplier with various values."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        for max_val in [0.1, 1.0, 2.5, 5.0, 10.0]:
            result = multipliers_to_jsonl_dict(multipliers, layer_names, max_val)
            assert result["max_multiplier_used"] == max_val

    def test_all_ones(self):
        """Test with all 1.0 multipliers."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        assert all(v == 1.0 for v in result["gradient_multipliers"].values())
        assert result["max_multiplier_used"] == 1.0

    def test_varied_multipliers(self):
        """Test with varied multiplier values."""
        multipliers = [float(i) for i in range(1, 17)]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 16.0)

        gm = result["gradient_multipliers"]

        for i, layer in enumerate(layer_names):
            assert gm[layer] == float(i + 1)

    def test_min_multiplier_value(self):
        """Test with min_multiplier (0.1)."""
        multipliers = [0.1] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 0.1)

        assert all(v == 0.1 for v in result["gradient_multipliers"].values())
        assert result["max_multiplier_used"] == 0.1

    def test_max_multiplier_value(self):
        """Test with max_multiplier (5.0)."""
        multipliers = [5.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 5.0)

        assert all(v == 5.0 for v in result["gradient_multipliers"].values())
        assert result["max_multiplier_used"] == 5.0

    def test_float_precision_preserved(self):
        """Test that float precision is preserved."""
        multipliers = [2.123456789] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 2.123456789)

        gm = result["gradient_multipliers"]
        assert gm["L20"] == pytest.approx(2.123456789)
        assert result["max_multiplier_used"] == pytest.approx(2.123456789)

    def test_layer_names_l20_to_l35(self):
        """Test that layer names are L20 through L35."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.0)

        gm = result["gradient_multipliers"]

        for i in range(20, 36):
            assert f"L{i}" in gm

    def test_json_serializable(self):
        """Test that result is JSON serializable."""
        import json

        multipliers = [
            2.1,
            0.5,
            3.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            1.2,
            1.8,
            2.8,
            3.2,
            0.8,
            1.6,
            2.4,
            3.6,
        ]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 3.6)

        # Should be JSON serializable
        json_str = json.dumps(result)
        restored = json.loads(json_str)

        assert restored == result

    def test_nested_structure_for_rollout_merge(self):
        """Test that structure is suitable for merging into rollout dict."""
        multipliers = [1.5] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 1.5)

        # Simulate merging into a rollout dict
        rollout_dict = {
            "prompt": "test",
            "completion": "test response",
            "reward": 0.8,
        }

        rollout_dict.update(result)

        # Should have all original keys plus new keys
        assert "prompt" in rollout_dict
        assert "completion" in rollout_dict
        assert "reward" in rollout_dict
        assert "gradient_multipliers" in rollout_dict
        assert "max_multiplier_used" in rollout_dict

    def test_empty_multipliers_not_expected(self):
        """Test behavior with edge case of empty lists (not expected in practice)."""
        # Function expects 16 values, but test robustness
        result = multipliers_to_jsonl_dict([], [], 1.0)

        assert isinstance(result, dict)
        assert result["gradient_multipliers"] == {}
        assert result["max_multiplier_used"] == 1.0

    def test_real_training_scenario(self):
        """Test with realistic training multiplier values."""
        multipliers = [
            0.1,
            0.5,
            1.0,
            1.2,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            2.1,
            1.8,
            2.3,
            1.5,
        ]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 5.0)

        gm = result["gradient_multipliers"]

        # Verify specific values
        assert gm["L20"] == 0.1
        assert gm["L31"] == 5.0
        assert gm["L35"] == 1.5
        assert result["max_multiplier_used"] == 5.0

    def test_mismatch_multipliers_names_length(self):
        """Test behavior with mismatched lengths (not expected but edge case)."""
        multipliers = [1.0, 2.0, 3.0]  # Only 3
        layer_names = [f"L{20 + i}" for i in range(16)]  # 16

        result = multipliers_to_jsonl_dict(multipliers, layer_names, 3.0)

        # Should only have 3 entries in gradient_multipliers
        assert len(result["gradient_multipliers"]) == 3
