"""Tests for dynamic_scaling_logger.multipliers_to_csv_row."""

from __future__ import annotations

from mlx_grpo.training.dynamic_scaling_logger import multipliers_to_csv_row


class TestMultipliersToCSVRow:
    """Test CSV row generation for stats.csv."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.005)

        assert isinstance(result, dict)

    def test_has_per_layer_columns(self):
        """Test that dict has per-layer columns."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.005)

        # Should have L20_mult through L35_mult
        for i in range(20, 36):
            key = f"L{i}_mult"
            assert key in result

    def test_has_aggregate_columns(self):
        """Test that dict has aggregate columns."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 3.5, 0.042)

        assert "max_multiplier_used" in result
        assert "mean_angle_drift" in result

    def test_total_columns_18(self):
        """Test that total columns equals 18 (16 layers + 2 aggregates)."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.005)

        assert len(result) == 18

    def test_per_layer_values_strings(self):
        """Test that per-layer values are formatted as strings."""
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

        result = multipliers_to_csv_row(multipliers, layer_names, 3.6, 0.05)

        # First value should match first multiplier
        assert result["L20_mult"] == "2.1000"
        assert result["L21_mult"] == "0.5000"
        assert result["L35_mult"] == "3.6000"

    def test_aggregate_values_strings(self):
        """Test that aggregate values are formatted as strings."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 3.5, 0.0425)

        assert result["max_multiplier_used"] == "3.5000"
        assert result["mean_angle_drift"] == "0.0425"

    def test_format_precision_four_decimals(self):
        """Test that values are formatted with 4 decimal places."""
        multipliers = [2.123456] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 2.123456, 0.123456)

        # Should be 4 decimals: .1235 (rounded)
        assert "2.1235" in result["L20_mult"]
        assert "2.1235" in result["max_multiplier_used"]
        assert "0.1235" in result["mean_angle_drift"]

    def test_with_min_multiplier(self):
        """Test with min_multiplier values."""
        multipliers = [0.1] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 0.1, 0.001)

        assert result["L20_mult"] == "0.1000"
        assert result["max_multiplier_used"] == "0.1000"

    def test_with_max_multiplier(self):
        """Test with max_multiplier values."""
        multipliers = [5.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 5.0, 0.1)

        assert result["L20_mult"] == "5.0000"
        assert result["max_multiplier_used"] == "5.0000"

    def test_with_varied_values(self):
        """Test with varied multiplier values."""
        multipliers = [float(i) for i in range(1, 17)]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 16.0, 0.05)

        assert result["L20_mult"] == "1.0000"
        assert result["L21_mult"] == "2.0000"
        assert result["L35_mult"] == "16.0000"
        assert result["max_multiplier_used"] == "16.0000"

    def test_drift_precision_small(self):
        """Test drift value with many decimals."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.00123456)

        # Should be formatted with 4 decimals
        assert result["mean_angle_drift"] == "0.0012"

    def test_drift_zero(self):
        """Test drift value of 0."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.0)

        assert result["mean_angle_drift"] == "0.0000"

    def test_drift_large_value(self):
        """Test large drift value."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 10.5)

        assert result["mean_angle_drift"] == "10.5000"

    def test_max_mult_actual_max(self):
        """Test that max_mult parameter is used directly."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        # Pass different max_mult than actual max(multipliers)
        result = multipliers_to_csv_row(multipliers, layer_names, 3.7, 0.05)

        assert result["max_multiplier_used"] == "3.7000"

    def test_all_layer_columns_present(self):
        """Test that all 16 layer columns are present."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.005)

        expected_keys = [f"L{i}_mult" for i in range(20, 36)]
        for key in expected_keys:
            assert key in result

    def test_column_order_consistent(self):
        """Test that columns are in consistent order."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result1 = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.005)
        result2 = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.005)

        assert list(result1.keys()) == list(result2.keys())

    def test_with_real_training_values(self):
        """Test with realistic training values."""
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

        result = multipliers_to_csv_row(multipliers, layer_names, 5.0, 0.0547)

        assert result["L20_mult"] == "0.1000"
        assert result["L31_mult"] == "5.0000"
        assert result["max_multiplier_used"] == "5.0000"
        assert result["mean_angle_drift"] == "0.0547"

    def test_negative_values_in_multipliers(self):
        """Test handling of edge case with negative values."""
        multipliers = [-1.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = multipliers_to_csv_row(multipliers, layer_names, 1.0, 0.005)

        # Should format negative correctly
        assert result["L20_mult"] == "-1.0000"
