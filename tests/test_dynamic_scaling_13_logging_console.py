"""Tests for dynamic_scaling_logger.format_multipliers_for_console."""

from __future__ import annotations

from mlx_grpo.training.dynamic_scaling_logger import format_multipliers_for_console


class TestFormatMultipliersForConsole:
    """Test console formatting of top-3 multipliers."""

    def test_basic_format(self):
        """Test basic formatting."""
        multipliers = [2.1, 0.5, 3.5, 1.0] * 4
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        assert "[DYNAMIC_SCALING]" in result
        assert "Top 3 Layers:" in result
        # Should have 3 layer:value pairs separated by |
        assert result.count("|") == 2

    def test_top_3_ordering(self):
        """Test that top 3 multipliers are returned in descending order."""
        multipliers = [1.0, 5.0, 0.5, 3.0, 2.0] + [1.0] * 11
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Should have L21:5.0 (highest), L23:3.0, L24:2.0
        assert "L21:5.0" in result
        assert "L23:3.0" in result
        assert "L24:2.0" in result

    def test_all_ones_returns_empty(self):
        """Test that all-1.0 multipliers return empty string."""
        multipliers = [1.0] * 16
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        assert result == ""

    def test_nearly_all_ones_returns_empty(self):
        """Test that multipliers very close to 1.0 return empty."""
        multipliers = [1.0 + 1e-7] * 16  # All within 1e-6 of 1.0
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        assert result == ""

    def test_two_high_multipliers(self):
        """Test with only 2 high multipliers."""
        multipliers = [
            4.5,
            3.8,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Should still show top 2 (with a third 1.0 if needed)
        assert "L20:4.5" in result
        assert "L21:3.8" in result

    def test_single_high_multiplier(self):
        """Test with only 1 high multiplier."""
        multipliers = [5.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Should show single value (and zeros/ones)
        assert "L20:5.0" in result

    def test_formatting_precision_one_decimal(self):
        """Test that values are formatted with 1 decimal place."""
        multipliers = [2.123, 3.456, 1.789] + [1.0] * 13
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Should be rounded to 1 decimal
        assert "3.5" in result or "3.4" in result  # Rounding
        assert "2.1" in result
        assert "1.8" in result

    def test_empty_multipliers(self):
        """Test with empty multipliers list."""
        result = format_multipliers_for_console([], [])

        assert result == ""

    def test_empty_layer_names(self):
        """Test with empty layer names list."""
        multipliers = [1.0] * 16
        result = format_multipliers_for_console(multipliers, [])

        assert result == ""

    def test_separator_format(self):
        """Test that separator is ' | ' (space-pipe-space)."""
        multipliers = [5.0, 4.0, 3.0, 1.0] * 4
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Check exact separator
        assert " | " in result
        parts = result.split(" | ")
        assert len(parts) == 3

    def test_layer_value_colon_format(self):
        """Test that each entry is formatted as Layer:Value."""
        multipliers = [5.0, 4.0, 3.0] + [1.0] * 13
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Each part should have format "L##:X.X"
        parts = result.split("[DYNAMIC_SCALING] Top 3 Layers: ")[1].split(" | ")
        for part in parts:
            assert ":" in part
            layer, value = part.split(":")
            assert layer.startswith("L")
            float(value)  # Should be parseable as float

    def test_maximum_multiplier_selection(self):
        """Test that maximum value is definitely in top 3."""
        multipliers = [2.0, 5.5, 1.5, 3.2, 4.8] + [1.0] * 11
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        assert "5.5" in result

    def test_with_min_multiplier_value(self):
        """Test with minimum multiplier value (0.1)."""
        multipliers = [0.1, 5.0, 1.0, 2.0, 3.0] + [1.0] * 11
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        assert "L21:5.0" in result
        assert "L23:2.0" in result
        assert "L24:3.0" in result

    def test_with_max_multiplier_value(self):
        """Test with maximum multiplier value (5.0)."""
        multipliers = [5.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        assert "L20:5.0" in result

    def test_mixed_high_low_multipliers(self):
        """Test with mixture of high and low values."""
        multipliers = [
            0.2,
            4.8,
            0.5,
            4.2,
            1.0,
            3.9,
            1.0,
            1.0,
            2.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Top 3: 4.8, 4.2, 3.9
        assert "4.8" in result
        assert "4.2" in result
        assert "3.9" in result

    def test_all_different_values(self):
        """Test with all different values."""
        multipliers = [float(i + 1) for i in range(16)]
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Top 3 should be 16.0, 15.0, 14.0
        assert "16.0" in result
        assert "15.0" in result
        assert "14.0" in result

    def test_duplicate_max_values(self):
        """Test with duplicate maximum values."""
        multipliers = [5.0, 5.0, 5.0, 1.0] * 4
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = format_multipliers_for_console(multipliers, layer_names)

        # Should show 3 copies of 5.0
        count_5 = result.count("5.0")
        assert count_5 >= 3
