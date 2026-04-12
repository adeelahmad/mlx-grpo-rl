"""Tests for backend base protocol and utilities."""

from unittest.mock import MagicMock

from mlx_grpo.backends import GenerateParams, generate_batch_serial


class TestGenerateBatchSerial:
    """Test serial fallback for backends without generate_batch."""

    def test_serial_calls_generate_for_each_sequence(self):
        """Should call generate() once per sequence."""
        mock_backend = MagicMock()
        mock_backend.generate = MagicMock(
            side_effect=[
                MagicMock(text="a", tokens=[1, 2]),
                MagicMock(text="b", tokens=[3, 4]),
                MagicMock(text="c", tokens=[5, 6]),
            ]
        )

        prefix_batch = [[1], [2], [3]]
        params = GenerateParams(max_tokens=128)

        results = generate_batch_serial(mock_backend, prefix_batch, params)

        assert len(results) == 3
        assert mock_backend.generate.call_count == 3

    def test_serial_preserves_order(self):
        """Results should be in same order as input."""
        outputs = [MagicMock(text=f"answer{i}", tokens=[10 * i, 10 * i + 1]) for i in range(3)]
        mock_backend = MagicMock()
        mock_backend.generate = MagicMock(side_effect=outputs)

        prefix_batch = [[1], [2], [3]]
        params = GenerateParams()

        results = generate_batch_serial(mock_backend, prefix_batch, params)

        assert results[0].text == "answer0"
        assert results[1].text == "answer1"
        assert results[2].text == "answer2"

    def test_serial_with_same_params(self):
        """Should use same params for all sequences."""
        mock_backend = MagicMock()
        mock_backend.generate = MagicMock(side_effect=[MagicMock(), MagicMock()])

        prefix_batch = [[1, 2], [3, 4]]
        params = GenerateParams(
            max_tokens=256,
            temperature=0.8,
            top_p=0.9,
        )

        generate_batch_serial(mock_backend, prefix_batch, params)

        # Both calls should use same params
        assert mock_backend.generate.call_count == 2
        for call in mock_backend.generate.call_args_list:
            assert call[0][1] is params  # Same params object

    def test_serial_with_empty_batch(self):
        """Should return empty list for empty batch."""
        mock_backend = MagicMock()

        results = generate_batch_serial(mock_backend, [], GenerateParams())

        assert results == []
        mock_backend.generate.assert_not_called()

    def test_serial_with_single_sequence(self):
        """Should work with single-element batch."""
        mock_output = MagicMock(text="answer", tokens=[1, 2, 3])
        mock_backend = MagicMock()
        mock_backend.generate = MagicMock(return_value=mock_output)

        results = generate_batch_serial(
            mock_backend,
            [[1, 2, 3]],
            GenerateParams(),
        )

        assert len(results) == 1
        assert results[0] is mock_output
