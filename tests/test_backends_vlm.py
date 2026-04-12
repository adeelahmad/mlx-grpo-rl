"""Tests for VLM image support in backend protocol (Phase 1, Task 2.3)."""

from __future__ import annotations

from mlx_grpo.backends.base import GenerateParams


def _make_params(**kwargs) -> GenerateParams:
    """Convenience factory for GenerateParams with test defaults."""
    return GenerateParams(max_tokens=16, **kwargs)


def test_mock_backend_images_none_default(mock_backend):
    """Calling generate() without images leaves last_images as None."""
    params = _make_params()
    assert params.images is None
    mock_backend.generate([72, 101, 108, 108, 111], params)
    assert mock_backend.last_images is None


def test_mock_backend_images_passed(mock_backend):
    """Calling generate() with images=[...] sets backend.last_images."""
    params = _make_params(images=["https://example.com/cat.jpg"])
    mock_backend.generate([72, 101], params)
    assert mock_backend.last_images == ["https://example.com/cat.jpg"]


def test_generate_params_images_field_default():
    """GenerateParams.images defaults to None (backward compatible)."""
    params = GenerateParams()
    assert params.images is None


def test_backward_compat_positional_args(mock_backend_factory):
    """MockBackend.generate() with no images arg still returns valid output."""
    backend = mock_backend_factory(**{"call_1": "hello world"})
    params = GenerateParams(max_tokens=64, stop=[])
    output = backend.generate([72], params)
    assert output.text == "hello world"
    assert backend.last_images is None
