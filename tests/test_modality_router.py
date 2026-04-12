"""Tests for TextVisualRouter and ModalityRouterConfig (task 10.3)."""

from __future__ import annotations

import pytest

from mlx_grpo.model.modality_router import ModalityRouterConfig, TextVisualRouter

# ── ModalityRouterConfig ──────────────────────────────────────────────


def test_default_config():
    cfg = ModalityRouterConfig()
    assert cfg.image_token_id == -1
    assert cfg.num_experts == 2
    assert cfg.text_expert_idx == 0
    assert cfg.vision_expert_idx == 1
    assert cfg.strategy == "hard"


def test_config_rejects_num_experts_lt_2():
    with pytest.raises(ValueError, match="num_experts"):
        ModalityRouterConfig(num_experts=1)


def test_config_rejects_same_expert_indices():
    with pytest.raises(ValueError, match="must differ"):
        ModalityRouterConfig(text_expert_idx=0, vision_expert_idx=0)


def test_config_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        ModalityRouterConfig(strategy="random")


def test_config_is_frozen():
    cfg = ModalityRouterConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.image_token_id = 99  # type: ignore[misc]


def test_all_image_ids_single():
    cfg = ModalityRouterConfig(image_token_id=32000)
    assert 32000 in cfg.all_image_ids()


def test_all_image_ids_with_extras():
    cfg = ModalityRouterConfig(
        image_token_id=32000,
        image_token_ids=(32001, 32002),
    )
    ids = cfg.all_image_ids()
    assert ids == {32000, 32001, 32002}


def test_all_image_ids_negative_one_excluded():
    """image_token_id=-1 (default) should NOT be in all_image_ids."""
    cfg = ModalityRouterConfig()
    assert -1 not in cfg.all_image_ids()


# ── TextVisualRouter — route() ────────────────────────────────────────


IMAGE_ID = 32000
BASE_CFG = ModalityRouterConfig(image_token_id=IMAGE_ID, num_experts=2)


def test_route_text_tokens():
    router = TextVisualRouter(BASE_CFG)
    tokens = [1, 2, 3, 4, 5]
    result = router.route(tokens)
    assert result == [0, 0, 0, 0, 0]  # all text → expert 0


def test_route_image_tokens():
    router = TextVisualRouter(BASE_CFG)
    tokens = [IMAGE_ID, IMAGE_ID]
    result = router.route(tokens)
    assert result == [1, 1]  # all image → expert 1


def test_route_mixed_tokens():
    router = TextVisualRouter(BASE_CFG)
    tokens = [1, IMAGE_ID, 2, IMAGE_ID, 3]
    result = router.route(tokens)
    assert result == [0, 1, 0, 1, 0]


def test_route_empty():
    router = TextVisualRouter(BASE_CFG)
    assert router.route([]) == []


def test_route_custom_expert_indices():
    cfg = ModalityRouterConfig(
        image_token_id=IMAGE_ID,
        num_experts=3,
        text_expert_idx=2,
        vision_expert_idx=1,
    )
    router = TextVisualRouter(cfg)
    tokens = [1, IMAGE_ID, 2]
    assert router.route(tokens) == [2, 1, 2]


def test_route_with_additional_image_ids():
    cfg = ModalityRouterConfig(
        image_token_id=32000,
        image_token_ids=(32001,),  # BOI marker
    )
    router = TextVisualRouter(cfg)
    tokens = [1, 32000, 32001, 2]
    result = router.route(tokens)
    assert result == [0, 1, 1, 0]


# ── TextVisualRouter — soft_weights() ────────────────────────────────


def test_hard_weights_one_hot():
    router = TextVisualRouter(BASE_CFG)
    weights = router.soft_weights([1, IMAGE_ID])
    assert weights[0] == [1.0, 0.0]  # text → [1, 0]
    assert weights[1] == [0.0, 1.0]  # image → [0, 1]


def test_soft_weights_uniform():
    cfg = ModalityRouterConfig(
        image_token_id=IMAGE_ID,
        num_experts=4,
        strategy="soft",
        vision_expert_idx=3,
        text_expert_idx=0,
    )
    router = TextVisualRouter(cfg)
    weights = router.soft_weights([1, IMAGE_ID])
    for w in weights:
        assert len(w) == 4
        assert abs(sum(w) - 1.0) < 1e-9
        assert all(v == pytest.approx(0.25) for v in w)


def test_soft_weights_empty():
    router = TextVisualRouter(BASE_CFG)
    assert router.soft_weights([]) == []


# ── TextVisualRouter — image_mask() ──────────────────────────────────


def test_image_mask():
    router = TextVisualRouter(BASE_CFG)
    tokens = [1, IMAGE_ID, 2, IMAGE_ID]
    mask = router.image_mask(tokens)
    assert mask == [False, True, False, True]


def test_image_mask_empty():
    router = TextVisualRouter(BASE_CFG)
    assert router.image_mask([]) == []


# ── as_mlx_arrays() — graceful fallback ──────────────────────────────


def test_as_mlx_arrays_returns_none_without_mlx(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mlx.core", None)
    router = TextVisualRouter(BASE_CFG)
    ei, w = router.as_mlx_arrays([1, IMAGE_ID])
    assert ei is None
    assert w is None
