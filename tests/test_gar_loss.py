"""Tests for GAR (Guided Activation Reinforcement) loss ops and config.

Uses mock MLX arrays to test pure logic without requiring real GPU hardware.
The actual MLX nn.value_and_grad integration is tested via gar_ops.py being
excluded from coverage (hardware-dependent, like mlx_ops.py).
"""

from __future__ import annotations

import math

import pytest

from mlx_grpo.config.loader import TrainingConfig, _build_dataclass, load_config

# ── Config tests ─────────────────────────────────────────────────────────────


def test_config_mode_field_defaults_to_grpo():
    cfg = TrainingConfig()
    assert cfg.mode == "grpo"


def test_config_mode_can_set_gar():
    cfg = _build_dataclass(TrainingConfig, {"mode": "gar"})
    assert cfg.mode == "gar"


def test_config_gar_drift_threshold_default():
    cfg = TrainingConfig()
    assert cfg.gar_drift_threshold_deg == pytest.approx(5.0)


def test_config_gar_normalize_default():
    cfg = TrainingConfig()
    assert cfg.gar_normalize is True


def test_config_gar_fields_from_yaml(tmp_path):
    yaml_file = tmp_path / "gar.yml"
    yaml_file.write_text(
        "training:\n"
        "  mode: gar\n"
        "  gar_drift_threshold_deg: 3.5\n"
        "  gar_normalize: false\n"
        "  group_size: 4\n"
        "  lora_layers: 16\n"
    )
    cfg = load_config(str(yaml_file))
    assert cfg.training.mode == "gar"
    assert cfg.training.gar_drift_threshold_deg == pytest.approx(3.5)
    assert cfg.training.gar_normalize is False


def test_config_gar_validation_enforces_group_size_2(tmp_path):
    yaml_file = tmp_path / "bad_gs.yml"
    yaml_file.write_text("training:\n  mode: gar\n  group_size: 1\n  lora_layers: 16\n")
    with pytest.raises(ValueError, match="group_size >= 2"):
        load_config(str(yaml_file))


def test_config_gar_validation_enforces_lora_layers_not_none(tmp_path):
    yaml_file = tmp_path / "bad_lora.yml"
    yaml_file.write_text("training:\n  mode: gar\n  group_size: 4\n")
    with pytest.raises(ValueError, match="lora_layers"):
        load_config(str(yaml_file))


def test_config_grpo_mode_no_validation(tmp_path):
    """GRPO mode should not trigger GAR validation even with group_size=1."""
    yaml_file = tmp_path / "grpo.yml"
    yaml_file.write_text("training:\n  mode: grpo\n  group_size: 1\n")
    cfg = load_config(str(yaml_file))
    assert cfg.training.mode == "grpo"
    assert cfg.training.group_size == 1


# ── GAR ops import guard ─────────────────────────────────────────────────────


def test_gar_ops_import():
    """Module should be importable even without MLX (HAS_MLX=False path)."""
    from mlx_grpo.loss import gar_ops

    assert hasattr(gar_ops, "compute_activation_drift_loss")
    assert hasattr(gar_ops, "capture_activations_forward")
    assert hasattr(gar_ops, "gar_gradient_step")
    assert hasattr(gar_ops, "GARGradStepResult")


def test_gar_grad_step_result_defaults():
    from mlx_grpo.loss.gar_ops import GARGradStepResult

    r = GARGradStepResult()
    assert r.loss == 0.0
    assert r.grad_norm == 0.0
    assert r.effective_tokens == 0
    assert r.per_layer_metrics is None


# ── Drift loss tests (with real MLX if available, else skip) ─────────────────

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")


@pytestmark_mlx
def test_compute_activation_drift_loss_basic():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    # Identical activations should give zero loss
    acts = {"0": mx.ones((4, 8))}
    loss, metrics = compute_activation_drift_loss(acts, acts, threshold_deg=5.0)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert "0" in metrics
    assert metrics["0"]["angle_mean"] == pytest.approx(0.0, abs=0.05)


@pytestmark_mlx
def test_compute_activation_drift_loss_threshold_dead_zone():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    # Small perturbation below threshold should give zero loss
    ref = mx.ones((4, 8))
    pol = mx.ones((4, 8)) + mx.array([[0.001] * 8] * 4)
    loss, metrics = compute_activation_drift_loss({"0": ref}, {"0": pol}, threshold_deg=10.0)
    # Angle is tiny (< 10 degrees), should be in dead zone
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-4)


@pytestmark_mlx
def test_compute_activation_drift_loss_above_threshold_contributes():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    # Orthogonal vectors: 90 degrees apart, well above any threshold
    seq_len, hidden = 2, 4
    ref = mx.zeros((seq_len, hidden))
    ref = ref.at[:, 0].add(1.0)  # [1,0,0,0]
    pol = mx.zeros((seq_len, hidden))
    pol = pol.at[:, 1].add(1.0)  # [0,1,0,0]

    loss, metrics = compute_activation_drift_loss(
        {"0": ref}, {"0": pol}, threshold_deg=5.0, normalize=True
    )
    # 90 degrees - 5 degrees threshold = 85 degrees excess in radians
    expected_excess = (90.0 - 5.0) * math.pi / 180.0
    assert float(loss.item()) == pytest.approx(expected_excess, abs=0.05)
    assert metrics["0"]["tokens_above"] == 2.0


@pytestmark_mlx
def test_compute_activation_drift_loss_normalize_enabled():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    # Different magnitudes, same direction: angle should be ~0 with normalize
    ref = mx.ones((2, 4)) * 10.0
    pol = mx.ones((2, 4)) * 0.1
    loss, _ = compute_activation_drift_loss(
        {"0": ref}, {"0": pol}, threshold_deg=1.0, normalize=True
    )
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-4)


@pytestmark_mlx
def test_compute_activation_drift_loss_with_mask():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    # Orthogonal but first token masked
    ref = mx.zeros((2, 4))
    ref = ref.at[:, 0].add(1.0)
    pol = mx.zeros((2, 4))
    pol = pol.at[:, 1].add(1.0)

    masks = [0.0, 1.0]  # first token masked
    loss, metrics = compute_activation_drift_loss(
        {"0": ref}, {"0": pol}, threshold_deg=5.0, masks=masks
    )
    # Only 1 active token contributes
    expected = (90.0 - 5.0) * math.pi / 180.0
    assert float(loss.item()) == pytest.approx(expected, abs=0.05)


@pytestmark_mlx
def test_compute_activation_drift_loss_with_proximity_weights():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    ref = mx.zeros((2, 4))
    ref = ref.at[:, 0].add(1.0)
    pol = mx.zeros((2, 4))
    pol = pol.at[:, 1].add(1.0)

    pw = [2.0, 0.5]
    loss_pw, _ = compute_activation_drift_loss(
        {"0": ref}, {"0": pol}, threshold_deg=5.0, proximity_weights=pw
    )
    loss_no_pw, _ = compute_activation_drift_loss({"0": ref}, {"0": pol}, threshold_deg=5.0)
    # With proximity weights the loss should differ from uniform
    assert float(loss_pw.item()) != pytest.approx(float(loss_no_pw.item()), abs=0.01)


@pytestmark_mlx
def test_compute_activation_drift_loss_empty_keys():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    loss, metrics = compute_activation_drift_loss({}, {}, threshold_deg=5.0)
    assert float(loss.item()) == pytest.approx(0.0)
    assert metrics == {}


@pytestmark_mlx
def test_compute_activation_drift_loss_multi_layer():
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    ref = mx.zeros((2, 4))
    ref = ref.at[:, 0].add(1.0)
    pol = mx.zeros((2, 4))
    pol = pol.at[:, 1].add(1.0)

    loss, metrics = compute_activation_drift_loss(
        {"0": ref, "1": ref}, {"0": pol, "1": pol}, threshold_deg=5.0
    )
    # Two layers, each contributing same amount
    single_excess = (90.0 - 5.0) * math.pi / 180.0
    assert float(loss.item()) == pytest.approx(2 * single_excess, abs=0.1)
    assert "0" in metrics and "1" in metrics


# ── Capture activations tests (mocked model) ────────────────────────────────


@pytestmark_mlx
def test_capture_activations_forward_basic():
    from mlx_grpo.loss.gar_ops import capture_activations_forward

    # Build a minimal mock model
    hidden_dim = 8

    class MockLayer:
        def __call__(self, h, mask=None, cache=None):
            return h + 0.1  # simple transformation

    class MockBaseModel:
        def __init__(self):
            self.layers = [MockLayer() for _ in range(4)]

        def embed_tokens(self, x):
            return mx.ones((1, x.shape[1], hidden_dim))

    class MockModel:
        def __init__(self):
            self.model = MockBaseModel()

    model = MockModel()
    result = capture_activations_forward(model, [1, 2, 3], lora_layers=[0, 2])
    assert "0" in result
    assert "2" in result
    assert "1" not in result
    assert "3" not in result
    assert result["0"].shape == (3, hidden_dim)
    assert result["2"].shape == (3, hidden_dim)


@pytestmark_mlx
def test_capture_activations_forward_selective_layers():
    from mlx_grpo.loss.gar_ops import capture_activations_forward

    hidden_dim = 4

    class MockLayer:
        def __call__(self, h, mask=None, cache=None):
            return h * 1.1

    class MockBaseModel:
        def __init__(self):
            self.layers = [MockLayer() for _ in range(6)]

        def embed_tokens(self, x):
            return mx.ones((1, x.shape[1], hidden_dim))

    class MockModel:
        def __init__(self):
            self.model = MockBaseModel()

    model = MockModel()
    result = capture_activations_forward(model, [10, 20], lora_layers=[3, 5])
    assert set(result.keys()) == {"3", "5"}
    for key in result:
        assert result[key].shape == (2, hidden_dim)
