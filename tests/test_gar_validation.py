"""GAR Phase 4 — final validation tests: edge cases, E2E, backward compat."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.config.loader import TrainingConfig, load_config
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training import StepMetrics, Trainer
from mlx_grpo.training.rollout_logger import RolloutLogger, StepContext

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
_GAR_CFG = dict(mode="gar", iterations=1, group_size=2, save_every=0, eval_every=0)


def _dataset() -> DatasetLoader:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps({"prompt": f"Q{i}", "answer": f"A{i}"}) + "\n")
        f.flush()
        return DatasetLoader(DatasetConfig(path=f.name, shuffle=False)).load()


def _rollout(*, naked: bool = False, tokens: list[int] | None = None) -> RolloutResult:
    toks = tokens or [1, 2, 3, 4, 5]
    segs = [
        Segment(
            tokens=toks, text="out", source=SegmentSource.GENERATED, masked=False, tag="thinking"
        )
    ]
    if not naked:
        segs.insert(
            0,
            Segment(
                tokens=[10, 11],
                text="sc",
                source=SegmentSource.INJECTED,
                masked=True,
                tag="scaffold_hint",
            ),
        )
    r = RolloutResult(segments=segs, type_name="general_qna")
    r.metadata["_view_is_naked"] = naked
    r.metadata["_view_scaffold_ratio"] = 0.0 if naked else 0.5
    return r


# ── Edge Cases ──────────────────────────────────────────────────────────────


@pytestmark_mlx
def test_gar_combined_weighting_bounded():
    """Combined mask * proximity weights (4.0 * 5.0 = 20x) must not produce NaN/Inf."""
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    ref = mx.zeros((5, 8)).at[:, 0].add(1.0)
    pol = mx.zeros((5, 8)).at[:, 1].add(1.0)
    loss, metrics = compute_activation_drift_loss(
        {"0": ref},
        {"0": pol},
        threshold_deg=5.0,
        masks=[4.0] * 5,
        proximity_weights=[5.0] * 5,
    )
    val = float(loss.item())
    assert math.isfinite(val) and val > 0
    assert metrics["0"]["tokens_above"] > 0


@pytestmark_mlx
def test_gar_zero_overlapping_tokens_graceful():
    """Zero active tokens (all weight=0) returns loss=0, no NaN, no divide-by-zero."""
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    ref = mx.ones((3, 4))
    pol = mx.zeros((3, 4)).at[:, 1].add(1.0)
    loss, metrics = compute_activation_drift_loss(
        {"0": ref},
        {"0": pol},
        threshold_deg=5.0,
        masks=[0.0, 0.0, 0.0],
    )
    val = float(loss.item())
    assert val == pytest.approx(0.0, abs=1e-6) and math.isfinite(val)
    assert metrics["0"]["tokens_above"] == 0.0
    assert metrics["0"]["tokens_below"] == 0.0


@pytestmark_mlx
def test_gar_with_proximity_weights_applied():
    """Proximity weights decay reduces loss vs unweighted; all-zero weights = zero loss."""
    from mlx_grpo.loss.gar_ops import compute_activation_drift_loss

    ref = mx.zeros((5, 8)).at[:, 0].add(1.0)
    pol = mx.zeros((5, 8)).at[:, 1].add(1.0)
    loss_w, _ = compute_activation_drift_loss(
        {"0": ref},
        {"0": pol},
        threshold_deg=5.0,
        proximity_weights=[1.0, 0.7, 0.5, 0.3, 0.0],
    )
    loss_u, _ = compute_activation_drift_loss({"0": ref}, {"0": pol}, threshold_deg=5.0)
    assert float(loss_w.item()) < float(loss_u.item()), "Decayed weights should reduce loss"
    loss_z, _ = compute_activation_drift_loss(
        {"0": ref},
        {"0": pol},
        threshold_deg=5.0,
        proximity_weights=[0.0] * 5,
    )
    assert float(loss_z.item()) == pytest.approx(0.0, abs=1e-6)


# ── E2E Validation ──────────────────────────────────────────────────────────


def test_gar_e2e_training_step(mock_backend):
    """GAR trainer runs 1 iteration, populates gar_loss/gar_angles/gar_scaffold_fraction."""
    t = Trainer(config=TrainingConfig(**_GAR_CFG), dataset=_dataset(), backend=mock_backend)
    captured: list[StepMetrics] = []
    t.on("after_step", lambda _t, m: captured.append(m))
    t.train()
    assert len(captured) >= 1
    m = captured[0]
    assert math.isfinite(m.loss)
    for key in ("gar_loss", "gar_angles", "gar_scaffold_fraction"):
        assert key in m.extras
    assert isinstance(m.extras["gar_scaffold_fraction"], float)


def test_gar_e2e_loss_trend_stable(mock_backend):
    """GAR trainer runs 3 iterations without NaN or crash."""
    cfg = TrainingConfig(mode="gar", iterations=3, group_size=2, save_every=0, eval_every=0)
    t = Trainer(config=cfg, dataset=_dataset(), backend=mock_backend)
    captured: list[StepMetrics] = []
    t.on("after_step", lambda _t, m: captured.append(m))
    t.train()
    assert len(captured) == 3
    for m in captured:
        assert math.isfinite(m.loss) and "gar_loss" in m.extras


# ── Backward Compatibility ──────────────────────────────────────────────────


def test_grpo_mode_unchanged_config():
    """GRPO config defaults are unaffected by GAR addition."""
    cfg = TrainingConfig()
    assert cfg.mode == "grpo"
    assert cfg.gar_drift_threshold_deg == pytest.approx(5.0)
    assert cfg.gar_normalize is True


def test_grpo_mode_unchanged_loss(mock_backend):
    """GRPO training step produces no gar_* metrics."""
    cfg = TrainingConfig(iterations=2, group_size=2, save_every=0, eval_every=0)
    t = Trainer(config=cfg, dataset=_dataset(), backend=mock_backend)
    assert t._gar_mode is False
    captured: list[StepMetrics] = []
    t.on("after_step", lambda _t, m: captured.append(m))
    t.train()
    for m in captured:
        assert "gar_loss" not in m.extras and "gar_angles" not in m.extras


# ── Config Validation ───────────────────────────────────────────────────────


def test_gar_config_yaml_validation(tmp_path):
    """Full GAR YAML round-trips through load_config correctly."""
    yml = tmp_path / "gar.yml"
    yml.write_text(
        "training:\n  mode: gar\n  gar_drift_threshold_deg: 5.0\n  gar_normalize: true\n"
        "  group_size: 2\n  lora_layers: 16\n  adapter_path: adapters/gar-spike-qwen4b-math\n"
    )
    cfg = load_config(str(yml))
    assert cfg.training.mode == "gar"
    assert cfg.training.gar_drift_threshold_deg == pytest.approx(5.0)
    assert cfg.training.gar_normalize is True
    assert cfg.training.group_size == 2
    assert cfg.training.adapter_path == "adapters/gar-spike-qwen4b-math"


# ── Integration Smoke Tests ─────────────────────────────────────────────────


def test_gar_training_all_backends_compat(mock_backend):
    """Mock backend works for GAR; Trainer has GAR gradient step method."""
    t = Trainer(config=TrainingConfig(**_GAR_CFG), dataset=_dataset(), backend=mock_backend)
    t.train()  # no crash = pass
    assert hasattr(Trainer, "_gar_gradient_step")
    assert hasattr(Trainer, "_gar_gradient_step_mlx")
    # GRPO mode with mock backend also works, not routed through GAR
    cfg_grpo = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
    t2 = Trainer(config=cfg_grpo, dataset=_dataset(), backend=mock_backend)
    assert t2._gar_mode is False
    t2.train()


def test_gar_metrics_complete_roundtrip():
    """JSONL round-trip: gar_mode, gar_activation_drift, reward=0, advantage=0."""
    r = _rollout(naked=True)
    r.gar_activation_drift = {"20": {"angle_mean": 3.2, "angle_std": 1.5, "angle_max": 8.1}}
    entry = r.to_log_entry(include_tokens=False)
    assert entry["gar_mode"] is True and "20" in entry["gar_activation_drift"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "rollouts.jsonl"
        logger = RolloutLogger(path, log_tokens=False)
        ctx = StepContext(
            iteration=1, total_iterations=10, loss=0.0, gar_threshold_deg=5.0, gar_normalize=True
        )
        logger.log(iteration=1, rollouts=[r], rewards=[0.0], advantages=[0.0], step_ctx=ctx)
        logger.close()
        parsed = json.loads(path.read_text().strip().splitlines()[0])
        assert parsed["gar_mode"] is True
        assert parsed["gar_activation_drift"]["20"]["angle_mean"] == pytest.approx(3.2)
        assert parsed["reward"] == 0.0 and parsed["advantage"] == 0.0
        assert parsed["step"]["metadata"]["gar_threshold_deg"] == pytest.approx(5.0)
        assert parsed["step"]["metadata"]["gar_normalize"] is True
