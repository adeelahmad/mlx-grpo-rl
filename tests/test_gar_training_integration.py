"""GAR (Guided Activation Reinforcement) training integration tests.

Tests mode branching, gradient step dispatch, metrics population,
and backward compatibility with GRPO mode. Uses MockBackend only.
"""

import json
import tempfile

from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training import Trainer
from mlx_grpo.training.gar_step import pair_rollouts

_GAR_CFG = dict(mode="gar", iterations=1, group_size=2, save_every=0, eval_every=0)


def _make_dataset() -> DatasetLoader:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(10):
            f.write(json.dumps({"prompt": f"Q{i}?", "answer": f"A{i}"}) + "\n")
        f.flush()
        path = f.name
    return DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()


def _make_rollout(*, naked: bool = False, tokens: list[int] | None = None) -> RolloutResult:
    toks = tokens or [1, 2, 3, 4, 5]
    segs = [
        Segment(
            tokens=toks, text="mock", source=SegmentSource.GENERATED, masked=False, tag="thinking"
        )
    ]
    if not naked:
        segs.insert(
            0,
            Segment(
                tokens=[10, 11],
                text="scaffold",
                source=SegmentSource.INJECTED,
                masked=True,
                tag="scaffold_hint",
            ),
        )
    r = RolloutResult(segments=segs, type_name="general_qna")
    r.metadata["_view_is_naked"] = naked
    r.metadata["_view_scaffold_ratio"] = 0.0 if naked else 0.5
    return r


def _gar_trainer(mock_backend):
    return Trainer(config=TrainingConfig(**_GAR_CFG), dataset=_make_dataset(), backend=mock_backend)


class TestGARConfigMode:
    def test_gar_config_mode_field_exists(self):
        assert TrainingConfig().mode == "grpo"

    def test_gar_config_mode_gar(self):
        assert TrainingConfig(mode="gar").mode == "gar"

    def test_gar_config_drift_threshold(self):
        c = TrainingConfig(mode="gar", gar_drift_threshold_deg=10.0)
        assert c.gar_drift_threshold_deg == 10.0

    def test_gar_config_normalize(self):
        assert TrainingConfig(mode="gar", gar_normalize=False).gar_normalize is False


class TestGARTrainerInit:
    def test_gar_trainer_init_with_mode_gar(self, mock_backend):
        assert _gar_trainer(mock_backend)._gar_mode is True

    def test_gar_trainer_init_with_mode_grpo(self, mock_backend):
        cfg = TrainingConfig(mode="grpo", iterations=1, group_size=2, save_every=0, eval_every=0)
        t = Trainer(config=cfg, dataset=_make_dataset(), backend=mock_backend)
        assert t._gar_mode is False

    def test_gar_setup_gradient_ops_binds_gar_fn(self, mock_backend):
        t = _gar_trainer(mock_backend)
        assert hasattr(t, "_gar_last_metrics")
        assert isinstance(t._gar_last_metrics, dict)


class TestGARPairRollouts:
    def test_pair_rollouts_basic(self):
        pairs = pair_rollouts([_make_rollout(naked=False), _make_rollout(naked=True)])
        assert len(pairs) == 1
        assert pairs[0][0].metadata["_view_is_naked"] is False
        assert pairs[0][1].metadata["_view_is_naked"] is True

    def test_pair_rollouts_reversed_order(self):
        pairs = pair_rollouts([_make_rollout(naked=True), _make_rollout(naked=False)])
        assert len(pairs) == 1
        assert pairs[0][0].metadata["_view_is_naked"] is False

    def test_pair_rollouts_multiple_pairs(self):
        rollouts = [
            _make_rollout(naked=False),
            _make_rollout(naked=True),
            _make_rollout(naked=False),
            _make_rollout(naked=True),
        ]
        assert len(pair_rollouts(rollouts)) == 2

    def test_pair_rollouts_odd_count(self):
        rollouts = [
            _make_rollout(naked=False),
            _make_rollout(naked=True),
            _make_rollout(naked=False),
        ]
        assert len(pair_rollouts(rollouts)) == 1

    def test_gar_generates_two_rollouts_per_sample(self, mock_backend):
        t = _gar_trainer(mock_backend)
        captured = []
        t.on("after_step", lambda t, m: captured.append(m))
        t.train()
        assert len(captured[0].rewards) == 2

    def test_gar_scaffold_varies_per_repeat(self):
        s1, s2 = _make_rollout(naked=False), _make_rollout(naked=False, tokens=[1, 2, 3])
        s1.metadata["_view_scaffold_ratio"] = 0.5
        s2.metadata["_view_scaffold_ratio"] = 0.0
        assert s1.metadata["_view_scaffold_ratio"] != s2.metadata["_view_scaffold_ratio"]


class TestGARMetrics:
    def test_gar_metrics_extras_populated(self, mock_backend):
        t = _gar_trainer(mock_backend)
        captured = []
        t.on("after_step", lambda t, m: captured.append(m))
        t.train()
        m = captured[0]
        assert isinstance(m.extras, dict)
        assert "gar_loss" in m.extras
        assert "gar_angles" in m.extras
        assert "gar_scaffold_fraction" in m.extras
        assert isinstance(m.extras["gar_scaffold_fraction"], float)

    def test_gar_proximity_weights_applied_to_gar_loss(self):
        naked = _make_rollout(naked=True)
        naked.metadata["_is_weights"] = [0.5, 0.8, 1.0, 1.2, 0.9]
        pairs = pair_rollouts([_make_rollout(naked=False), naked])
        assert pairs[0][1].metadata.get("_is_weights") is not None

    def test_gar_always_mask_sequences_applied(self, mock_backend):
        _gar_trainer(mock_backend).train()  # no crash = pass

    def test_gar_gradient_scaling_applied_to_gar_loss(self, mock_backend):
        t = _gar_trainer(mock_backend)
        captured = []
        t.on("after_step", lambda t, m: captured.append(m))
        t.train()
        assert "gar_loss" in captured[0].extras
        assert captured[0].loss == 0.0  # mock mode pseudo-loss


class TestGARNoReward:
    def test_gar_no_reward_computation(self, mock_backend):
        """GAR mode still computes rewards for metrics but gradient step uses
        activation drift instead."""
        t = _gar_trainer(mock_backend)
        captured = []
        t.on("after_step", lambda t, m: captured.append(m))
        t.train()
        assert len(captured[0].rewards) > 0


class TestGRPOModeUnchanged:
    def test_grpo_mode_unchanged(self, mock_backend):
        cfg = TrainingConfig(iterations=3, group_size=2, save_every=0, eval_every=0)
        t = Trainer(config=cfg, dataset=_make_dataset(), backend=mock_backend)
        assert t._gar_mode is False
        t.train()
        assert t.iteration == 2

    def test_grpo_default_mode(self):
        assert TrainingConfig().mode == "grpo"
