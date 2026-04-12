"""Tests for model loader config, training modes, and MLX ops sequence extraction."""

import pytest

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.loss.mlx_ops import (
    GradStepResult,
    _tree_norm,
    _tree_scale,
    extract_sequences_from_rollouts,
)
from mlx_grpo.training.model_loader import (
    AdapterConfig,
    AdapterType,
    ModelConfig,
)

# ── AdapterType ──


def test_adapter_type_from_str_lora():
    assert AdapterType.from_str("lora") == AdapterType.LORA
    assert AdapterType.from_str("LORA") == AdapterType.LORA
    assert AdapterType.from_str("  LoRa  ") == AdapterType.LORA


def test_adapter_type_from_str_dora():
    assert AdapterType.from_str("dora") == AdapterType.DORA
    assert AdapterType.from_str("DoRA") == AdapterType.DORA


def test_adapter_type_from_str_full():
    assert AdapterType.from_str("full") == AdapterType.FULL
    assert AdapterType.from_str("FULL") == AdapterType.FULL


def test_adapter_type_from_str_invalid():
    with pytest.raises(ValueError, match="Unknown adapter type"):
        AdapterType.from_str("qlora")


# ── AdapterConfig / ModelConfig ──


def test_adapter_config_defaults():
    cfg = AdapterConfig()
    assert cfg.rank == 8
    assert cfg.alpha == 16.0
    assert "self_attn.q_proj" in cfg.target_modules
    assert cfg.layers is None


def test_model_config_defaults():
    cfg = ModelConfig(model_path="Qwen/Qwen3-4B")
    assert cfg.model_path == "Qwen/Qwen3-4B"
    assert cfg.adapter_path is None
    assert cfg.adapter_type == "lora"
    assert cfg.adapter.rank == 8


def test_model_config_with_dora():
    cfg = ModelConfig(
        model_path="test",
        adapter_type="dora",
        adapter=AdapterConfig(rank=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj"]),
    )
    assert cfg.adapter_type == "dora"
    assert cfg.adapter.rank == 16
    assert len(cfg.adapter.target_modules) == 3


def test_model_config_full():
    cfg = ModelConfig(
        model_path="test",
        adapter_type="full",
        freeze_base=False,
    )
    assert cfg.adapter_type == "full"
    assert cfg.freeze_base is False


# ── DoRALinear (unit test without real MLX model) ──


def test_dora_linear_importable():
    """DoRALinear class exists and has expected interface."""
    from mlx_grpo.training.model_loader import DoRALinear

    assert hasattr(DoRALinear, "from_linear")
    assert callable(DoRALinear)


def test_dora_linear_is_dora_property():
    """DoRALinear marks itself as DoRA for adapter detection."""
    try:
        from mlx_grpo.training.model_loader import DoRALinear

        # Can't instantiate without MLX, but can check class has property
        assert hasattr(DoRALinear, "is_dora")
    except ImportError:
        pytest.skip("MLX not available")


# ── _AdapterReferenceModel / _SnapshotReferenceModel ──


def test_reference_model_classes_exist():
    """Both reference model strategies importable."""
    from mlx_grpo.training.model_loader import (
        _AdapterReferenceModel,
        _SnapshotReferenceModel,
    )

    assert callable(_AdapterReferenceModel)
    assert callable(_SnapshotReferenceModel)


# ── Training config with adapter_type ──


def test_training_config_adapter_type():
    from mlx_grpo.config.loader import TrainingConfig

    cfg = TrainingConfig(adapter_type="dora")
    assert cfg.adapter_type == "dora"

    cfg_full = TrainingConfig(adapter_type="full")
    assert cfg_full.adapter_type == "full"

    cfg_default = TrainingConfig()
    assert cfg_default.adapter_type == "lora"


def test_training_config_lora_fields():
    from mlx_grpo.config.loader import TrainingConfig

    cfg = TrainingConfig(lora_rank=16, lora_alpha=32.0)
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32.0
    assert "self_attn.q_proj" in cfg.lora_targets


# ── get_trainable_params / count_params ──


def test_get_trainable_params_full():
    """Full mode returns all parameters."""
    from mlx_grpo.training.model_loader import get_trainable_params

    class FakeModel:
        def parameters(self):
            return {"layer.weight": "w1", "layer.bias": "b1"}

    params = get_trainable_params(FakeModel(), AdapterType.FULL)
    assert "layer.weight" in params
    assert "layer.bias" in params


def test_get_trainable_params_lora():
    """LoRA mode returns only adapter parameters."""
    from mlx_grpo.training.model_loader import get_trainable_params

    class FakeModel:
        def parameters(self):
            return {
                "model.layers.0.self_attn.q_proj.weight": "w",
                "model.layers.0.self_attn.q_proj.lora_a": "la",
                "model.layers.0.self_attn.q_proj.lora_b": "lb",
            }

    params = get_trainable_params(FakeModel(), AdapterType.LORA)
    assert "model.layers.0.self_attn.q_proj.lora_a" in params
    assert "model.layers.0.self_attn.q_proj.lora_b" in params
    assert "model.layers.0.self_attn.q_proj.weight" not in params


def test_get_trainable_params_dora():
    """DoRA mode returns adapter + magnitude parameters."""
    from mlx_grpo.training.model_loader import get_trainable_params

    class FakeModel:
        def parameters(self):
            return {
                "model.layers.0.weight": "w",
                "model.layers.0.magnitude": "m",
                "model.layers.0.lora_a": "la",
                "model.layers.0.lora_b": "lb",
            }

    params = get_trainable_params(FakeModel(), AdapterType.DORA)
    assert "model.layers.0.magnitude" in params
    assert "model.layers.0.lora_a" in params
    assert "model.layers.0.lora_b" in params
    assert "model.layers.0.weight" not in params


# ── extract_sequences_from_rollouts ──


def _make_rollout(prompt_tokens, segments):
    return RolloutResult(segments=segments, prompt_tokens=prompt_tokens)


def test_extract_single_generated():
    r = _make_rollout(
        [1, 2, 3],
        [
            Segment(tokens=[10, 11], text="ab", source=SegmentSource.GENERATED, masked=False),
        ],
    )
    seqs, plens, masks = extract_sequences_from_rollouts([r], None)
    assert seqs == [[1, 2, 3, 10, 11]]
    assert plens == [3]
    assert masks == [[True, True]]


def test_extract_mixed_segments():
    r = _make_rollout(
        [1],
        [
            Segment(tokens=[10, 11, 12], text="abc", source=SegmentSource.GENERATED, masked=False),
            Segment(tokens=[20], text="<s>", source=SegmentSource.INJECTED, masked=True),
            Segment(tokens=[30, 31], text="de", source=SegmentSource.GENERATED, masked=False),
        ],
    )
    seqs, plens, masks = extract_sequences_from_rollouts([r], None)
    assert seqs == [[1, 10, 11, 12, 20, 30, 31]]
    assert plens == [1]
    assert masks == [[True, True, True, False, True, True]]


def test_extract_empty_rollout():
    r = _make_rollout([], [])
    seqs, plens, masks = extract_sequences_from_rollouts([r], None)
    assert seqs == [[]]
    assert plens == [0]
    assert masks == [[]]


def test_extract_multiple_rollouts():
    r1 = _make_rollout(
        [1, 2],
        [
            Segment(tokens=[10], text="a", source=SegmentSource.GENERATED, masked=False),
        ],
    )
    r2 = _make_rollout(
        [3, 4, 5],
        [
            Segment(tokens=[20, 21], text="bc", source=SegmentSource.GENERATED, masked=False),
        ],
    )
    seqs, plens, masks = extract_sequences_from_rollouts([r1, r2], None)
    assert len(seqs) == 2
    assert seqs[0] == [1, 2, 10]
    assert seqs[1] == [3, 4, 5, 20, 21]
    assert plens == [2, 3]


# ── GradStepResult ──


def test_grad_step_result():
    r = GradStepResult(loss=0.5, grad_norm=1.2, effective_tokens=100)
    assert r.loss == 0.5
    assert r.grad_norm == 1.2
    assert r.effective_tokens == 100


# ── Tree utilities ──


def test_tree_norm_empty():
    assert _tree_norm({}) == 0.0
    assert _tree_norm([]) == 0.0


def test_tree_norm_nested():
    assert _tree_norm({"a": 1.0, "b": {"c": 2.0}}) == 0.0


def test_tree_scale_dict():
    tree = {"a": {"b": 1.0}}
    result = _tree_scale(tree, 2.0)
    assert result["a"]["b"] == 1.0  # Non-array leaves unchanged


# ── Trainer with new signature ──


def test_trainer_accepts_new_args():
    """Verify Trainer accepts model/ref_model/optimizer."""
    import json
    import tempfile

    from mlx_grpo.config.loader import TrainingConfig
    from mlx_grpo.data import DatasetConfig, DatasetLoader
    from mlx_grpo.training import Trainer

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"prompt": "hi", "answer": "A", "type": "mcq"}], f)
        path = f.name

    ds = DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()

    class FakeBackend:
        def generate(self, tokens, params):
            from mlx_grpo.backends.base import GenerateOutput

            return GenerateOutput(text="A", tokens=[65], stop_reason="eos")

        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, tokens):
            return "".join(chr(t) for t in tokens)

    trainer = Trainer(
        config=TrainingConfig(iterations=1),
        dataset=ds,
        backend=FakeBackend(),
        model=None,
        ref_model=None,
        optimizer=None,
    )
    assert trainer.model is None
    assert trainer.ref_model is None
    assert not trainer._mlx_ops_available


# ── Context → build_result includes prompt_tokens ──


def test_context_preserves_prompt_tokens():
    """Verify RolloutContext.build_result() includes prompt_tokens."""
    from mlx_grpo.backends.base import GenerateOutput
    from mlx_grpo.engine.context import RolloutContext

    class StubBackend:
        def generate(self, tokens, params):
            return GenerateOutput(text="ok", tokens=[10, 11], stop_reason="eos")

        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, tokens):
            return "".join(chr(t) for t in tokens)

    ctx = RolloutContext(
        backend=StubBackend(),
        prompt="hello",
        prompt_tokens=[1, 2, 3, 4, 5],
    )
    ctx.generate(max_tokens=10)

    result = ctx.build_result()
    assert result.prompt_tokens == [1, 2, 3, 4, 5]
    assert result.completion_tokens == [10, 11]
