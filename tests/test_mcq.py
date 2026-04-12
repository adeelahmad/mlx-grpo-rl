"""Tests for MCQ built-in type."""

from types import SimpleNamespace

import mlx_grpo.types.builtins  # noqa: F401 — trigger registration
from mlx_grpo.engine.callbacks import CallbackRegistry
from mlx_grpo.engine.context import BatchContext, RolloutContext
from mlx_grpo.types.base import TaskType


class TestMCQType:
    def setup_method(self):
        self.t = TaskType.get("mcq")

    def test_registered(self):
        assert "mcq" in TaskType.list_types()

    def test_config(self):
        cfg = self.t.config()
        assert "temperature" in cfg
        assert "thinking_budget" in cfg
        assert cfg["recovery_ratio"] == 1.0

    def test_chat_params(self):
        cp = self.t.chat_params()
        assert cp.enable_thinking is True
        assert cp.system_prompt is not None  # MCQ has a default system prompt

    def test_chat_messages(self):
        msgs = self.t.chat_messages("What is 2+2?")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is 2+2?"

    def test_callbacks(self):
        cbs = self.t.callbacks()
        assert "is_valid_mcq" in cbs
        assert cbs["is_valid_mcq"]("The answer is (A)")
        assert not cbs["is_valid_mcq"]("no letter here")

    def test_rollout_generates_segments(self, mock_backend_factory):
        backend = mock_backend_factory(
            call_1="Let me think about this</think>",  # thinking
            call_2="(B",  # answer
        )
        cb = CallbackRegistry(seed=0)
        cb.register_many(self.t.callbacks())
        batch = BatchContext(iteration=400, total_iters=500, group_size=1)

        ctx = RolloutContext(
            backend,
            prompt="What is 2+2?",
            prompt_tokens=backend.encode("What is 2+2?"),
            answer="B",
            type_name="mcq",
            type_config=self.t.config(),
            batch=batch,
            callbacks=cb,
        )
        self.t.rollout(ctx)
        result = ctx.build_result()

        assert result.generated_token_count > 0
        assert any(s.tag == "thinking" for s in result.segments)

    def test_reward_correct(self):
        from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource

        # Properly formatted: <think> with reasoning + \boxed{B}
        result = RolloutResult(
            segments=[
                Segment(
                    list(range(30)),
                    "<think>The question asks about several options. "
                    "Because option B matches the requirement, since it is "
                    "correct. Therefore the answer is B.</think>\n",
                    SegmentSource.GENERATED,
                    False,
                    tag="thinking",
                ),
                Segment([90], "\\boxed{", SegmentSource.INJECTED, True),
                Segment([91], "B}", SegmentSource.GENERATED, False, tag="answer"),
            ]
        )
        score = self.t.reward(result, "B")
        assert score > 0.5, f"Correct answer should score > 0.5, got {score}"

    def test_reward_incorrect(self):
        from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource

        result = RolloutResult(
            segments=[
                Segment(
                    list(range(30)),
                    "<think>The question asks about several options. "
                    "Because option A matches the requirement, since it is "
                    "correct. Therefore the answer is A.</think>\n",
                    SegmentSource.GENERATED,
                    False,
                    tag="thinking",
                ),
                Segment([90], "\\boxed{", SegmentSource.INJECTED, True),
                Segment([91], "A}", SegmentSource.GENERATED, False, tag="answer"),
            ]
        )
        score = self.t.reward(result, "B")
        assert score < 0.5, f"Wrong answer should score < 0.5, got {score}"

    def test_rollout_inject_target_thinking(self, mock_backend_factory):
        """inject_target_thinking=True injects expert trace after </think>."""
        backend = mock_backend_factory(
            call_1="Let me think step by step.</think>",  # thinking stops cleanly
            call_2="B",  # answer
        )
        cb = CallbackRegistry(seed=0)
        cb.register_many(self.t.callbacks())
        batch = BatchContext(iteration=400, total_iters=500, group_size=1)
        cfg = {**self.t.config(), "inject_target_thinking": True}

        ctx = RolloutContext(
            backend,
            prompt="What is 2+2?",
            prompt_tokens=backend.encode("What is 2+2?"),
            # Dataset-style: full response with <think>...</think> in the answer field
            answer="<think>Expert reasoning: 2+2=4 so the answer is B.</think>B",
            type_name="mcq",
            type_config=cfg,
            batch=batch,
            callbacks=cb,
        )
        self.t.rollout(ctx)
        result = ctx.build_result()
        tags = [s.tag for s in result.segments]
        assert "target_thinking" in tags, f"target_thinking segment missing; tags={tags}"


class TestTargetThinking:
    """Unit tests for the _target_thinking() module-level helper."""

    def _ctx(self, sample=None, answer=None):
        return SimpleNamespace(sample=sample or {}, answer=answer or "")

    def test_returns_none_when_empty(self):
        from mlx_grpo.types.builtins.mcq import _target_thinking

        assert _target_thinking(self._ctx()) is None

    def test_from_thinking_key(self):
        from mlx_grpo.types.builtins.mcq import _target_thinking

        ctx = self._ctx(sample={"thinking": "  Expert reasoning  "})
        assert _target_thinking(ctx) == "Expert reasoning"

    def test_from_target_thinking_key(self):
        from mlx_grpo.types.builtins.mcq import _target_thinking

        ctx = self._ctx(sample={"target_thinking": "Alt key reasoning"})
        assert _target_thinking(ctx) == "Alt key reasoning"

    def test_from_target_completion(self):
        from mlx_grpo.types.builtins.mcq import _target_thinking

        ctx = self._ctx(sample={"target_completion": "<think>From completion</think>answer"})
        assert _target_thinking(ctx) == "From completion"

    def test_from_completion_fallback(self):
        from mlx_grpo.types.builtins.mcq import _target_thinking

        ctx = self._ctx(sample={"completion": "<think>From completion field</think>B"})
        assert _target_thinking(ctx) == "From completion field"

    def test_from_answer_fallback(self):
        from mlx_grpo.types.builtins.mcq import _target_thinking

        ctx = self._ctx(answer="<think>From answer field</think>B")
        assert _target_thinking(ctx) == "From answer field"

    def test_no_think_tags_returns_none(self):
        from mlx_grpo.types.builtins.mcq import _target_thinking

        ctx = self._ctx(answer="Just plain answer B")
        assert _target_thinking(ctx) is None
