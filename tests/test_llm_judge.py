"""Tests for the LLM Judge reward — graded rewards, caching, fallback, logging."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mlx_grpo.engine.result import GenerateResult, RolloutResult, Segment, SegmentSource
from mlx_grpo.rewards.judge_logger import JudgeCallRecord, JudgeLogger
from mlx_grpo.rewards.llm_judge import _JUDGE_PROMPT, JudgeVerdict, LLMJudgeReward


def _make_result(
    completion: str = r"\boxed{42}",
    prompt_text: str = "What is 6*7?",
    thinking: str = "Let me think... 6 times 7 is 42.",
    metadata: dict | None = None,
) -> RolloutResult:
    """Build a minimal RolloutResult for judge testing."""
    segments = []
    if thinking:
        segments.append(
            Segment(
                tokens=[1] * 10,
                text=thinking,
                source=SegmentSource.GENERATED,
                masked=False,
                tag="thinking",
            )
        )
    segments.append(
        Segment(
            tokens=[2] * 5,
            text=completion,
            source=SegmentSource.GENERATED,
            masked=False,
            tag="answer",
        )
    )
    return RolloutResult(
        segments=segments,
        prompt_text=prompt_text,
        metadata=metadata or {},
        type_name="math",
    )


# ── JudgeVerdict schema ──


def test_verdict_defaults():
    v = JudgeVerdict(correct=True)
    assert v.confidence == 1.0
    assert v.extracted_answer == ""


def test_verdict_with_all_fields():
    v = JudgeVerdict(correct=False, confidence=0.3, extracted_answer="41")
    assert not v.correct
    assert v.confidence == 0.3
    assert v.extracted_answer == "41"


def test_verdict_confidence_clamped():
    with pytest.raises(ValueError):
        JudgeVerdict(correct=True, confidence=1.5)
    with pytest.raises(ValueError):
        JudgeVerdict(correct=True, confidence=-0.1)


# ── Prompt template ──


def test_prompt_has_all_placeholders():
    for key in ("problem", "ground_truth", "predicted", "reasoning_tail"):
        assert f"{{{key}}}" in _JUDGE_PROMPT


def test_prompt_format():
    filled = _JUDGE_PROMPT.format(
        problem="What is 2+2?",
        ground_truth="4",
        predicted="4",
        reasoning_tail="2+2=4",
    )
    assert "What is 2+2?" in filled
    assert "4" in filled


# ── Graded reward calculation ──


class TestGradedReward:
    def setup_method(self):
        self.judge = LLMJudgeReward(graded=True)

    def test_correct_full_confidence(self):
        v = JudgeVerdict(correct=True, confidence=1.0)
        assert self.judge._reward_from_verdict(v) == 1.0

    def test_correct_half_confidence(self):
        v = JudgeVerdict(correct=True, confidence=0.5)
        assert self.judge._reward_from_verdict(v) == 0.75

    def test_correct_zero_confidence(self):
        v = JudgeVerdict(correct=True, confidence=0.0)
        assert self.judge._reward_from_verdict(v) == 0.5

    def test_incorrect_full_confidence(self):
        v = JudgeVerdict(correct=False, confidence=1.0)
        assert self.judge._reward_from_verdict(v) == 0.0

    def test_incorrect_half_confidence(self):
        v = JudgeVerdict(correct=False, confidence=0.5)
        assert self.judge._reward_from_verdict(v) == 0.25

    def test_incorrect_zero_confidence(self):
        v = JudgeVerdict(correct=False, confidence=0.0)
        assert self.judge._reward_from_verdict(v) == 0.5

    def test_binary_mode_ignores_confidence(self):
        judge = LLMJudgeReward(graded=False)
        v = JudgeVerdict(correct=True, confidence=0.3)
        assert judge._reward_from_verdict(v) == 1.0
        v2 = JudgeVerdict(correct=False, confidence=0.9)
        assert judge._reward_from_verdict(v2) == 0.0


# ── Answer extraction ──


class TestExtraction:
    def setup_method(self):
        self.judge = LLMJudgeReward()

    def test_extract_boxed(self):
        result = _make_result(completion=r"Therefore \boxed{42}")
        assert self.judge._extract_predicted(result) == "42"

    def test_extract_from_variable(self):
        result = _make_result(completion="no boxed here")
        result.variables["answer"] = GenerateResult(text="42}")
        assert self.judge._extract_predicted(result) == "42"

    def test_extract_none_when_empty(self):
        result = _make_result(completion="no answer", thinking="just thinking")
        assert self.judge._extract_predicted(result) is None

    def test_extract_problem_from_metadata(self):
        result = _make_result(metadata={"question": "What is 6*7?"})
        assert self.judge._extract_problem(result) == "What is 6*7?"

    def test_extract_problem_fallback_to_prompt(self):
        result = _make_result(prompt_text="Solve: 6*7=?")
        assert "6*7" in self.judge._extract_problem(result)

    def test_reasoning_tail_from_thinking(self):
        result = _make_result(thinking="A" * 500)
        tail = self.judge._reasoning_tail(result)
        assert len(tail) == 300

    def test_reasoning_tail_fallback(self):
        result = _make_result(thinking="", completion="short answer")
        tail = self.judge._reasoning_tail(result)
        assert "short answer" in tail


# ── Cache behavior ──


class TestCache:
    def test_cache_hit(self):
        judge = LLMJudgeReward(cache_size=10)
        verdict = JudgeVerdict(correct=True, confidence=0.9)
        judge._cache[("4", "4")] = verdict
        result = judge._judge("4", "4", "problem", "tail")
        assert result is verdict
        assert judge._cache_hits == 1
        assert judge._api_calls == 0

    def test_cache_eviction(self):
        judge = LLMJudgeReward(cache_size=2)
        judge._cache[("a", "b")] = JudgeVerdict(correct=True)
        judge._cache[("c", "d")] = JudgeVerdict(correct=False)

        # Manually add a third (simulating API call)
        new_verdict = JudgeVerdict(correct=True, confidence=0.5)
        if len(judge._cache) >= judge._cache_size:
            judge._cache.pop(next(iter(judge._cache)))
        judge._cache[("e", "f")] = new_verdict

        assert ("a", "b") not in judge._cache
        assert ("c", "d") in judge._cache
        assert ("e", "f") in judge._cache


# ── Compute integration ──


class TestCompute:
    def test_none_answer_returns_zero(self):
        judge = LLMJudgeReward()
        result = _make_result()
        assert judge.compute(result, answer=None) == 0.0

    def test_no_predicted_returns_zero(self):
        judge = LLMJudgeReward()
        result = _make_result(completion="no answer here", thinking="")
        assert judge.compute(result, answer="42") == 0.0

    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_successful_judge_call(self, mock_judge):
        mock_judge.return_value = JudgeVerdict(
            correct=True,
            confidence=0.8,
            extracted_answer="42",
        )
        judge = LLMJudgeReward(graded=True)
        result = _make_result(completion=r"\boxed{42}")
        reward = judge.compute(result, answer=r"\boxed{42}")
        assert reward == 0.5 + 0.5 * 0.8  # 0.9
        assert result.metadata["_judge_correct"] is True
        assert result.metadata["_judge_confidence"] == 0.8
        assert result.metadata["_judge_extracted"] == "42"

    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_api_failure_uses_fallback(self, mock_judge):
        mock_judge.side_effect = ConnectionError("API timeout")
        mock_fallback = MagicMock()
        mock_fallback.compute.return_value = 0.7

        judge = LLMJudgeReward(fallback_reward="math")
        judge._fallback = mock_fallback

        result = _make_result(completion=r"\boxed{42}")
        reward = judge.compute(result, answer="42")
        assert reward == 0.7
        assert result.metadata["_judge_fallback"] is True
        assert result.metadata["_judge_fallback_reward"] == 0.7
        assert judge._api_failures == 1

    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_metadata_written_on_success(self, mock_judge):
        mock_judge.return_value = JudgeVerdict(
            correct=False,
            confidence=0.95,
            extracted_answer="41",
        )
        judge = LLMJudgeReward(graded=True)
        result = _make_result(completion=r"\boxed{41}")
        reward = judge.compute(result, answer="42")
        assert reward == 0.5 * (1.0 - 0.95)  # 0.025
        assert result.metadata["_judge_correct"] is False


# ── Client init ──


def test_missing_api_key_raises():
    judge = LLMJudgeReward(api_key_env="NONEXISTENT_KEY_12345")
    try:
        import instructor  # noqa: F401
        import openai  # noqa: F401

        with pytest.raises(RuntimeError, match="not set"):
            judge._get_client()
    except ImportError:
        pytest.skip("instructor/openai not installed")


def test_custom_prompt():
    custom = "Is {predicted} equal to {ground_truth}? Problem: {problem} Tail: {reasoning_tail}"
    judge = LLMJudgeReward(judge_prompt=custom)
    assert judge._judge_prompt == custom


# ── Permanent fallback ──


class TestPermanentFallback:
    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_import_error_sets_fallback_only(self, mock_judge):
        mock_judge.side_effect = ImportError("No module named 'instructor'")
        mock_fb = MagicMock()
        mock_fb.compute.return_value = 0.5
        judge = LLMJudgeReward()
        judge._fallback = mock_fb
        result = _make_result(completion=r"\boxed{42}")
        judge.compute(result, answer="42")
        assert judge._fallback_only is True
        # Second call skips _judge entirely
        mock_judge.reset_mock()
        judge.compute(result, answer="42")
        mock_judge.assert_not_called()

    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_runtime_error_sets_fallback_only(self, mock_judge):
        mock_judge.side_effect = RuntimeError("API_KEY not set")
        mock_fb = MagicMock()
        mock_fb.compute.return_value = 0.3
        judge = LLMJudgeReward()
        judge._fallback = mock_fb
        result = _make_result(completion=r"\boxed{42}")
        r = judge.compute(result, answer="42")
        assert r == 0.3
        assert judge._fallback_only is True


# ── Retry config ──


def test_retry_defaults():
    judge = LLMJudgeReward()
    assert judge._max_retries == 2
    assert judge._api_max_retries == 3


def test_custom_retry_values():
    judge = LLMJudgeReward(max_retries=5, api_max_retries=10)
    assert judge._max_retries == 5
    assert judge._api_max_retries == 10


# ── Template variables ──


class TestTemplateVars:
    def test_builtin_vars(self):
        judge = LLMJudgeReward()
        result = _make_result()
        tv = judge._build_template_vars(result, "42", "42", "42", "prob", "tail")
        assert tv["ground_truth"] == "42"
        assert tv["predicted"] == "42"
        assert tv["problem"] == "prob"
        assert tv["reasoning_tail"] == "tail"

    def test_custom_lambda_var(self):
        judge = LLMJudgeReward(
            template_vars={"difficulty": "lambda result, answer: 'hard'"},
        )
        result = _make_result()
        tv = judge._build_template_vars(result, "42", "42", "42", "prob", "tail")
        assert tv["difficulty"] == "hard"

    def test_lambda_var_from_metadata(self):
        judge = LLMJudgeReward(
            template_vars={"src": "lambda result, answer: result.metadata.get('source', 'unk')"},
        )
        result = _make_result(metadata={"source": "gsm8k"})
        tv = judge._build_template_vars(result, "42", "42", "42", "prob", "tail")
        assert tv["src"] == "gsm8k"

    def test_failing_lambda_returns_empty(self):
        judge = LLMJudgeReward(
            template_vars={"bad": "lambda result, answer: 1/0"},
        )
        result = _make_result()
        tv = judge._build_template_vars(result, "42", "42", "42", "prob", "tail")
        assert tv["bad"] == ""

    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_template_vars_passed_to_judge(self, mock_judge):
        mock_judge.return_value = JudgeVerdict(correct=True, confidence=1.0)
        judge = LLMJudgeReward(
            template_vars={"src": "lambda result, answer: 'test'"},
        )
        result = _make_result(completion=r"\boxed{42}")
        judge.compute(result, answer="42")
        _, kwargs = mock_judge.call_args
        assert kwargs["template_vars"]["src"] == "test"


# ── Stats property ──


def test_stats_property():
    judge = LLMJudgeReward()
    judge._api_calls = 10
    judge._cache_hits = 3
    judge._api_failures = 1
    stats = judge.stats
    assert stats["judge_api_calls"] == 10
    assert stats["judge_cache_hits"] == 3
    assert stats["judge_api_failures"] == 1


# ── JudgeLogger ──


class TestJudgeLogger:
    def test_log_writes_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        logger = JudgeLogger(path)
        logger.log(
            JudgeCallRecord(
                timestamp=1000.0,
                ground_truth="4",
                predicted="4",
                problem="2+2=?",
                correct=True,
                confidence=0.9,
                extracted_answer="4",
                reward=0.95,
                cache_hit=False,
                latency_ms=50.0,
            )
        )
        logger.close()
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["correct"] is True
        assert entry["confidence"] == 0.9
        assert entry["reward"] == 0.95
        assert entry["latency_ms"] == 50.0

    def test_step_stats_and_reset(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        logger = JudgeLogger(path)
        logger.log(
            JudgeCallRecord(
                timestamp=1.0,
                ground_truth="4",
                predicted="4",
                problem="q",
                correct=True,
                confidence=0.8,
                reward=0.9,
                latency_ms=30.0,
            )
        )
        logger.log(
            JudgeCallRecord(
                timestamp=2.0,
                ground_truth="5",
                predicted="5",
                problem="q",
                correct=False,
                confidence=0.6,
                reward=0.2,
                cache_hit=True,
                latency_ms=0.1,
            )
        )
        stats = logger.step_stats()
        assert stats["judge_calls"] == 2
        assert stats["judge_cache_hits"] == 1
        assert stats["judge_correct"] == 1
        assert abs(stats["judge_mean_confidence"] - 0.7) < 0.01
        # After reset
        stats2 = logger.step_stats()
        assert stats2["judge_calls"] == 0
        logger.close()

    def test_fallback_logged(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        logger = JudgeLogger(path)
        logger.log(
            JudgeCallRecord(
                timestamp=1.0,
                ground_truth="4",
                predicted="4",
                problem="q",
                fallback=True,
                error="timeout",
                latency_ms=10000.0,
            )
        )
        stats = logger.step_stats()
        assert stats["judge_failures"] == 1
        logger.close()

    def test_error_field_in_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        logger = JudgeLogger(path)
        logger.log(
            JudgeCallRecord(
                timestamp=1.0,
                ground_truth="4",
                predicted="4",
                problem="q",
                fallback=True,
                error="ConnectionTimeout",
                latency_ms=5000.0,
            )
        )
        logger.close()
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["error"] == "ConnectionTimeout"
        assert entry["fallback"] is True


# ── Integration: compute with logger ──


class TestComputeWithLogger:
    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_success_logged(self, mock_judge):
        mock_judge.return_value = JudgeVerdict(correct=True, confidence=0.9)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        judge = LLMJudgeReward(graded=True, judge_log_path=path)
        result = _make_result(completion=r"\boxed{42}")
        judge.compute(result, answer="42")
        judge._logger.close()
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["correct"] is True
        assert entry["cache_hit"] is False

    @patch("mlx_grpo.rewards.llm_judge.LLMJudgeReward._judge")
    def test_failure_logged(self, mock_judge):
        mock_judge.side_effect = ConnectionError("boom")
        mock_fb = MagicMock()
        mock_fb.compute.return_value = 0.5
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        judge = LLMJudgeReward(judge_log_path=path)
        judge._fallback = mock_fb
        result = _make_result(completion=r"\boxed{42}")
        judge.compute(result, answer="42")
        judge._logger.close()
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["fallback"] is True
        assert "boom" in entry["error"]
