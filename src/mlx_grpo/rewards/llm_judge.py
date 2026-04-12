"""LLM-as-judge reward — graded correctness via external API (instructor + OpenAI)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from pydantic import BaseModel, Field

from ..engine.result import RolloutResult
from ..types.builtins.math_helpers import _extract_boxed, _extract_last_boxed
from .base import BaseReward
from .judge_logger import JudgeCallRecord, JudgeLogger

_log = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    """Structured output schema for the LLM judge."""

    correct: bool = Field(description="Whether the model's answer matches the ground truth")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Judgment confidence")
    extracted_answer: str = Field(default="", description="Extracted answer from model response")


_JUDGE_PROMPT = """\
You are a math answer verifier. Determine if the model's answer is correct.

Problem: {problem}
Ground truth: {ground_truth}
Model reasoning (last 300 chars): {reasoning_tail}
Model answer: {predicted}

Are these mathematically equivalent? Account for different notations (fractions, \
decimals, scientific), equivalent expressions (2/4 = 1/2 = 0.5), LaTeX formatting \
differences, units (must match if ground truth has them), MCQ letter matches.
Extract the model's answer, then judge correctness.
Confidence: 1.0 = obviously equivalent/different, 0.5 = ambiguous."""


def _safe_format(template: str, variables: dict[str, str]) -> str:
    """Replace {key} placeholders for known keys only.

    Unlike str.format(), this will NOT crash on unknown brace patterns
    like LaTeX \\dfrac{1}{2} or other literal braces in the template.
    Only keys present in `variables` are substituted.
    """
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", str(value))
    return result


class LLMJudgeReward(BaseReward, name="llm_judge"):
    """Graded reward via external LLM judge. Falls back to local reward on API failure."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_retries: int = 2,
        api_max_retries: int = 3,
        timeout: int = 10,
        fallback_reward: str = "math",
        judge_prompt: str | None = None,
        graded: bool = True,
        cache_size: int = 256,
        judge_log_path: str | None = None,
        template_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self._model, self._base_url = model, base_url
        self._api_key_env, self._temperature = api_key_env, temperature
        self._max_retries, self._api_max_retries = max_retries, api_max_retries
        self._timeout, self._fallback_name = timeout, fallback_reward
        self._judge_prompt = judge_prompt or _JUDGE_PROMPT
        self._graded, self._cache_size = graded, cache_size
        self._client: Any = None
        self._fallback: BaseReward | None = None
        self._fallback_only: bool = False
        self._cache: dict[tuple[str, str], JudgeVerdict] = {}
        self._api_calls = self._api_failures = self._cache_hits = 0
        self._template_var_fns: dict[str, Any] = {}
        if template_vars:
            from ..types.yaml_type import parse_lambda

            for name, expr in template_vars.items():
                self._template_var_fns[name] = parse_lambda(expr)
        self._logger: JudgeLogger | None = JudgeLogger(judge_log_path) if judge_log_path else None

        # ── VERBOSE INIT ──
        _log.info("=" * 60)
        _log.info("LLMJudgeReward INIT (safe_format=YES)")
        _log.info("  model=%s", model)
        _log.info("  base_url=%s", base_url)
        _log.info("  graded=%s", graded)
        _log.info("  fallback=%s", fallback_reward)
        _log.info("  api_key_env=%s (set=%s)", api_key_env, bool(os.environ.get(api_key_env)))
        _log.info("  cache_size=%d", cache_size)
        _log.info("  prompt preview: %s", self._judge_prompt[:150].replace("\n", " "))
        _log.info("=" * 60)
        # After line 96 in the current file, add:
        _log.info(
            "ENV KEYS with 'KEY' or 'API': %s",
            [k for k in os.environ if "KEY" in k.upper() or "API" in k.upper()],
        )

    def _get_client(self) -> Any:
        """Lazy-init instructor-patched OpenAI client."""
        if self._client is not None:
            return self._client
        import instructor
        from openai import OpenAI

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"LLMJudgeReward: {self._api_key_env} not set in environment")
        _log.info("LLMJudge: creating OpenAI client (base_url=%s)", self._base_url)
        raw = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._api_max_retries,
        )
        self._client = instructor.from_openai(raw)
        _log.info("LLMJudge: client ready")
        return self._client

    def _get_fallback(self) -> BaseReward:
        if self._fallback is None:
            self._fallback = BaseReward.get(self._fallback_name)
            _log.info("LLMJudge: fallback initialized → %s", self._fallback_name)
        return self._fallback

    def _extract_predicted(self, result: RolloutResult) -> str | None:
        predicted = _extract_last_boxed(result.completion_text or result.generated_text)
        if predicted is None:
            av = result.variables.get("answer")
            if av and av.text.strip():
                raw = av.text.strip()
                bi = raw.find("}")
                predicted = raw[:bi].strip() if bi > 0 else raw.strip()
        return predicted

    def _extract_problem(self, result: RolloutResult) -> str:
        q = result.metadata.get("question") or result.metadata.get("problem", "")
        return str(q)[:500] if q else (result.prompt_text or "")[-500:]

    def _reasoning_tail(self, result: RolloutResult) -> str:
        thinking = result.segment_text("thinking")
        return thinking[-300:] if thinking else (result.generated_text or "")[-300:]

    @staticmethod
    def _split_expert(answer: str) -> tuple[str, str]:
        """Split expert answer into (thinking, post-thinking) on </think> boundary."""
        idx = answer.find("</think>")
        if idx < 0:
            return "", answer
        think = answer[:idx]
        if think.startswith("<think>"):
            think = think[len("<think>") :]
        return think.strip(), answer[idx + len("</think>") :].strip()

    def _build_template_vars(
        self,
        result: RolloutResult,
        answer: str,
        gt: str,
        predicted: str,
        problem: str,
        tail: str,
    ) -> dict[str, str]:
        cot = result.segment_text("thinking")
        answer_text = result.segment_text("answer")
        expert_cot, expert_answer = self._split_expert(answer)
        vs = {
            "ground_truth": gt,
            "predicted": predicted,
            "problem": problem,
            "reasoning_tail": tail,
            "cot": cot,
            "answer": answer_text,
            "expert_cot": expert_cot,
            "expert_answer": expert_answer,
        }
        for name, fn in self._template_var_fns.items():
            try:
                vs[name] = str(fn(result, answer))
            except Exception:
                vs[name] = ""
        return vs

    def _judge(
        self,
        gt: str,
        predicted: str,
        problem: str,
        tail: str,
        template_vars: dict[str, str] | None = None,
    ) -> JudgeVerdict:
        key = (gt, predicted)
        if key in self._cache:
            self._cache_hits += 1
            _log.info("LLMJudge CACHE HIT: gt='%s' pred='%s'", gt[:30], predicted[:30])
            return self._cache[key]

        self._api_calls += 1
        tv = template_vars or {
            "ground_truth": gt,
            "predicted": predicted,
            "problem": problem,
            "reasoning_tail": tail,
        }

        _log.info("LLMJudge FORMAT: keys=%s", list(tv.keys()))
        prompt_text = _safe_format(self._judge_prompt, tv)
        _log.info("LLMJudge PROMPT (first 200 chars): %s", prompt_text[:200].replace("\n", " "))

        client = self._get_client()
        _log.info(
            "LLMJudge API CALL #%d: gt='%s' pred='%s'", self._api_calls, gt[:40], predicted[:40]
        )

        verdict: JudgeVerdict = client.chat.completions.create(
            model=self._model,
            response_model=JudgeVerdict,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=self._temperature,
            max_retries=self._max_retries,
        )
        _log.info(
            "LLMJudge VERDICT: correct=%s conf=%.2f extracted='%s'",
            verdict.correct,
            verdict.confidence,
            verdict.extracted_answer[:50],
        )

        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = verdict
        return verdict

    def _reward_from_verdict(self, verdict: JudgeVerdict) -> float:
        if not self._graded:
            reward = 1.0 if verdict.correct else 0.0
            _log.info("LLMJudge REWARD (binary): correct=%s → %.1f", verdict.correct, reward)
            return reward
        if verdict.correct:
            reward = 0.5 + 0.5 * verdict.confidence
        else:
            reward = 0.5 * (1.0 - verdict.confidence)
        _log.info(
            "LLMJudge REWARD (graded): correct=%s conf=%.2f → %.3f",
            verdict.correct,
            verdict.confidence,
            reward,
        )
        return reward

    def _log_record(self, gt: str, predicted: str, problem: str, **kw: Any) -> None:
        if self._logger:
            self._logger.log(
                JudgeCallRecord(
                    timestamp=time.time(),
                    ground_truth=gt,
                    predicted=predicted,
                    problem=problem,
                    **kw,
                )
            )

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        if answer is None:
            _log.warning("LLMJudge COMPUTE: answer=None → 0.0")
            return 0.0
        if self._fallback_only:
            _log.info("LLMJudge COMPUTE: permanent fallback → %s", self._fallback_name)
            return self._get_fallback().compute(result, answer)

        predicted = self._extract_predicted(result)
        if predicted is None:
            _log.warning("LLMJudge COMPUTE: no predicted answer extracted → 0.0")
            return 0.0

        gt = _extract_boxed(answer) or answer.strip()
        problem = self._extract_problem(result)
        tail = self._reasoning_tail(result)
        tv = self._build_template_vars(result, answer, gt, predicted, problem, tail)
        t0 = time.monotonic()

        _log.info("LLMJudge COMPUTE: gt='%s' pred='%s'", gt[:50], predicted[:50])

        try:
            verdict = self._judge(gt, predicted, problem, tail, template_vars=tv)
            reward = self._reward_from_verdict(verdict)
            latency = (time.monotonic() - t0) * 1000
            result.metadata["_judge_extracted"] = verdict.extracted_answer
            result.metadata["_judge_confidence"] = verdict.confidence
            result.metadata["_judge_correct"] = verdict.correct
            _log.info(
                "LLMJudge OK: '%s' → %s (conf=%.2f, reward=%.3f, %.0fms)",
                predicted[:30],
                "✓" if verdict.correct else "✗",
                verdict.confidence,
                reward,
                latency,
            )
            self._log_record(
                gt,
                predicted,
                problem,
                correct=verdict.correct,
                confidence=verdict.confidence,
                extracted_answer=verdict.extracted_answer,
                reward=reward,
                cache_hit=False,
                latency_ms=latency,
            )
            return reward
        except (ImportError, RuntimeError) as e:
            self._fallback_only = True
            _log.error(
                "LLMJudge PERMANENT FAIL: %s (%s) — switching to %s forever",
                e,
                type(e).__name__,
                self._fallback_name,
            )
            return self._get_fallback().compute(result, answer)
        except Exception as e:
            self._api_failures += 1
            latency = (time.monotonic() - t0) * 1000
            _log.error(
                "LLMJudge API FAIL #%d/%d: %s (%s, %.0fms) — fallback=%s",
                self._api_failures,
                self._api_calls,
                e,
                type(e).__name__,
                latency,
                self._fallback_name,
            )
            fb = self._get_fallback().compute(result, answer)
            _log.info("LLMJudge FALLBACK: reward=%.3f (from %s)", fb, self._fallback_name)
            result.metadata["_judge_fallback"] = True
            result.metadata["_judge_fallback_reward"] = fb
            self._log_record(
                gt,
                predicted,
                problem,
                reward=fb,
                fallback=True,
                latency_ms=latency,
                error=str(e),
            )
            return fb

    @property
    def stats(self) -> dict[str, float]:
        return {
            "judge_api_calls": self._api_calls,
            "judge_cache_hits": self._cache_hits,
            "judge_api_failures": self._api_failures,
        }

    @property
    def judge_logger(self) -> JudgeLogger | None:
        return self._logger
