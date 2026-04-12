"""General QnA task type — think → prose answer.

For open-ended questions where accuracy is measured by
semantic overlap rather than exact match.
"""

from __future__ import annotations

import re
from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType
from ._anchor import inject_reason_anchor


class GeneralQnAType(TaskType, name="general_qna"):
    """Open-ended question answering type."""

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.7,
            "thinking_budget": 768,
            "answer_max_tokens": 256,
            "recovery_ratio": 0.7,
            "reason_anchor": False,
            "reason_anchor_masked": True,
            "reason_anchor_weight": 1.0,
            "reason_anchor_max_words": 80,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            # system_prompt=(
            #     "You are a helpful assistant. Think carefully about the question, "
            #     "then provide a clear, accurate answer."
            # ),
            enable_thinking=True,
        )

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config

        # Curriculum scaffolding
        sc = ctx.batch.scaffold
        if ctx.answer and ctx.batch.scaffold_thinking_ratio > sc.min_ratio:
            tokens = ctx.encode(ctx.callbacks.strip_boxed_tail(ctx.answer))
            budget = sc.effective_tokens(
                ctx.batch.progress,
                len(tokens),
                ctx.batch.group_index,
            )
            if budget > 0:
                sliced = ctx.callbacks.slice_answer_tokens(tokens, budget, sc.mode)
                if sliced:
                    ctx.inject(ctx.decode(sliced), masked=True, tag="scaffold")

        inject_reason_anchor(ctx)

        # Phase 1: Thinking — reserve tokens for answer phase
        thinking_budget = cfg.get("thinking_budget", 768)
        answer_reserve = cfg.get("answer_max_tokens", 256)
        if ctx.max_completion_tokens > 0:
            thinking_budget = min(
                thinking_budget,
                max(1, ctx.max_completion_tokens - answer_reserve),
            )
        thinking = ctx.generate(
            max_tokens=thinking_budget,
            stop=["</think>"],
            temperature=cfg.get("temperature", 0.7),
            tag="thinking",
        )
        ctx.capture("thinking", thinking)

        if thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
            if ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.7)):
                ctx.inject("</think>\n", masked=True)
            else:
                return

        # Phase 2: Answer
        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 256),
            temperature=0.5,
            tag="answer",
        )
        ctx.capture("answer", answer)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        if answer is None:
            return 0.3  # Can't verify, give partial credit

        completion = result.completion_text
        if not completion.strip():
            return 0.0

        # Extract the answer portion (after thinking)
        answer_var = result.variables.get("answer")
        pred = answer_var.text if answer_var else completion

        # Simple keyword overlap scoring
        pred_words = set(_tokenize(pred.lower()))
        gt_words = set(_tokenize(answer.lower()))

        if not gt_words:
            return 0.3

        overlap = pred_words & gt_words
        precision = len(overlap) / max(len(pred_words), 1)
        recall = len(overlap) / max(len(gt_words), 1)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)

        # Bonus for containing key answer tokens
        return min(1.0, f1 * 1.2)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization for overlap scoring."""
    # Remove common stop words
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "it",
        "and",
        "or",
        "but",
        "not",
        "this",
        "that",
        "from",
        "as",
    }
    words = re.findall(r"\b\w+\b", text)
    return [w for w in words if w not in stop and len(w) > 1]
