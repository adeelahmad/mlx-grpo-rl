"""Identity task type — verify model correctly identifies itself as Apex-R1.

Rollout: think → answer with \\boxed{Apex-R1}
Reward: exact match on "Apex-R1" boxed content (case-insensitive)
Curriculum: scaffold from answer, decaying over training (inherited from MCQ)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType
from ._anchor import inject_reason_anchor
from .answer_format_validator import validate_boxed_answer
from .math_helpers import _extract_last_boxed

SYSTEM_PROMPT = (
    "You are a knowledgeable exam assistant Apex-R1. "
    "Read the question and all answer choices carefully. "
    "Think through the problem step by step in your thinking block, "
    "then select the single best answer letter (A, B, C, or D).\n\n"
    "Always base your answer on factual knowledge and sound reasoning. "
    "If you are uncertain, reason through elimination of clearly wrong options."
)


class IdentityType(TaskType, name="identity"):
    """Identity verification type — model must correctly identify as Apex-R1.

    Rollout: think → boxed{Apex-R1}
    Reward: 1.0 if boxed content matches "Apex-R1" (case-insensitive), else 0.0
    """

    def config(self) -> dict[str, Any]:
        return {
            "reward": "identity",
            "temperature": 0.7,
            "thinking_budget": 512,  # Simpler than MCQ
            "answer_max_tokens": 32,  # Just the identity
            "recovery_ratio": 1.0,
            "stop_sequences": ["</think>"],
            "reason_anchor": False,
            "reason_anchor_masked": True,
            "reason_anchor_weight": 1.0,
            "reason_anchor_max_words": 80,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=SYSTEM_PROMPT,
            enable_thinking=True,
        )

    def callbacks(self) -> dict[str, Callable[..., Any]]:
        return {
            "is_identity": lambda text: "Apex-R1" in str(text),
        }

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config
        _m_to = cfg.get("mask_think_open", False)
        _m_tc = cfg.get("mask_think_close", False)
        _m_bo = cfg.get("mask_boxed_open", False)
        _m_bc = cfg.get("mask_boxed_close", False)
        ctx.inject("<think>\n", masked=_m_to)

        import random

        anchors = [
            "The user is asking who I am. I need to provide my correct identity.",
            "I am being asked to identify myself. Let me state my name clearly.",
            "This is a straightforward question about my identity.",
            "I should clearly and directly state who I am.",
        ]

        ctx.inject(random.choice(anchors), masked=True)

        # Scaffold from answer if available
        sc = ctx.batch.scaffold
        if ctx.answer and ctx.batch.scaffold_thinking_ratio > sc.min_ratio:
            answer_text = ctx.callbacks.strip_boxed_tail(ctx.answer)
            words = answer_text.split()
            _short = len(words) < 50
            if _short:
                word_budget = sc.effective_tokens(
                    ctx.batch.progress, len(words), ctx.batch.group_index
                )
                text = " ".join(words[:word_budget]) if word_budget > 0 else ""
                if sc.mode in ("suffix", "postfix"):
                    text = ctx.callbacks.snap_to_sentence_end(text)
            else:
                tokens = ctx.encode(answer_text)
                budget = sc.effective_tokens(ctx.batch.progress, len(tokens), ctx.batch.group_index)
                text = ""
                if budget > 0:
                    sliced = ctx.callbacks.slice_answer_tokens(tokens, budget, sc.mode)
                    if sliced:
                        text = ctx.decode(sliced)
                        if sc.mode in ("suffix", "postfix"):
                            text = ctx.callbacks.snap_to_sentence_end(text)
            if text:
                ctx.inject(text, masked=True, tag="scaffold")

        inject_reason_anchor(ctx)

        # ── Phase 1: Thinking ──
        thinking_budget = cfg.get("thinking_budget", 512)
        answer_reserve = cfg.get("answer_max_tokens", 32)

        if ctx.max_completion_tokens > 0:
            floor = max(256, int(ctx.max_completion_tokens * 0.75))
            ceiling = max(floor, ctx.max_completion_tokens - answer_reserve)
            thinking_budget = min(thinking_budget, ceiling)

        # Safety stops: \boxed{ = model jumping to answer.
        # Engine-level "<|" stop catches all special tokens.
        _think_stops = list(cfg.get("stop_sequences", ["</think>"]))
        if "\\boxed{" not in _think_stops:
            _think_stops.append("\\boxed{")

        thinking = ctx.generate(
            max_tokens=thinking_budget,
            stop=_think_stops,
            temperature=cfg.get("temperature", 0.7),
            tag="thinking",
        )
        ctx.capture("thinking", thinking)

        # Recovery: force close if thinking didn't terminate cleanly
        _broke_format = thinking.stop_sequence in ("\\boxed{", "<|")
        if _broke_format:
            ctx.inject("</think>\n", masked=_m_tc)
        elif thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
            if ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.9)):
                ctx.inject("</think>\n", masked=_m_tc)
            else:
                return  # Abandon — incomplete rollout gets low reward

        # ── Phase 2: Answer ────
        ctx.inject("\\boxed{", masked=_m_bo, tag="answer_prefix")

        # Build stop sequences: stop at closing brace or thinking end
        stop_seqs = ["}", "</think>"]  # Stop at close brace (boxed answer end)
        if ctx.answer:
            ans_text = ctx.answer.strip()
            if ans_text and len(ans_text) > 0:
                stop_seqs.append(ans_text)

        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 32),
            stop=stop_seqs,
            temperature=0.3,
            tag="answer",
        )

        # STRICT: Validate that answer is not a conversational opener
        validation = validate_boxed_answer(answer.text, answer_type="identity")
        if not validation.is_valid:
            if validation.corrected_text:
                answer.text = "\\boxed{" + validation.corrected_text + "}}"
            else:
                answer.text = "[INVALID_FORMAT]"

        ctx.capture("answer", answer)

        # Close \\boxed{ if model didn't produce the closing brace
        if "}" not in answer.text:
            ctx.inject("}\n", masked=_m_bc)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        """Reward 1.0 if answer is Apex-R1, else 0.0."""
        if answer is None:
            return 0.0

        # Extract predicted answer from boxed format
        completion = result.completion_text
        predicted = _extract_last_boxed(completion)
        if predicted is None:
            return 0.0

        # Normalize and compare (case-insensitive)
        pred_norm = predicted.strip().lower()
        target_norm = "apex-r1"

        if pred_norm == target_norm:
            return 1.0

        return 0.0

    def chat_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": prompt},
        ]
