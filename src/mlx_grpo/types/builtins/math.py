"""Math task type — think → \\boxed{answer}.

Rollout: generate thinking, then answer with \\boxed{} format
Reward: exact match on boxed content (normalized numerically)
Curriculum: scaffold from answer, decaying over training
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ..base import ChatParams, TaskType
from ._anchor import inject_reason_anchor
from .answer_format_validator import validate_boxed_answer
from .math_helpers import (
    _compute_thinking_budget,
    _detect_token_repetition_hack,
    _extract_boxed,
    _extract_last_boxed,
    _normalize_math_answer,
    _validate_boxed_answers,
    _validate_think_tags,
)


class MathType(TaskType, name="math"):
    """Mathematical reasoning type."""

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.7,
            "thinking_budget": 2048,
            "answer_max_tokens": 128,
            "scaffold_decay": True,
            "recovery_ratio": 1.0,
            "stop_sequences": ["</think>"],
            "reward": "math",
            "recovery_bonus_tokens": 128,
            "thinking_budget_ceiling": 4096,
            "reason_anchor": False,
            "reason_anchor_masked": True,
            "reason_anchor_weight": 1.0,
            "reason_anchor_max_words": 80,
            "post_boxed_tokens": 0,
            "post_boxed_reward_threshold": 0.8,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            # system_prompt=(
            #     "You are a mathematical problem solver. Think through the problem "
            #     "step by step, then provide your final answer in \\boxed{} notation."
            # ),
            enable_thinking=True,
        )

    def callbacks(self) -> dict[str, Callable[..., Any]]:
        return {
            "normalize_math": _normalize_math_answer,
            "has_boxed": lambda text: "\\boxed{" in str(text),
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
            "Okay, I see what they need — let me think through this "
            "cleanly and just give them the answer without rambling.",
            "Right, this is straightforward if I don't overthink it — solve it, say it, done.",
            "Let me not waste my own time here — I know what's being "
            "asked, I'll work through it efficiently and keep my response "
            "tight.",
            "Alright, I'm going to think this through step by step but "
            "only the steps that actually matter — no repeating myself.",
            "Okay let me just focus — what's the actual problem, what's "
            "the shortest path to correct, and what does the response "
            "need to contain. Nothing else.",
            "I already get what they want — I'm not going to echo it "
            "back to myself, I'll just solve it and write a clean answer.",
            "Let me catch myself before I spiral — this needs focused "
            "reasoning and a direct response, not a essay about how I'm "
            "going to reason.",
            "Alright, thinking cap on — but the efficient kind. Every "
            "thought here should push toward the answer, not circle "
            "around it.",
            "Okay I know what this is — let me reason through it "
            "quickly, check my work once, and give them exactly what "
            "they asked for.",
            "Let me be honest with myself about what's actually hard "
            "here and skip the parts that aren't — then answer concisely.",
            "Right — I'll think at the depth this actually requires and "
            "not a layer deeper. Then a clean output.",
            "Okay, no need to narrate my own understanding back to "
            "myself — I'll just work the problem and deliver.",
            "Let me stay tight here — think through the real logic, skip "
            "the filler thoughts, and land on a response that respects "
            "their time.",
            "Alright, I'm going to resist the urge to over-explain in "
            "my own head — just solve, verify, output.",
            "Okay this is clear enough — let me do the actual thinking "
            "instead of thinking about thinking.",
            "Let me cut to it — what's the core of this, what's the "
            "answer, and what's the cleanest way to say it.",
            "Right, I'll work through this properly but quickly — no "
            "restating the question, no hedging, just the goods.",
            "Okay — I understand the ask. Now I'm going to reason only "
            "where reasoning is needed and then give a sharp answer.",
            "Let me not be that guy who repeats the question back — I'll "
            "just solve it and write something actually useful.",
            "Alright, focusing now — minimal thinking overhead, maximum answer quality. Let's go.",
        ]

        ctx.inject(random.choice(anchors), masked=True)

        sc = ctx.batch.scaffold
        if (
            cfg.get("scaffold_decay", True)
            and ctx.answer
            and ctx.batch.scaffold_thinking_ratio > sc.min_ratio
        ):
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

        is_scaffolded = ctx.batch.scaffold.mode != "none"
        thinking_budget, bonus_budget = _compute_thinking_budget(
            cfg,
            ctx.answer or "",
            ctx.prompt or "",
            ctx.max_completion_tokens,
            is_scaffolded,
        )

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
        # STRICT: Enforce single <think>...</think> block, prevent reopening
        thinking.text = _validate_think_tags(thinking.text)
        ctx.capture("thinking", thinking)

        _broke_format = thinking.stop_sequence in ("\\boxed{", "<|")
        if _broke_format:
            ctx.inject("</think>\n", masked=_m_tc)
        elif thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
            if ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.8)):
                if is_scaffolded and bonus_budget > 0:
                    bonus = ctx.generate(
                        max_tokens=bonus_budget,
                        stop=cfg.get("stop_sequences", ["</think>"]),
                        temperature=0.3,
                        tag="recovery_thinking",
                        masked=True,
                    )
                    if "</think>" not in bonus.text:
                        ctx.inject("</think>\n", masked=_m_tc)
                else:
                    ctx.inject("</think>\n", masked=_m_tc)
            else:
                return  # Abandon

        ctx.inject("\n\\boxed{", masked=_m_bo, tag="answer_prefix")

        # Build stop sequences: stop at closing brace or thinking end
        stop_seqs = ["}", "</think>"]  # Stop at close brace (boxed answer end)
        if ctx.answer:
            # Extract plain answer text (without boxed wrapper)
            ans_text = _extract_boxed(ctx.answer) or ctx.answer.strip()
            if ans_text and len(ans_text) > 1:  # Only add if meaningful
                stop_seqs.append(ans_text)

        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 128),
            stop=stop_seqs,
            temperature=0.3,
            tag="answer",
        )
        # STRICT: Keep only first boxed answer, drop hacking attempts
        answer.text = _validate_boxed_answers(answer.text)
        # Check for obvious repetition hacks (word spam to game token budgets)
        if _detect_token_repetition_hack(answer.text):
            # Model tried to spam tokens — sanitize by taking only first meaningful part
            words = answer.text.split()
            answer.text = " ".join(words[:128])  # Limit to reasonable length
        # Validate format: ensure boxed content is answer, not conversational opener
        format_validation = validate_boxed_answer(answer.text, answer_type="math")
        if not format_validation.is_valid and format_validation.corrected_text:
            answer.text = "\\boxed{" + format_validation.corrected_text + "}"
        ctx.capture("answer", answer)

        depth = sum(1 if c == "{" else -1 if c == "}" else 0 for c in answer.text)
        if depth >= 0:
            ctx.inject("}" * (depth + 1), masked=_m_bc)

        # Post-boxed forced continuation: if answer is above threshold, force more tokens
        _pbt = cfg.get("post_boxed_tokens", 0)
        if _pbt > 0 and ctx.answer:
            _pred = _extract_last_boxed("\\boxed{" + answer.text)
            _gt = _extract_boxed(ctx.answer) or ctx.answer.strip()
            if _pred is not None:
                _pred_n = _normalize_math_answer(_pred)
                _gt_n = _normalize_math_answer(_gt)
                if _gt_n != 0:
                    try:
                        _rel_err = abs(float(_pred_n) - float(_gt_n)) / abs(float(_gt_n))
                        _inline_r = (
                            1.0
                            if _pred_n == _gt_n
                            else (0.8 if _rel_err < 0.01 else 0.5 if _rel_err < 0.05 else 0.0)
                        )
                    except (ValueError, ZeroDivisionError):
                        _inline_r = 1.0 if _pred_n == _gt_n else 0.0
                else:
                    _inline_r = 1.0 if _pred_n == _gt_n else 0.0
            else:
                _inline_r = 0.0
            if _inline_r >= cfg.get("post_boxed_reward_threshold", 0.8):
                ctx.max_completion_tokens += _pbt
                ctx.generate(max_tokens=_pbt, tag="post_boxed", masked=False)

    def escalation_rollout(self, ctx: RolloutContext) -> None:
        """Adaptive escalation attempt — prefix injected by executor via pre_inject_text."""
        cfg = ctx.type_config
        _m_tc = cfg.get("mask_think_close", False)
        _m_bo = cfg.get("mask_boxed_open", False)
        _m_bc = cfg.get("mask_boxed_close", False)

        postfix_text = ctx.batch.custom.get("postfix_text")
        if postfix_text:
            ctx.inject(postfix_text, masked=True, tag="postfix_scaffold")

        inject_reason_anchor(ctx)

        thinking_budget, _ = _compute_thinking_budget(
            cfg,
            ctx.answer or "",
            ctx.prompt or "",
            ctx.max_completion_tokens,
            False,
        )
        thinking = ctx.generate(
            max_tokens=thinking_budget,
            stop=cfg.get("stop_sequences", ["</think>"]),
            temperature=cfg.get("temperature", 0.7),
            tag="thinking",
        )
        # STRICT: Enforce single <think>...</think> block, prevent reopening
        thinking.text = _validate_think_tags(thinking.text)
        ctx.capture("thinking", thinking)
        ctx.set_meta("thinking_stop_reason", thinking.stop_reason)

        if thinking.stop_reason != "stop_sequence":
            # Thinking hit max_tokens — apply recovery if configured so this
            # attempt still generates an answer and contributes reward signal.
            if not ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.8)):
                return  # Abandon this attempt
            ctx.inject("</think>\n", masked=_m_tc, tag="recovery")

        ctx.inject("\n\\boxed{", masked=_m_bo, tag="answer_prefix")

        # Build stop sequences: stop at closing brace or thinking end
        stop_seqs = ["}", "</think>"]  # Stop at close brace (boxed answer end)
        if ctx.answer:
            # Extract plain answer text (without boxed wrapper)
            ans_text = _extract_boxed(ctx.answer) or ctx.answer.strip()
            if ans_text and len(ans_text) > 1:  # Only add if meaningful
                stop_seqs.append(ans_text)

        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 128),
            stop=stop_seqs,
            temperature=0.3,
            tag="answer",
        )
        # STRICT: Keep only first boxed answer, drop hacking attempts
        answer.text = _validate_boxed_answers(answer.text)
        # Check for obvious repetition hacks (word spam to game token budgets)
        if _detect_token_repetition_hack(answer.text):
            words = answer.text.split()
            answer.text = " ".join(words[:128])
        # Validate format: ensure boxed content is answer, not conversational opener
        format_validation = validate_boxed_answer(answer.text, answer_type="math")
        if not format_validation.is_valid and format_validation.corrected_text:
            answer.text = "\\boxed{" + format_validation.corrected_text + "}"
        ctx.capture("answer", answer)
        depth = sum(1 if c == "{" else -1 if c == "}" else 0 for c in answer.text)
        if depth >= 0:
            ctx.inject("}" * (depth + 1), masked=_m_bc)

        # Post-boxed forced continuation: if answer is above threshold, force more tokens
        _pbt = cfg.get("post_boxed_tokens", 0)
        if _pbt > 0 and ctx.answer:
            _pred = _extract_last_boxed("\\boxed{" + answer.text)
            _gt = _extract_boxed(ctx.answer) or ctx.answer.strip()
            if _pred is not None:
                _pred_n = _normalize_math_answer(_pred)
                _gt_n = _normalize_math_answer(_gt)
                if _gt_n != 0:
                    try:
                        _rel_err = abs(float(_pred_n) - float(_gt_n)) / abs(float(_gt_n))
                        _inline_r = (
                            1.0
                            if _pred_n == _gt_n
                            else (0.8 if _rel_err < 0.01 else 0.5 if _rel_err < 0.05 else 0.0)
                        )
                    except (ValueError, ZeroDivisionError):
                        _inline_r = 1.0 if _pred_n == _gt_n else 0.0
                else:
                    _inline_r = 1.0 if _pred_n == _gt_n else 0.0
            else:
                _inline_r = 0.0
            if _inline_r >= cfg.get("post_boxed_reward_threshold", 0.8):
                ctx.max_completion_tokens += _pbt
                ctx.generate(max_tokens=_pbt, tag="post_boxed", masked=False)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        if answer is None:
            return 0.0

        # Extract predicted answer from the LAST boxed (injected prefix may
        # contain intermediate \boxed{} answers from the reference trace).
        completion = result.completion_text
        predicted = _extract_last_boxed(completion)
        if predicted is None:
            return 0.0

        # Extract ground truth (reference answer has no intermediate boxes)
        gt = _extract_boxed(answer)
        if gt is None:
            gt = answer.strip()

        # Normalize and compare
        pred_norm = _normalize_math_answer(predicted)
        gt_norm = _normalize_math_answer(gt)

        if pred_norm == gt_norm:
            return 1.0

        try:
            pred_val = float(pred_norm)
            gt_val = float(gt_norm)
            if gt_val != 0:
                rel_error = abs(pred_val - gt_val) / abs(gt_val)
                if rel_error < 0.01:
                    return 0.8
                elif rel_error < 0.05:
                    return 0.5
        except (ValueError, ZeroDivisionError):
            pass  # Not numeric — fall through to return 0.0

        return 0.0
