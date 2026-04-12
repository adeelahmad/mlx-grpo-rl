"""MCQ (Multiple Choice Question) — the built-in reference type.

Demonstrates the full TaskType pattern:
- Rollout with thinking + answer extraction
- Curriculum scaffolding via batch.scaffold_ratio
- Recovery on incomplete thinking
- Custom callbacks for MCQ-specific validation

This single file is the complete implementation. No other files needed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ...rewards.exam import ExamReward
from ..base import ChatParams, TaskType
from ._anchor import inject_reason_anchor
from .answer_format_validator import validate_boxed_answer


def remove_unmatched_closing_brace(s: str) -> str:
    balance = 0
    for ch in s:
        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1

    # If balance is negative, there's an extra unmatched }
    if balance < 0:
        # Remove only the LAST }
        idx = s.rfind("}")
        if idx != -1:
            s = s[:idx] + s[idx + 1 :]
    return s


def _target_thinking(ctx: RolloutContext) -> str | None:
    """Extract expert thinking text from the dataset sample for masked injection.

    Tries ctx.sample["thinking"] / ["target_thinking"] first.
    Falls back to parsing <think>...</think> from ["target_completion"] or ctx.answer.
    ctx.answer is the final fallback because datasets often store the full response
    (including <think>...</think>) in the answer field, which the data loader extracts
    from metadata into sample.answer — not accessible via ctx.sample.
    """
    raw = ctx.sample.get("thinking") or ctx.sample.get("target_thinking")
    if raw:
        return str(raw).strip()
    for src in (
        ctx.sample.get("target_completion"),
        ctx.sample.get("completion"),
        ctx.answer,
    ):
        if src:
            m = re.search(r"<think>(.*?)</think>", str(src), re.DOTALL)
            if m:
                return m.group(1).strip()
    return None


SYSTEM_PROMPT = (
    "You are a knowledgeable exam assistant NeuralAI. "
    "Read the question and all answer choices carefully. "
    "Think through the problem step by step in your thinking block, "
    "then select the single best answer letter (A, B, C, or D).\n\n"
    "Always base your answer on factual knowledge and sound reasoning. "
    "If you are uncertain, reason through elimination of clearly wrong options."
)


class MCQType(TaskType, name="mcq"):
    """Multiple choice question type.

    Rollout: think → extract answer letter (A/B/C/D)
    Reward: composite exam reward (accuracy + format + gaming detection)
    """

    _reward = ExamReward()

    def config(self) -> dict[str, Any]:
        return {
            "reward": "exam",
            "temperature": 0.7,
            "thinking_budget": 768,  # was 512 — some MCQs need reasoning
            "answer_max_tokens": 8,  # just "A}" inside \boxed{}
            "recovery_ratio": 1.0,
            "stop_sequences": ["</think>"],
            "reason_anchor": False,
            "reason_anchor_masked": True,
            "reason_anchor_weight": 1.0,
            "reason_anchor_max_words": 80,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            # system_prompt=SYSTEM_PROMPT,
            enable_thinking=True,
        )

    def callbacks(self) -> dict[str, Callable[..., Any]]:
        return {
            "is_valid_mcq": lambda text: bool(__import__("re").search(r"\(?[A-D]\)?", text)),
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

        # ── Curriculum: scaffold from answer, decaying over training ──
        # For short answers (< 50 words), use word-level units — token slicing
        # on short content produces mid-word cuts that confuse the model.
        sc = ctx.batch.scaffold
        if ctx.answer and ctx.batch.scaffold_thinking_ratio > sc.min_ratio:
            answer_text = ctx.callbacks.strip_boxed_tail(ctx.answer)
            words = answer_text.split()
            _short = len(words) < 50
            if _short:
                # Word-level scaffold: budget as word count, not token count
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

        # ── Phase 1: Thinking ──────────────────────────────────────────
        thinking_budget = cfg.get("thinking_budget", 768)
        answer_reserve = cfg.get("answer_max_tokens", 32)

        if ctx.max_completion_tokens > 0:
            # Guarantee thinking gets at least 75% of total budget, floor 256
            floor = max(256, int(ctx.max_completion_tokens * 0.75))
            ceiling = max(floor, ctx.max_completion_tokens - answer_reserve)
            thinking_budget = min(thinking_budget, ceiling)

        # Safety stops: \boxed{ = model jumping to answer.
        # Engine-level "<|" stop catches all special tokens (<|im_end|>, <|im_start|>, etc.)
        _think_stops = list(cfg.get("stop_sequences", ["</think>"]))
        if "\\boxed{" not in _think_stops:
            _think_stops.append("\\boxed{")

        thinking = ctx.generate(
            max_tokens=thinking_budget,
            stop=_think_stops,
            temperature=cfg.get("temperature", 1.1),
            tag="thinking",
        )
        ctx.capture("thinking", thinking)

        # Recovery: force close if thinking didn't terminate cleanly.
        # Guard: only inject if </think> isn't already in the text — the
        # backend's post-loop stop check trims it in most cases, but this
        # is a belt-and-suspenders guard for edge cases.
        _tgt_think = _target_thinking(ctx) if cfg.get("inject_target_thinking", False) else None
        _broke_format = thinking.stop_sequence in ("\\boxed{", "<|")
        if _broke_format:
            # Model broke out of thinking — force close and proceed to answer
            ctx.inject("</think>\n", masked=_m_tc)
        elif thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
            # Target thinking available → always close and inject; never abandon.
            if _tgt_think or ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 0.9)):
                ctx.inject("</think>\n", masked=_m_tc)
            else:
                return  # Abandon — incomplete rollout gets low reward

        # Inject complete expert thinking as masked context before the answer.
        # Placed after </think> closes (whether model-generated or force-injected).
        if _tgt_think:
            ctx.inject("\n" + _tgt_think + "\n", masked=True, tag="target_thinking")

        # ── Phase 2: Answer ────────────────────────────────────────────
        # Use \boxed{} format — consistent with target_completion and math type.
        # Model generates "A}" to complete \boxed{A}; reward extracts via boxed.
        ctx.inject("\n\\boxed{", masked=_m_bo, tag="answer_prefix")

        # Build stop sequences: stop at closing brace or thinking end
        stop_seqs = ["}", "</think>"]  # Stop at close brace (boxed answer end)
        if ctx.answer:
            ans_text = ctx.answer.strip()
            if ans_text and len(ans_text) > 0:
                stop_seqs.append(ans_text)

        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 3),
            stop=stop_seqs,
            temperature=1.1,
            tag="answer",
        )
        # STRICT: Validate that answer is not a conversational opener like "Hi! I'm"
        validation = validate_boxed_answer(answer.text, answer_type="mcq")
        if not validation.is_valid:
            # Model boxed conversational opener instead of just the letter
            # Log this and apply correction if available
            if validation.corrected_text:
                # Inject corrected answer
                answer.text = "\\boxed{" + validation.corrected_text + "}}"
            else:
                # No correction available, mark for loss penalty
                answer.text = "[INVALID_FORMAT]"

        ctx.capture("answer", answer)

        # Close \boxed{ if model didn't produce the closing brace
        if "}" not in answer.text:
            ctx.inject("}\n", masked=_m_bc)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        return self._reward.compute(result, answer)

    def chat_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": prompt},
        ]


class ExamType(MCQType, name="exam"):
    """Exam type — identical to MCQ rollout + ExamReward.

    Registered separately so datasets with type_name='exam' resolve correctly.
    """
