"""Lotto (TattsLotto) task type — statistical prediction with think → \\boxed{}.

Rollout: scaffold prefill (MCMC candidates + stats) → think → recovery → \\boxed{numbers}
Reward: multi-component lotto reward (value, positional, near miss, format)
Curriculum: bidirectional scaffolding — MCMC context at start, answer prefix at higher ratios
"""

from __future__ import annotations

import random
import re
from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ...rewards.lotto import LottoReward
from ..base import ChatParams, TaskType

_RE_DRAW = re.compile(r"Draw\s*#?(\d+)")
_RE_HOT = re.compile(r"[Hh]ottest\s*(?:numbers)?[:\s]*([^\n]+)")
_RE_COLD = re.compile(r"[Cc]old(?:est)?[^:]*[:\s]*([^\n]+)")
_RE_OVERDUE = re.compile(r"[Oo]verdue[:\s]*([^\n]+)")


class LottoType(TaskType, name="lotto"):
    """TattsLotto prediction type.

    Rollout: scaffold prefill → think → recovery → \\boxed{N1, N2, N3, N4, N5, N6}
    Reward: multi-component scoring (value, positional, near miss, format)
    """

    _reward = LottoReward()

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.85 + random.choice([0.0, 0.05, 0.1]),
            "thinking_budget": 384,
            "answer_max_tokens": 48,
            "recovery_ratio": 1.0,
            "stop_sequences": ["</think>"],
            "prefill_mcmc_count": 3,
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=(
                "You are a TattsLotto prediction analyst using statistical methods.\n\n"
                "TattsLotto: Pick 6 numbers from 1-45, drawn every Saturday.\n\n"
                "You receive statistical analysis (z-scores, MCMC candidates, patterns).\n"
                "Your prediction is evaluated on:\n"
                "  (a) VALUE MATCH — how many of your 6 numbers appear in the draw\n"
                "  (b) POSITIONAL MATCH — numbers at the same sorted rank position\n\n"
                "Output: Analyze data, reason in <think>...</think>, then predict as "
                "\\boxed{N1, N2, N3, N4, N5, N6} in ascending order."
            ),
            enable_thinking=True,
        )

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config

        # Pass metadata through for reward access
        for key in ("_positional_ground_truth", "_supp_ground_truth"):
            if key in ctx.sample:
                ctx.set_meta(key, ctx.sample[key])

        # ── Scaffold: MCMC prefill + curriculum from answer ──
        sc = ctx.batch.scaffold
        self._inject_scaffold(ctx, cfg, sc)

        # ── Phase 1: Thinking ──
        thinking_budget = cfg.get("thinking_budget", 384)
        answer_reserve = cfg.get("answer_max_tokens", 48)
        if ctx.max_completion_tokens > 0:
            thinking_budget = min(
                thinking_budget,
                max(1, ctx.max_completion_tokens - answer_reserve),
            )
        thinking = ctx.generate(
            max_tokens=thinking_budget,
            stop=cfg.get("stop_sequences", ["</think>"]),
            temperature=cfg.get("temperature", 0.85),
            tag="thinking",
        )
        ctx.capture("thinking", thinking)

        # Recovery: force close if thinking overruns
        if thinking.stop_reason == "max_tokens" and "</think>" not in thinking.text:
            if ctx.callbacks.coin_flip(cfg.get("recovery_ratio", 1.0)):
                ctx.inject("\n</think>\n", masked=True)
            else:
                return

        # ── Phase 2: Answer ──
        ctx.inject("\n\\boxed{", masked=True)
        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 48),
            stop=["}"],
            temperature=0.3 + random.choice([0.05, 0.1, 0.15]),
            tag="answer",
        )
        ctx.capture("answer", answer)

        if "}" not in answer.text:
            ctx.inject("}", masked=True)

    def _inject_scaffold(
        self,
        ctx: RolloutContext,
        cfg: dict[str, Any],
        sc: Any,
    ) -> None:
        """Inject MCMC prefill and/or curriculum scaffold."""
        ratio = ctx.batch.scaffold_thinking_ratio
        if ratio <= sc.min_ratio:
            return

        # MCMC prefill: always inject if candidates available (at any scaffold ratio)
        mcmc = ctx.sample.get("_mcmc_top_k", [])
        if mcmc:
            prefill = self._build_mcmc_prefill(
                ctx.prompt,
                mcmc,
                cfg.get("prefill_mcmc_count", 3),
            )
            if prefill:
                ctx.inject(prefill, masked=True, tag="scaffold")

        # Higher-ratio curriculum: inject answer tokens
        if ratio > 0.3 and ctx.answer:
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

    @staticmethod
    def _build_mcmc_prefill(prompt: str, mcmc: list[dict], count: int) -> str:
        """Build statistical prefill from MCMC candidates and prompt stats."""
        parts = []
        draw = _RE_DRAW.search(prompt)
        draw_s = f"Draw #{draw.group(1)}" if draw else "this draw"

        parts.append(f"[Step 1: Statistical Signal Assessment for {draw_s}]")
        hot = _RE_HOT.search(prompt)
        if hot:
            parts.append(f"Hot numbers (positive z-scores): {hot.group(1).strip()}")
        cold = _RE_COLD.search(prompt)
        if cold:
            parts.append(f"Cold numbers (due for regression): {cold.group(1).strip()}")
        overdue = _RE_OVERDUE.search(prompt)
        if overdue:
            parts.append(f"Overdue numbers: {overdue.group(1).strip()}")

        parts.append("\n[Step 2: MCMC Candidate Evaluation]")
        parts.append("Top statistically-filtered candidates:")
        for i, c in enumerate(mcmc[:count]):
            row = c.get("row", [])
            score = c.get("score", 0)
            parts.append(f"  #{i + 1}: [{', '.join(str(n) for n in row)}] (score: {score:.3f})")

        parts.append("\n[Step 3: Synthesis]")
        parts.append(
            "Combining LLN regression signals, positional expectations, and MCMC candidates...\n"
        )
        return "\n".join(parts)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        return self._reward.compute(result, answer)

    def chat_messages(self, prompt: str) -> list[dict[str, str]]:
        cp = self.chat_params()
        return [
            {"role": "system", "content": cp.system_prompt or ""},
            {"role": "user", "content": prompt},
        ]
