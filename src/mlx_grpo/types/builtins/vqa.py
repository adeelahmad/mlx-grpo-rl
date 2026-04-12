"""VQA (Visual Question Answering) task type.

Images are propagated via ctx._images (populated by RolloutExecutor from
Sample.image_path or Sample.image_b64). The backend receives the image
through GenerateParams.images; this type only controls prompt format and reward.

Rollout:
    Generate a concise answer to an image-grounded question.
    Thinking is disabled by default (VQA answers are short and factual).

Reward:
    Token-level F1 between predicted answer and reference answer.
    Falls back to exact-match when reference is a single word.
"""

from __future__ import annotations

from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ...rewards.vision import VisionExactMatchReward, VisionTokenF1Reward
from ..base import ChatParams, TaskType
from ._anchor import inject_reason_anchor

_SYSTEM_PROMPT = (
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question concisely. "
    "Give a short, direct answer — typically one to a few words."
)

_f1_reward = VisionTokenF1Reward()
_exact_reward = VisionExactMatchReward()


class VQAType(TaskType, name="vqa"):
    """Visual Question Answering task type.

    Rollout: generate short factual answer to an image question.
    Reward:  token F1 against reference (exact match for single-token refs).
    """

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.3,
            "answer_max_tokens": 64,
            "reason_anchor": False,
            "reason_anchor_masked": True,
            "reason_anchor_weight": 1.0,
            "reason_anchor_max_words": 80,
            "multi_image_mode": "per_image",  # "per_image" or "unified"
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=_SYSTEM_PROMPT,
            enable_thinking=False,
        )

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config
        inject_reason_anchor(ctx)

        # Multi-image support: if multiple images, generate per image
        if ctx._images and len(ctx._images) > 1:
            mode = cfg.get("multi_image_mode", "per_image")
            if mode == "per_image":
                # Generate separate answer for each image
                answers = []
                for i, img in enumerate(ctx._images):
                    answer = ctx.generate(
                        max_tokens=cfg.get("answer_max_tokens", 64),
                        temperature=cfg.get("temperature", 0.3),
                        images=[img],
                        tag=f"answer_{i}",
                    )
                    answers.append(answer)
                ctx.capture("answer_var", answers)
                return

        # Single image (or per_image mode with 1 image)
        answer = ctx.generate(
            max_tokens=cfg.get("answer_max_tokens", 64),
            temperature=cfg.get("temperature", 0.3),
            tag="answer",
        )
        ctx.capture("answer_var", answer)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        if not answer:
            return 0.0

        # Handle multi-image: answer may be semicolon-separated answers
        # or computed from multiple segments. Take mean of per-image rewards.
        answers = [a.strip() for a in answer.split(";") if a.strip()]
        if len(answers) > 1:
            scores = []
            for ans in answers:
                ref_tokens = ans.split()
                if len(ref_tokens) <= 1:
                    score = _exact_reward.compute(result, ans)
                else:
                    score = _f1_reward.compute(result, ans)
                scores.append(score)
            return sum(scores) / len(scores)

        # Single image: use original logic
        ref_tokens = answer.strip().split()
        if len(ref_tokens) <= 1:
            return _exact_reward.compute(result, answer)
        return _f1_reward.compute(result, answer)


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: rewards.vision (stdlib only)
# Usage: TaskType.get("vqa")
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
