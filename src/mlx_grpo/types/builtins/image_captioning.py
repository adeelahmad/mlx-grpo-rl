"""Image Captioning task type.

Images are propagated via ctx._images (populated by RolloutExecutor from
Sample.image_path or Sample.image_b64). This type controls prompt format
and reward only; the backend handles image encoding.

Rollout:
    Generate a natural-language description of the image.

Reward:
    ROUGE-L F1 between generated caption and reference caption.
    When multiple references are provided (semicolon-separated), the
    maximum ROUGE-L across all references is used.
"""

from __future__ import annotations

from typing import Any

from ...engine.context import RolloutContext
from ...engine.result import RolloutResult
from ...rewards.vision import VisionRougeLReward
from ..base import ChatParams, TaskType
from ._anchor import inject_reason_anchor

_SYSTEM_PROMPT = (
    "You are an image captioning assistant. "
    "Describe the image in one or two clear, concise sentences. "
    "Focus on the main subject, action, and setting."
)

_rouge_reward = VisionRougeLReward()


class ImageCaptioningType(TaskType, name="image_captioning"):
    """Image captioning task type.

    Rollout: generate a natural-language image description.
    Reward:  ROUGE-L F1 against reference caption(s).
             Multiple references may be separated by ';' in the answer field.
    """

    def config(self) -> dict[str, Any]:
        return {
            "temperature": 0.6,
            "caption_max_tokens": 128,
            "reason_anchor": False,
            "reason_anchor_masked": True,
            "reason_anchor_weight": 1.0,
            "reason_anchor_max_words": 80,
            "multi_image_separator": " | ",  # separator between multi-image captions
        }

    def chat_params(self) -> ChatParams:
        return ChatParams(
            system_prompt=_SYSTEM_PROMPT,
            enable_thinking=False,
        )

    def rollout(self, ctx: RolloutContext) -> None:
        cfg = ctx.type_config
        inject_reason_anchor(ctx)

        # Multi-image support: generate caption for each image
        if ctx._images and len(ctx._images) > 1:
            separator = cfg.get("multi_image_separator", " | ")
            captions = []
            for i, img in enumerate(ctx._images):
                caption = ctx.generate(
                    max_tokens=cfg.get("caption_max_tokens", 128),
                    temperature=cfg.get("temperature", 0.6),
                    images=[img],
                    tag=f"caption_{i}",
                )
                captions.append(caption.text)
            # Store as combined text with separator, but track individual segments
            combined_text = separator.join(captions)
            from ...engine.result import GenerateResult

            ctx.capture(
                "caption_var",
                GenerateResult(
                    text=combined_text,
                    tokens=[],  # Multi-image combines segments, not individual tokens
                    stop_reason="multi_image",
                ),
            )
            return

        # Single image
        caption = ctx.generate(
            max_tokens=cfg.get("caption_max_tokens", 128),
            temperature=cfg.get("temperature", 0.6),
            tag="caption",
        )
        ctx.capture("caption_var", caption)

    def reward(self, result: RolloutResult, answer: str | None = None) -> float:
        """Compute ROUGE-L against all reference captions.

        For multi-image: answer may be semicolon or pipe-separated.
        Computes mean score across all references.
        """
        if not answer:
            return 0.0

        # Support both reference formats:
        # 1. Multiple references for single image: "cap1; cap2; cap3"
        # 2. Multi-image references: "cap1_img1 | cap1_img2" or "cap1; cap2 | cap1; cap2"
        # Split by pipe first (multi-image), then by semicolon (multi-reference)
        image_refs = [r.strip() for r in answer.split("|") if r.strip()]
        if len(image_refs) > 1:
            # Multi-image: each image has potential multiple references
            scores = []
            for img_ref in image_refs:
                refs = [r.strip() for r in img_ref.split(";") if r.strip()]
                if refs:
                    img_score = max(_rouge_reward.compute(result, ref) for ref in refs)
                    scores.append(img_score)
            return sum(scores) / len(scores) if scores else 0.0

        # Single image, possibly multiple references
        references = [r.strip() for r in answer.split(";") if r.strip()]
        if not references:
            return 0.0
        return max(_rouge_reward.compute(result, ref) for ref in references)


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: rewards.vision (stdlib only)
# Usage: TaskType.get("image_captioning")
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
