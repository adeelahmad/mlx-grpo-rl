"""ICL-Guided Dynamic Escalation Engine — vertical + horizontal scaffold loops.

Vertical dimension: interleaved scaffold_inject → two-phase model_generate cycles.
Horizontal dimension: escalating scaffold depth across group members with inline reward.
Reward shaping: bidirectional group-level shaping after all members complete.

Planning helpers and weight building live in escalation_plan.py to respect 200-line limit.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

from ..engine.context import BatchContext, RolloutContext
from ..engine.result import RolloutResult, SegmentSource
from ..scaffold import ScaffoldConfig
from ..types.builtins.mcq import _target_thinking
from .escalation_plan import (
    _anchor_text_for_scaffold,
    _answer_prefix_text,
    _apply_group_shaping,
    _build_tri_factor_weights,
    _count_n_eff_max,
    _extract_forced_answer,
    _jaccard_anchor_from_naked,
    _plan_adaptive_scaffold,
    _two_phase_generate,
)
from .intervention import GroupMemberRecord, build_group_progress, call_intervention

if TYPE_CHECKING:
    from ..config.escalation import EscalationConfig, VerticalConfig
    from ..data import Sample
    from ..rewards.base import BaseReward
    from ..types.base import TaskType


# ── GPU cooldown ──────────────────────────────────────────────────────────────


def _sync_gpu() -> None:
    """Sync Metal GPU and clear cache. Call INSIDE _ROLLOUT_LOCK."""
    try:
        import mlx.core as mx

        mx.synchronize()
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def _sync_and_cool(cooldown: float) -> None:
    """Serial path: sync GPU then sleep. Use _sync_gpu + sleep separately for parallel."""
    _sync_gpu()
    if cooldown > 0:
        time.sleep(cooldown)


# ── Vertical rollout ──────────────────────────────────────────────────────────


def _vertical_rollout(ctx: RolloutContext) -> None:
    """Interleaved scaffold+generation rollout — the vertical dimension.

    Reads from ctx.batch.custom:
      "segments"            : list[str]  — scaffold chunks from _plan_adaptive_scaffold
      "vert_cfg"            : VerticalConfig | None
      "N_eff"               : int        — effective cycle count
      "B_eff"               : int        — Phase A token budget per cycle (0 = fallback)
      "b_ext"               : int        — Phase B extension budget (0 = disabled)
      "answer_prefix_text"  : str        — first N tok of GT answer injected pre-generation
      "answer_prefix_toks"  : int        — token count of answer_prefix_text (for budget)

    Pattern: [scaffold_i] → Phase_A_i → [Phase_B_i] → … → </think> → \\boxed{answer}
    """
    custom = ctx.batch.custom
    segments: list[str] = custom.get("segments", [])
    vert: VerticalConfig | None = custom.get("vert_cfg")
    b_ext: int = custom.get("b_ext", 0)
    seg_cb = custom.get("segment_cb")  # Optional[(source, text, tag) -> None]
    cfg = ctx.type_config
    _m_to = cfg.get("mask_think_open", False)
    _m_tc = cfg.get("mask_think_close", False)
    _m_bo = cfg.get("mask_boxed_open", False)
    _m_bc = cfg.get("mask_boxed_close", False)

    ctx.inject("<think>\n", masked=_m_to)

    answer_reserve: int = cfg.get("answer_max_tokens", 32)
    temp: float = cfg.get("temperature", 0.7)
    closed_think = False

    # When forced_answer will fire: ignore thinking_budget + per-cycle caps.
    # The answer is guaranteed correct (reward=1.0), so let the model think freely.
    _is_forced = bool(custom.get("forced_answer_text", ""))
    thinking_budget: int = (
        max(cfg.get("thinking_budget", 512), max(0, ctx.max_completion_tokens - answer_reserve))
        if _is_forced
        else cfg.get("thinking_budget", 512)
    )

    if segments:
        # Phase A budget per cycle: divide thinking_budget evenly across scaffold chunks.
        # Do NOT use b_eff from the planner — it is based on expert trace token count,
        # which is unrelated to the model's actual generation budget.
        eff_b = max(32, thinking_budget // max(len(segments), 1))
        # Optional per-cycle cap — set max_cycle_gen_tokens to enforce N:1 gen:scaffold
        # ratio per cycle (e.g. 192 for 3:1 with scaffold.max_tokens=64).
        # Forced-answer members bypass per-cycle cap — unconstrained thinking is the goal.
        _vert_mcg = vert.max_cycle_gen_tokens if vert and vert.max_cycle_gen_tokens > 0 else 0
        _mcg = 0 if _is_forced else _vert_mcg
        if _mcg > 0:
            eff_b = min(eff_b, _mcg)
        # Jaccard-based strategy anchor — inject before scaffold cycles when present.
        # Built from the intersection of the model's failed naked attempt and expert trace,
        # highlighting reasoning branches the model was already approaching.
        _janc = custom.get("jaccard_anchor", "")
        if _janc:
            _j_masked = bool(cfg.get("reason_anchor_masked", True))
            _j_weight = float(cfg.get("reason_anchor_weight", 1.0))
            ctx.inject(
                _janc, masked=_j_masked, tag="jaccard_anchor", loss_weight=max(_j_weight, 0.0)
            )
        for i, chunk in enumerate(segments):
            ctx.inject(chunk, masked=True, tag=f"scaffold_{i}")
            if seg_cb:
                seg_cb("scaffold", chunk, f"scaffold_{i}")  # RED block in TUI stream
            last_gen, closed_think = _two_phase_generate(ctx, eff_b, b_ext, temp, i)
            if closed_think and (not vert or vert.stop_on_close):
                break
    else:
        gen = ctx.generate(
            max_tokens=thinking_budget,
            stop=["</think>"],
            temperature=temp,
            tag="thinking",
        )
        if gen.stop_reason == "stop_sequence" or "</think>" in gen.text:
            closed_think = True

    if not closed_think and (not vert or vert.close_inject):
        ctx.inject("</think>\n", masked=_m_tc)

    # Inject complete expert thinking as masked context before the answer.
    # Applied after </think> closes (model-generated or force-injected).
    if cfg.get("inject_target_thinking", False):
        _tgt = _target_thinking(ctx)
        if _tgt:
            ctx.inject("\n" + _tgt + "\n", masked=True, tag="target_thinking")

    # Intervention: inject thinking-level hint from external handler (before answer).
    # Placed after </think> close so the hint appears as additional context.
    intervention_text: str = custom.get("intervention_text", "")
    intervention_masked: bool = bool(custom.get("intervention_masked", True))
    if intervention_text:
        ctx.inject(intervention_text, masked=intervention_masked, tag="intervention")

    forced_answer_text: str = custom.get("forced_answer_text", "")
    if forced_answer_text:
        # Inject GT answer prefix as masked reference INSIDE the open \boxed{ — do NOT
        # close the brace. Model continues inside the box (same structure as normal path).
        # Loss does NOT flow through injected tokens; model's continuation IS trained on.
        ctx.inject("\n\\boxed{", masked=_m_bo, tag="answer_prefix")
        ctx.inject(forced_answer_text, masked=True, tag="forced_answer")
        _forced_budget = max(1, int(answer_reserve * custom.get("forced_answer_budget_mult", 3.0)))
        forced_gen = ctx.generate(max_tokens=_forced_budget, temperature=0.2, tag="answer")
        ctx.inject("}", masked=_m_bc)
        # Capture FULL answer = injected chars + model continuation so reward_fn sees
        # the complete answer text in result.variables["answer"] (e.g. "abc" + "def").
        from dataclasses import replace as _dc_replace  # noqa: PLC0415

        _full_ans = forced_answer_text + (forced_gen.text or "")
        ctx.capture("answer", _dc_replace(forced_gen, text=_full_ans))
        # Post-answer scaffold: inject start of target solution so model learns to verify.
        # Encodes ground_truth, slices first N tokens, decodes back, injects as masked scaffold.
        _ps_toks = custom.get("forced_answer_post_scaffold_tokens", 0)
        if _ps_toks > 0:
            _gt_text: str = custom.get("forced_answer_ground_truth", "")
            if _gt_text:
                _gt_ids = ctx.encode(_gt_text)
                _gt_prefix = ctx.decode(_gt_ids[:_ps_toks])
                if _gt_prefix:
                    ctx.inject(_gt_prefix, masked=True, tag="post_scaffold")
        # Post-answer generation: let the model reason/verify after seeing the answer.
        # Loss flows through these tokens — ICL effect trains on post-answer reflection.
        _post_toks = custom.get("forced_answer_post_tokens", 0)
        if _post_toks > 0:
            ctx.generate(max_tokens=_post_toks, temperature=temp, tag="post_answer")
    else:
        ap_text: str = custom.get("answer_prefix_text", "")
        ap_toks: int = custom.get("answer_prefix_toks", 0)
        ctx.inject("\n\\boxed{", masked=_m_bo, tag="answer_prefix")
        if ap_text:
            ctx.inject(ap_text, masked=True, tag="answer_hint")
        _ans_stop: list[str] | None = cfg.get("answer_stop") or None
        answer_result = ctx.generate(
            max_tokens=max(1, answer_reserve - ap_toks),
            temperature=0.2,
            stop=_ans_stop,
            tag="answer",
        )
        ctx.capture("answer", answer_result)
        if seg_cb:
            seg_cb("answer_prefix", "\\boxed{" + (answer_result.text or "") + "}", "answer")
        ctx.inject("}", masked=_m_bc)


# ── Horizontal loop ───────────────────────────────────────────────────────────


def run_escalation(
    trainer: Any,
    sample: Sample,
    task_type: TaskType,
    merged_type_config: dict[str, Any],
    reward_fn: BaseReward,
    prompt_tokens: list[int] | None,
    prompt_cache: Any,
    iteration: int,
    max_completion_tokens: int,
    esc_cfg: EscalationConfig,
    pending_events: list | None = None,
    force_all_answers: str = "",
    region_offset_chars: int = 0,
    region_length_chars: int = 0,
    chunk_count_override: int = 0,
    fallback: bool = False,
) -> list[RolloutResult]:
    """Horizontal escalation loop with adaptive planning and group-level shaping.

    Member 0 gets 1 chunk, member k gets (k+1)*member_step_chunks chunks.
    Reward computed inline; early-stop when reward >= threshold.
    Naked view always appended last when naked_always=True.
    Group-level shaping runs AFTER all members finish.
    """
    # Lazy import — avoids circular at module load; safe at call-time.
    from . import StepMetrics  # noqa: PLC0415

    horiz = esc_cfg.horizontal
    vert = esc_cfg.vertical
    sc = esc_cfg.scaffold
    rs = esc_cfg.reward_shaping
    tf = esc_cfg.tri_factor

    target_text = sample.answer or ""
    ground_truth = sample.metadata.get("ground_truth", sample.answer)
    callbacks = trainer._build_callbacks(task_type)
    _cd = getattr(trainer.config, "rollout_cooldown", 0.5)
    total_iters = getattr(trainer.config, "iterations", 1)
    _bridge = getattr(trainer, "_tui_bridge", None)  # TUI bridge for segment callbacks

    # Use reason_anchor compressed skeleton as ICL scaffold source when enabled.
    # The anchor's logical skeleton becomes the scaffold content, so:
    # - Each chunk is a key reasoning step (not a raw CoT slice)
    # - IS proximity automatically keys off anchor sentence boundaries
    # - Progressive masking reveals the anchor skeleton step by step
    _scaffold_text = (
        _anchor_text_for_scaffold(sample.prompt or "", target_text, ground_truth or "")
        if merged_type_config.get("reason_anchor", False)
        else target_text
    )

    # Adaptive member cap: proportional to expert trace richness (n_eff_max + 1 levels)
    n_eff_max = _count_n_eff_max(_scaffold_text, sc.boundary_mode, sc.split_boundaries)
    scaffolded_max = min(horiz.max_members, max(n_eff_max + 1, 1))
    # forced_answer_naked_followup ignores max_members — add its slot unconditionally
    _followup_slot = horiz.forced_answer_on_last_member and horiz.forced_answer_naked_followup
    total_members = scaffolded_max + (1 if horiz.naked_always else 0) + (1 if _followup_slot else 0)
    # Cap C_max at vert.n_chunks — long expert traces (200+ boundaries) would otherwise
    # produce n_chunks=67-200 per member, triggering hundreds of generation cycles.
    _vert_cap = vert.n_chunks if vert is not None else 16
    c_max = max(min(n_eff_max, _vert_cap), 1)
    _step = c_max / max(scaffolded_max, 1)
    is_capped = scaffolded_max < horiz.max_members

    group_rollouts: list[RolloutResult] = []
    early_stop = False
    _forced_fired = False  # set True when forced_answer actually injected
    jaccard_anchor = ""  # populated after naked fails (naked_first=True only)
    _ap_max = sc.answer_prefix_tokens
    from .parallel import _ROLLOUT_LOCK  # noqa: PLC0415

    # ── Naked-first path ──────────────────────────────────────────────────────
    # When naked_first=True: run naked first; if it passes skip all scaffolded.
    # On failure: compute Jaccard anchor to focus the scaffolded members.
    if horiz.naked_always and horiz.naked_first:
        _nidx = 0
        _nseg_cb = _bridge.make_segment_cb(_nidx) if _bridge else None
        _nbatch = BatchContext(
            iteration=iteration,
            total_iters=total_iters,
            group_index=_nidx,
            group_size=total_members,
            scaffold=ScaffoldConfig(mode="none"),
            custom={
                "segments": [],
                "vert_cfg": vert,
                "N_eff": 0,
                "B_eff": 0,
                "b_ext": tf.b_ext,
                "answer_prefix_text": "",
                "answer_prefix_toks": 0,
                "segment_cb": _nseg_cb,
            },
        )
        _nrs_ev0 = StepMetrics(
            iteration=iteration,
            extras={
                "prompt": sample.prompt[:300],
                "type_name": sample.type_name,
                "group_index": 1,
                "group_size": total_members,
                "member_idx": _nidx,
                "allocated_chunks": 0,
                "is_naked": True,
            },
        )
        if pending_events is not None:
            pending_events.append(("on_rollout_start", _nrs_ev0))
        else:
            trainer._fire("on_rollout_start", _nrs_ev0)
        with _ROLLOUT_LOCK:
            _nr0 = trainer.executor.execute(
                _vertical_rollout,
                prompt=sample.prompt,
                prompt_tokens=prompt_tokens,
                answer=sample.answer,
                type_name=sample.type_name,
                type_config=merged_type_config,
                batch=_nbatch,
                callbacks=callbacks,
                sample=sample.metadata,
                max_completion_tokens=max_completion_tokens,
                prompt_cache=prompt_cache,
            )
            _sync_gpu()
        if _cd > 0:
            time.sleep(_cd)
        _nr0_reward = reward_fn.compute(_nr0, ground_truth)
        _nr0_np = sum(len(s.tokens) for s in _nr0.segments if s.source != SegmentSource.PROMPT)
        _nr0.metadata.update(
            {
                "_esc_member_idx": _nidx,
                "_esc_allocated_chunks": 0,
                "_esc_n_eff": 0,
                "_esc_reward": _nr0_reward,
                "_esc_weights": [1.0] * _nr0_np,
                "_esc_early_stop": False,
                "_esc_n_eff_max": n_eff_max,
                "_esc_effective_members": total_members,
                "_view_name": "naked",
                "_view_is_naked": True,
                "_member_role": "naked_first",
            }
        )
        group_rollouts.append(_nr0)
        _nog_ev0 = StepMetrics(
            iteration=iteration,
            extras={
                "phase": "rollout",
                "group_index": 1,
                "group_size": total_members,
                "type_name": sample.type_name,
            },
        )
        if pending_events is not None:
            pending_events.append(("on_group", _nog_ev0))
        else:
            trainer._fire("on_group", _nog_ev0)

        if _nr0_reward >= horiz.reward_threshold and horiz.skip_if_naked_correct:
            # Naked passed and skip enabled — skip all scaffolded members
            _apply_group_shaping(group_rollouts, rs, c_max, iteration, total_iters)
            print(
                f"[escalation] naked_first=✓ naked_passed (r={_nr0_reward:.2f}) "
                f"— skipped {scaffolded_max} scaffolded members"
            )
            return group_rollouts

        # Naked failed — build Jaccard anchor from intersection with expert trace
        _naked_text = " ".join(
            s.text
            for s in _nr0.segments
            if s.source != SegmentSource.PROMPT and s.tag not in ("answer", "answer_prefix")
        )
        _expert_stripped = target_text.lstrip()
        if _expert_stripped.startswith("<think>"):
            _expert_stripped = _expert_stripped[len("<think>") :]
        _etag_end = _expert_stripped.find("</think>")
        if _etag_end != -1:
            _expert_stripped = _expert_stripped[:_etag_end]
        _max_w = merged_type_config.get("reason_anchor_max_words", 80) if merged_type_config else 80
        jaccard_anchor = _jaccard_anchor_from_naked(_naked_text, _expert_stripped, _max_w)

    # ── Fallback path: full thinking block injection as last resort ──────────
    if fallback and not force_all_answers:
        force_all_answers = ground_truth or target_text

    # ── Scaffolded members (horizontal loop) ─────────────────────────────────
    # group_index offset: +1 when naked already occupies slot 0 (naked_first=True)
    _gidx_offset = 1 if (horiz.naked_always and horiz.naked_first) else 0

    for member_idx in range(scaffolded_max):
        if sc.random_prefix:
            n_chunks = random.randint(sc.random_prefix_min, max(1, n_eff_max))
        else:
            n_chunks = max(1, round((member_idx + 1) * _step))
        if chunk_count_override > 0:
            n_chunks = chunk_count_override
        segments, n_eff, b_eff = _plan_adaptive_scaffold(
            _scaffold_text,
            n_chunks,
            trainer.backend,
            sc.max_tokens,
            sc.boundary_mode,
            sc.split_boundaries,
            region_offset_chars=region_offset_chars,
            region_length_chars=region_length_chars,
        )
        ap_k = max(0, int(_ap_max * (1.0 - member_idx / max(scaffolded_max, 1))))
        ap_text = _answer_prefix_text(target_text, ap_k, trainer.backend) if ap_k > 0 else ""
        ap_toks = len(trainer.backend.encode(ap_text)) if ap_text else 0
        seg_cb = _bridge.make_segment_cb(member_idx + _gidx_offset) if _bridge else None
        # forced_answer injection: last scaffolded member only when no prior member passed.
        # Probability decays linearly from 1.0 → forced_answer_end_prob over training.
        _forced_answer_text = ""
        if (
            horiz.forced_answer_on_last_member
            and member_idx == scaffolded_max - 1
            and not early_stop
        ):
            _progress = min(1.0, iteration / max(total_iters, 1))
            _forced_p = 1.0 - _progress * (1.0 - horiz.forced_answer_end_prob)
            if random.random() < _forced_p:
                _fa_raw = _extract_forced_answer(ground_truth or "")
                _n_chars = horiz.forced_answer_inject_chars
                _forced_answer_text = _fa_raw if _n_chars < 0 else _fa_raw[:_n_chars]
        # Override: force answer on ALL members when trainer requests it (repeat-level forcing)
        if force_all_answers:
            _forced_answer_text = force_all_answers

        # Intervention: call external handler to get thinking-level injection text.
        # trigger controls when it fires (last_member | always | on_failure).
        _intervention_text = ""
        _iv = esc_cfg.intervention
        if _iv.enabled:
            _iv_trigger = _iv.trigger
            _fires = (
                _iv_trigger == "always"
                or (_iv_trigger == "last_member" and member_idx == scaffolded_max - 1)
                or (
                    _iv_trigger == "on_failure"
                    and member_idx == scaffolded_max - 1
                    and not early_stop
                )
            )
            if _fires:
                _gp: list[GroupMemberRecord] = build_group_progress(group_rollouts)
                _iv_result = call_intervention(
                    _iv, sample.prompt or "", sample.answer, ground_truth, _gp
                )
                if _iv_result is not None:
                    _intervention_text = _iv_result

        batch_ctx = BatchContext(
            iteration=iteration,
            total_iters=total_iters,
            group_index=member_idx + _gidx_offset,
            group_size=total_members,
            scaffold=ScaffoldConfig(mode="prefix" if segments else "none"),
            custom={
                "segments": segments,
                "vert_cfg": vert,
                "N_eff": n_eff,
                "B_eff": b_eff,
                "b_ext": tf.b_ext,
                "answer_prefix_text": ap_text,
                "answer_prefix_toks": ap_toks,
                "segment_cb": seg_cb,
                "jaccard_anchor": jaccard_anchor,
                "forced_answer_text": _forced_answer_text,
                "forced_answer_budget_mult": horiz.forced_answer_budget_mult,
                "forced_answer_post_tokens": horiz.forced_answer_post_tokens,
                "forced_answer_post_scaffold_tokens": horiz.forced_answer_post_scaffold_tokens,
                "forced_answer_weight": horiz.forced_answer_weight,
                "forced_answer_ground_truth": ground_truth or "",
                "intervention_text": _intervention_text,
                "intervention_masked": esc_cfg.intervention.masked,
            },
        )
        _rs_ev = StepMetrics(
            iteration=iteration,
            extras={
                "prompt": sample.prompt[:300],
                "type_name": sample.type_name,
                "group_index": member_idx + _gidx_offset + 1,
                "group_size": total_members,
                "member_idx": member_idx,
                "allocated_chunks": n_chunks,
                "is_naked": False,
            },
        )
        if pending_events is not None:
            pending_events.append(("on_rollout_start", _rs_ev))
        else:
            trainer._fire("on_rollout_start", _rs_ev)
        with _ROLLOUT_LOCK:
            result = trainer.executor.execute(
                _vertical_rollout,
                prompt=sample.prompt,
                prompt_tokens=prompt_tokens,
                answer=sample.answer,
                type_name=sample.type_name,
                type_config=merged_type_config,
                batch=batch_ctx,
                callbacks=callbacks,
                sample=sample.metadata,
                max_completion_tokens=max_completion_tokens,
                prompt_cache=prompt_cache,
            )
            _sync_gpu()
        if _cd > 0:
            time.sleep(_cd)
        reward = reward_fn.compute(result, ground_truth)
        if _forced_answer_text:
            # GT was injected as masked context — force reward=1.0 so this member
            # always provides a positive advantage signal (ICL accumulation).
            reward = 1.0
            result.metadata["_esc_forced_answer"] = True
            _forced_fired = True
        esc_w = _build_tri_factor_weights(result, n_chunks, c_max, n_eff, tf)
        if _forced_answer_text:
            # Override tri-factor weights with flat custom value so the ICL member
            # doesn't get compressed by the scaffold-depth decay.
            _fa_w = horiz.forced_answer_weight
            esc_w = [_fa_w] * len(esc_w)
        _member_role = "escalation"
        if _forced_answer_text:
            _member_role = "forced_answer"
        result.metadata.update(
            {
                "_esc_member_idx": member_idx + _gidx_offset,
                "_esc_allocated_chunks": n_chunks,
                "_esc_n_eff": n_eff,
                "_esc_reward": reward,
                "_esc_weights": esc_w,
                "_esc_n_eff_max": n_eff_max,
                "_esc_effective_members": total_members,
                "_view_name": f"esc_{member_idx}",
                "_view_is_naked": False,
                "_member_role": _member_role,
                "_esc_region_offset": region_offset_chars,
                "_esc_region_length": region_length_chars,
            }
        )
        group_rollouts.append(result)
        _og_ev = StepMetrics(
            iteration=iteration,
            extras={
                "phase": "rollout",
                "group_index": member_idx + _gidx_offset + 1,
                "group_size": total_members,
                "type_name": sample.type_name,
            },
        )
        if pending_events is not None:
            pending_events.append(("on_group", _og_ev))
        else:
            trainer._fire("on_group", _og_ev)
        if reward >= horiz.reward_threshold:
            early_stop = True
            break

    # ── Naked followup after forced answer ───────────────────────────────────
    # When forced_answer_naked_followup=True and forced fired: always run regardless
    # of max_members (slot was pre-counted in total_members above).
    if _forced_fired and _followup_slot:
        _fnf_idx = len(group_rollouts)
        _fnf_batch = BatchContext(
            iteration=iteration,
            total_iters=total_iters,
            group_index=_fnf_idx,
            group_size=total_members,
            scaffold=ScaffoldConfig(mode="none"),
            custom={
                "segments": [],
                "vert_cfg": vert,
                "N_eff": 0,
                "B_eff": 0,
                "b_ext": tf.b_ext,
                "answer_prefix_text": "",
                "answer_prefix_toks": 0,
                "segment_cb": _bridge.make_segment_cb(_fnf_idx) if _bridge else None,
            },
        )
        _fnf_rs_ev = StepMetrics(
            iteration=iteration,
            extras={
                "prompt": sample.prompt[:300],
                "type_name": sample.type_name,
                "group_index": _fnf_idx + 1,
                "group_size": total_members,
                "member_idx": _fnf_idx,
                "allocated_chunks": 0,
                "is_naked": True,
            },
        )
        if pending_events is not None:
            pending_events.append(("on_rollout_start", _fnf_rs_ev))
        else:
            trainer._fire("on_rollout_start", _fnf_rs_ev)
        with _ROLLOUT_LOCK:
            _fnf_result = trainer.executor.execute(
                _vertical_rollout,
                prompt=sample.prompt,
                prompt_tokens=prompt_tokens,
                answer=sample.answer,
                type_name=sample.type_name,
                type_config=merged_type_config,
                batch=_fnf_batch,
                callbacks=callbacks,
                sample=sample.metadata,
                max_completion_tokens=max_completion_tokens,
                prompt_cache=prompt_cache,
            )
            _sync_gpu()
        if _cd > 0:
            time.sleep(_cd)
        _fnf_reward = reward_fn.compute(_fnf_result, ground_truth)
        _fnf_np = sum(
            len(s.tokens) for s in _fnf_result.segments if s.source != SegmentSource.PROMPT
        )
        _fnf_result.metadata.update(
            {
                "_esc_member_idx": _fnf_idx,
                "_esc_allocated_chunks": 0,
                "_esc_n_eff": 0,
                "_esc_reward": _fnf_reward,
                "_esc_weights": [1.0] * _fnf_np,
                "_esc_n_eff_max": n_eff_max,
                "_esc_effective_members": total_members,
                "_view_name": "naked_followup",
                "_view_is_naked": True,
                "_esc_forced_followup": True,
                "_member_role": "naked_followup",
            }
        )
        group_rollouts.append(_fnf_result)
        if pending_events is not None:
            pending_events.append(
                (
                    "on_group",
                    StepMetrics(
                        iteration=iteration,
                        extras={
                            "phase": "rollout",
                            "group_index": _fnf_idx + 1,
                            "group_size": total_members,
                            "type_name": sample.type_name,
                        },
                    ),
                )
            )
        else:
            trainer._fire(
                "on_group",
                StepMetrics(
                    iteration=iteration,
                    extras={
                        "phase": "rollout",
                        "group_index": _fnf_idx + 1,
                        "group_size": total_members,
                        "type_name": sample.type_name,
                    },
                ),
            )

    # ── Naked at end (original path: naked_always=True, naked_first=False) ────
    if horiz.naked_always and not horiz.naked_first:
        naked_idx = len(group_rollouts)
        naked_seg_cb = _bridge.make_segment_cb(naked_idx) if _bridge else None
        naked_batch = BatchContext(
            iteration=iteration,
            total_iters=total_iters,
            group_index=naked_idx,
            group_size=total_members,
            scaffold=ScaffoldConfig(mode="none"),
            custom={
                "segments": [],
                "vert_cfg": vert,
                "N_eff": 0,
                "B_eff": 0,
                "b_ext": tf.b_ext,
                "answer_prefix_text": "",
                "answer_prefix_toks": 0,
                "segment_cb": naked_seg_cb,
            },
        )
        _nrs_ev = StepMetrics(
            iteration=iteration,
            extras={
                "prompt": sample.prompt[:300],
                "type_name": sample.type_name,
                "group_index": naked_idx + 1,
                "group_size": total_members,
                "member_idx": naked_idx,
                "allocated_chunks": 0,
                "is_naked": True,
            },
        )
        if pending_events is not None:
            pending_events.append(("on_rollout_start", _nrs_ev))
        else:
            trainer._fire("on_rollout_start", _nrs_ev)
        with _ROLLOUT_LOCK:
            naked_result = trainer.executor.execute(
                _vertical_rollout,
                prompt=sample.prompt,
                prompt_tokens=prompt_tokens,
                answer=sample.answer,
                type_name=sample.type_name,
                type_config=merged_type_config,
                batch=naked_batch,
                callbacks=callbacks,
                sample=sample.metadata,
                max_completion_tokens=max_completion_tokens,
                prompt_cache=prompt_cache,
            )
            _sync_gpu()
        if _cd > 0:
            time.sleep(_cd)
        naked_reward = reward_fn.compute(naked_result, ground_truth)
        n_np = sum(len(s.tokens) for s in naked_result.segments if s.source != SegmentSource.PROMPT)
        naked_result.metadata.update(
            {
                "_esc_member_idx": naked_idx,
                "_esc_allocated_chunks": 0,
                "_esc_n_eff": 0,
                "_esc_reward": naked_reward,
                "_esc_weights": [1.0] * n_np,
                "_esc_early_stop": early_stop,
                "_esc_n_eff_max": n_eff_max,
                "_esc_effective_members": total_members,
                "_view_name": "naked",
                "_view_is_naked": True,
                "_member_role": "naked_last",
            }
        )
        group_rollouts.append(naked_result)
        _nog_ev = StepMetrics(
            iteration=iteration,
            extras={
                "phase": "rollout",
                "group_index": naked_idx + 1,
                "group_size": total_members,
                "type_name": sample.type_name,
            },
        )
        if pending_events is not None:
            pending_events.append(("on_group", _nog_ev))
        else:
            trainer._fire("on_group", _nog_ev)

    # Group-level shaping — MUST run after ALL members (including naked) complete
    _apply_group_shaping(group_rollouts, rs, c_max, iteration, total_iters)

    n_members = len(group_rollouts) - (1 if horiz.naked_always else 0)
    _cap = f" (capped from {horiz.max_members})" if is_capped else ""
    _nk_sym = "first" if horiz.naked_first else ("last" if horiz.naked_always else "✗")
    print(
        f"[escalation] members={n_members}{_cap} n_eff_max={n_eff_max} "
        f"early_stop={early_stop} naked={_nk_sym}"
    )
    return group_rollouts


# ── Stats collection ──────────────────────────────────────────────────────────


def collect_escalation_stats(group_rollouts: list[RolloutResult]) -> dict[str, Any]:
    """Aggregate per-member stats from escalation rollouts for CSV/WandB."""
    members = [r for r in group_rollouts if "_esc_member_idx" in r.metadata]
    if not members:
        return {}
    rewards = [r.metadata.get("_esc_reward", 0.0) for r in members]
    n_eff = [r.metadata.get("_esc_n_eff", 0) for r in members]
    return {
        "esc_n_members": len(members),
        "esc_member_rewards": rewards,
        "esc_mean_reward": sum(rewards) / len(rewards),
        "esc_max_reward": max(rewards),
        "esc_chunks_used": n_eff,
        "esc_early_stop": any(r.metadata.get("_esc_early_stop") for r in members),
    }
