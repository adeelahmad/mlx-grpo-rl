"""EscalationConfig — ICL-Guided Dynamic Escalation Engine configuration.

Vertical loop: interleaved scaffold inject → model generate cycles within one member.
Horizontal loop: escalating scaffold depth across group members, inline reward check.

Usage in YAML:
  escalation:
    enabled: true
    vertical:
      n_chunks: 4
      stop_on_close: true
      close_inject: true
    horizontal:
      reward_threshold: 1.0
      max_members: 6
      member_step_chunks: 1
      naked_always: true
    scaffold:
      target: thinking
      max_tokens: 512
    tracking:
      log_member_rewards: true
      wandb_prefix: "esc"
    reward_shaping:
      lambda_s_max: 0.05
      lambda_f_max: 0.02
    tri_factor:
      alpha_h: 0.7
      alpha_v: 0.8
      alpha_p: 0.9
      b_ext: 64
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class VerticalConfig:
    """Controls interleaved scaffold+generation cycles within one rollout member."""

    n_chunks: int = 4  # max scaffold injection points per rollout member
    stop_on_close: bool = True  # halt scaffold cycle when model emits </think>
    close_inject: bool = True  # inject </think> if model never closes naturally
    # Per-cycle generation cap for scaffolded members. 0 = use thinking_budget/n_cycles.
    # Set to scaffold.max_tokens (e.g. 32) to enforce 1:1 scaffold:generated ratio per cycle.
    # Naked member is unaffected — it always uses thinking_budget directly.
    max_cycle_gen_tokens: int = 0


@dataclass(slots=True, frozen=True)
class HorizontalConfig:
    """Controls the outer loop across group members."""

    reward_threshold: float = 1.0  # reward >= this → skip remaining scaffold members
    max_members: int = 6  # max scaffold members (naked view added on top)
    member_step_chunks: int = 1  # additional scaffold chunks per horizontal step
    naked_always: bool = True  # always run a pure-naked view
    naked_first: bool = False  # run naked first; skip scaffolded if reward >= threshold
    skip_if_naked_correct: bool = True  # naked_first=True only: skip scaffolded when naked passes
    # When no group member has passed by the last scaffolded slot: inject the GT answer
    # as masked context so the model sees the correct answer, then generate with a
    # bigger budget. Reward is forced to 1.0 (GT is present); loss flows through the
    # model's generated thinking and post-answer tokens only (injected = masked).
    # ICL effect: positive advantage on the thinking that preceded the correct answer.
    forced_answer_on_last_member: bool = False
    forced_answer_budget_mult: float = 3.0  # answer_max_tokens × this for post-inject gen
    forced_answer_end_prob: float = 0.2  # injection probability floor (1.0→end_prob over training)
    forced_answer_inject_chars: int = -1  # chars of GT to inject; -1 = full answer
    forced_answer_post_tokens: int = 0  # tokens model may generate AFTER closing }; 0=off
    forced_answer_post_scaffold_tokens: int = 0  # inject first N tok of target after }; 0=off
    forced_answer_weight: float = 1.0  # flat per-token IS weight for forced member; 1.0 = no change
    # When forced_answer fires, optionally run one more naked rollout right after.
    # Lets you measure immediate ICL transfer: did seeing the GT answer help?
    forced_answer_naked_followup: bool = False


@dataclass(slots=True, frozen=True)
class EscalationScaffoldConfig:
    """Scaffold source and token budget per chunk."""

    target: str = "thinking"  # thinking | answer | both
    max_tokens: int = 512  # max tokens injected per scaffold chunk (cap)
    boundary_mode: str = "sentence"  # sentence (.\n ?.\n !\n \n\n) | line (\n)
    answer_prefix_tokens: int = 0  # inject first N tok of ground-truth answer; 0=off
    # Regex patterns for boundary detection. Non-empty overrides boundary_mode.
    # e.g. ["\n\n", "\n", "\\. (?=[A-Z])"] splits on paragraphs, newlines, and sentences.
    split_boundaries: tuple[str, ...] = ()
    # Randomise prefix length per member: uniform draw from [random_prefix_min, n_eff_max].
    # Each repeat of the same sample gets a different scaffold depth → more diverse gradients.
    # 0=off (deterministic ladder). Works with and without sample_repeat.
    random_prefix: bool = False
    random_prefix_min: int = 0  # 0 = naked allowed; 1 = always at least 1 chunk


@dataclass(slots=True, frozen=True)
class EscalationTrackingConfig:
    """Logging and WandB reporting for escalation stats."""

    log_member_rewards: bool = True  # stamp _esc_reward in each member's metadata
    wandb_prefix: str = "esc"  # WandB key prefix (e.g. "esc/mean_reward")


@dataclass(slots=True, frozen=True)
class RewardShapingConfig:
    """Bidirectional group-level reward shaping hyperparameters.

    λ_s increases over training: nudges model toward independence (reward independence).
    λ_f decreases over training: early nudge to learn structure; fades as model matures.
    Both are tiny relative to the correctness signal (20:1 dominance at max shaping).

    prefer_scaffolded: when both naked and scaffolded pass (raw reward >= 1.0), add
    prefer_scaffolded_bonus to scaffolded _esc_shaped_reward. Creates a tiebreaker
    advantage gradient that rewards structured reasoning over unguided generation.
    """

    lambda_s_max: float = 0.05  # success: max independence nudge (tiebreaker, not penalty)
    lambda_f_max: float = 0.02  # failure: max structure nudge (tiny positive for structure)
    prefer_scaffolded: bool = False  # boost scaffolded when both naked+scaffolded pass
    prefer_scaffolded_bonus: float = 0.1  # shaped reward bonus for scaffolded on tie
    any_correct_threshold: float = 0.5  # min reward to count as "correct" in group shaping
    # Old implicit behaviour was > 0.0. 0.5 excludes ExamReward partial-format credit
    # (typically 0.14–0.29). Set to 0.0001 in YAML to recover exact old behaviour.
    failure_floor: float = 0.0  # cap on failure-path shaped reward: min(λ_f*c_k/cm, floor)
    # Default 0.0 = wrong answers get zero shaped reward (clean gradient signal).
    # Set > 0 (e.g. 0.01) to re-enable the structure nudge.
    reward_shaping_v2: bool = False
    # v2: shaped = raw + λ_s*(1-raw)*(1-c_k/c_max) — boosts correct with less scaffold, no penalty
    # v1 (default/False): shaped = raw * (1 - λ_s*c_k/c_max) — legacy, penalises scaffold-heavy


@dataclass(slots=True, frozen=True)
class TriFactorConfig:
    """Tri-factor per-token importance weight hyperparameters.

    w(k, i, j) = α_h^(C_k/C_max) · α_v^(i/max(N_eff-1,1)) · α_p^j
    where k=scaffold depth, i=cycle_idx (vertical), j=causal_dist (proximity).
    """

    alpha_h: float = 0.7  # horizontal: decay by scaffold depth fraction
    alpha_v: float = 0.8  # vertical: decay by cycle index fraction
    alpha_p: float = 0.9  # proximity: decay by causal distance from last scaffold
    b_ext: int = 64  # Phase B extension budget (tokens)


@dataclass(slots=True, frozen=True)
class InterventionConfig:
    """Pluggable intervention hook — called before a target group member's rollout.

    The handler receives group_progress (all completed members + rewards so far)
    and returns text to inject into the thinking context, or None to skip.

    Loading (first non-empty wins):
      lambda_expr  — inline lambda: "lambda prompt, answer, gt, progress: ..."
      script       — path to .py file with handle(prompt, answer, gt, progress)
      callable_path — "my.module:function_name" dotted-module:function

    trigger controls when the handler fires:
      last_member — only on the final scaffolded member (default, mirrors forced_answer)
      always      — on every scaffolded member
      on_failure  — last member only when no prior member reached reward_threshold
    """

    enabled: bool = False
    lambda_expr: str = ""  # inline lambda string
    script: str = ""  # path to .py file with handle()
    callable_path: str = ""  # "my.module:handle" dotted import
    trigger: str = "last_member"  # last_member | always | on_failure
    masked: bool = True  # mask injected text from gradient (no loss)


@dataclass(slots=True, frozen=True)
class DiverseScaffoldConfig:
    """Adaptive diverse scaffold repetition across sample repeats.

    Divides expert thinking blocks into non-overlapping regions. Each repeat
    scaffolds a DIFFERENT region, ensuring monotonic cumulative coverage growth.
    Reduces duplicate gradient signals and KL spikes from over-repetition.
    """

    enabled: bool = False
    num_regions: int = 4  # divide thinking block into this many regions
    density_decay: float = 0.8  # chunk count per repeat = base * decay^repeat_idx
    coverage_threshold: float = 0.9  # trigger fallback when coverage >= this
    min_chunks: int = 1  # floor for diminishing density
    fallback_inject_full: bool = True  # last resort: inject full thinking masked


@dataclass(slots=True, frozen=True)
class EscalationConfig:
    """Root escalation config — disabled by default, non-breaking."""

    enabled: bool = False
    vertical: VerticalConfig = field(default_factory=VerticalConfig)
    horizontal: HorizontalConfig = field(default_factory=HorizontalConfig)
    scaffold: EscalationScaffoldConfig = field(default_factory=EscalationScaffoldConfig)
    tracking: EscalationTrackingConfig = field(default_factory=EscalationTrackingConfig)
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)
    tri_factor: TriFactorConfig = field(default_factory=TriFactorConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    diverse: DiverseScaffoldConfig = field(default_factory=DiverseScaffoldConfig)


def build_escalation_config(data: dict | None) -> EscalationConfig:
    """Parse an EscalationConfig from a YAML dict.

    Unknown keys are silently ignored (forward-compatible).
    Sub-configs (vertical, horizontal, scaffold, tracking, reward_shaping,
    tri_factor) are optional; omitted sections use dataclass defaults.
    """
    if not data:
        return EscalationConfig()

    def _sub(cls: type, key: str) -> object:
        raw = data.get(key)
        if not raw or not isinstance(raw, dict):
            return cls()
        flds = cls.__dataclass_fields__
        kwargs = {k: v for k, v in raw.items() if k in flds}
        # Coerce list→tuple for split_boundaries (YAML lists → Python lists)
        if "split_boundaries" in kwargs and isinstance(kwargs["split_boundaries"], list):
            kwargs["split_boundaries"] = tuple(kwargs["split_boundaries"])
        return cls(**kwargs)

    _sub_keys = {
        "vertical",
        "horizontal",
        "scaffold",
        "tracking",
        "reward_shaping",
        "tri_factor",
        "intervention",
        "diverse",
    }
    top_flds = EscalationConfig.__dataclass_fields__
    top = {k: v for k, v in data.items() if k in top_flds and k not in _sub_keys}

    return EscalationConfig(
        **top,
        vertical=_sub(VerticalConfig, "vertical"),  # type: ignore[arg-type]
        horizontal=_sub(HorizontalConfig, "horizontal"),  # type: ignore[arg-type]
        scaffold=_sub(EscalationScaffoldConfig, "scaffold"),  # type: ignore[arg-type]
        tracking=_sub(EscalationTrackingConfig, "tracking"),  # type: ignore[arg-type]
        reward_shaping=_sub(RewardShapingConfig, "reward_shaping"),  # type: ignore[arg-type]
        tri_factor=_sub(TriFactorConfig, "tri_factor"),  # type: ignore[arg-type]
        intervention=_sub(InterventionConfig, "intervention"),  # type: ignore[arg-type]
        diverse=_sub(DiverseScaffoldConfig, "diverse"),  # type: ignore[arg-type]
    )
