"""Configuration loader — YAML with inline lambdas and dynamic types.

Loads a single YAML file and produces:
- TrainingConfig (dataclass)
- Dynamically created TaskTypes for any types defined in YAML
- Registered callbacks and rewards

    config = load_config("config.yml")
    config.types  → {"mcq": MCQType, "my_custom": DynamicType}
    config.training.lr  → 1e-5
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..rewards.base import register_lambda_reward
from ..scaffold import ScaffoldConfig, build_scaffold_config
from ..types.yaml_type import create_yaml_type, parse_lambda
from .dynamic_scaling import DynamicGradientScalingConfig
from .escalation import EscalationConfig, build_escalation_config
from .logging import LoggingConfig
from .vision import VisionConfig

# ── MoE Configuration ─────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ExpertTypeMappingConfig:
    """Maps a task type name to a specific expert index."""

    type_name: str = ""
    expert_index: int = 0
    lora_rank: int | None = None  # None → inherits MoEConfig.lora_rank
    lora_alpha: float | None = None


@dataclass(slots=True, frozen=True)
class GateConfig:
    """Gating network hyperparameters."""

    hidden_dim: int = 64
    pool_method: str = "mean"  # mean | first | last | max
    temperature: float = 1.0
    temperature_min: float = 0.1
    temperature_decay_steps: int = 500
    noise_std: float = 0.0
    noise_decay_steps: int = 200


@dataclass(slots=True, frozen=True)
class RoutingConfig:
    """Supervised-to-learned routing schedule."""

    supervised_steps: int = 100
    transition_steps: int = 100
    supervised_loss_weight: float = 1.0
    balance_loss_weight: float = 0.01


@dataclass(slots=True, frozen=True)
class MoEConfig:
    """LoRA Mixture-of-Experts adapter configuration."""

    num_experts: int = 3
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("self_attn.q_proj", "self_attn.v_proj")
    lora_layers: int | None = None
    gate: GateConfig = field(default_factory=GateConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    expert_types: tuple[ExpertTypeMappingConfig, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class JaccardBoostConfig:
    """Jaccard similarity token boosting configuration."""

    enabled: bool = False
    boost_factor: float = 1.5
    apply_segment: str = "answer"  # answer | thinking | both
    min_jaccard: float = 0.1
    apply_before_conditional: bool = True
    exclude_patterns: tuple[str, ...] = field(default_factory=tuple)
    stop_words: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    """Core training parameters."""

    model_path: str = ""
    adapter_path: str | None = None
    adapter_type: str = "lora"  # lora | dora | full
    learning_rate: float = 1e-5
    iterations: int = 500
    epochs: int = 0  # 0 = use iterations; >0 overrides iterations
    batch_size: int = 1
    samples_per_step: int = 1  # Repeat each batch N times (fresh rollouts each)
    # 1=off; N>1=fixed repeats; -1=until any non-forced member correct; -2=until naked correct
    sample_repeat: int = 1
    sample_repeat_threshold: float = 1.0  # reward threshold for -1/-2 stop condition
    sample_repeat_max: int = 20  # max repeats for -1/-2 (prevents infinite loop)
    # KL breaker: if KL exceeds this on any rep, skip remaining repeats for that sample.
    # 0.0 = disabled. Default 0.08 = 2x the default monitor kl_warning (0.04).
    sample_repeat_kl_threshold: float = 0.08
    # Naked repeats: run N repeats with zero scaffold before/after normal cycle.
    # Helps wean model off scaffold by training on unassisted attempts first/last.
    # 0 = disabled (default). Only applies when sample_repeat != 1.
    before_repeat_naked_repeats: int = 0
    after_repeat_naked_repeats: int = 0
    # Force answer injection every other repeat after N consecutive repeats.
    # 0 = disabled. When >0, reps ≥ N where (rep-N) % 2 == 0 inject ground truth.
    # Forced reps skip stop conditions and don't count toward repeat_max.
    # Uses escalation config's forced_answer_post_tokens for post-answer generation budget.
    force_answer_on_every_other_repeat_after_repeat_number: int = 0
    # Scheduled command: run a shell command every N iterations. Training pauses
    # while it runs. Each run gets its own timestamped log in eval_log_dir.
    # 0 = disabled.
    eval_command: str = ""
    eval_command_every: int = 0
    eval_log_dir: str = "logs/eval"
    # Log cosine angle change between current and initial LoRA weights every N steps. 0=off.
    log_weight_angle_every: int = 0
    group_size: int = 4
    beta: float = 0.04
    epsilon: float = 0.1
    epsilon_high: float | None = None
    loss_type: str = "grpo"  # grpo | bnpo | dr_grpo
    importance_sampling: str = "token"  # token | sequence | none
    save_every: int = 100
    eval_every: int = 50
    steps_per_report: int = 10
    gradient_accumulation_steps: int = 1
    # When True, accumulate all gradients across all sub-steps (sample_repeat or
    # samples_per_step) and apply a single optimizer update at the end of each
    # outer iteration. Overrides gradient_accumulation_steps for the duration.
    accumulate_over_step: bool = False
    seed: int = 0
    max_seq_length: int = 2048
    # Adapter config (applies to LoRA and DoRA modes, ignored for full)
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.v_proj",
        ]
    )
    lora_layers: int | None = None  # None = all layers
    # Adapter merging: fuse these checkpoint dirs into the base (in order) before training.
    # CLI --merge-adapters overrides. Each path must contain adapters.safetensors.
    merge_adapters: list[str] = field(default_factory=list)
    # Dataset options
    shuffle_data: bool = True
    shuffle_seed: int | None = None  # None = random
    balanced_shuffle: bool = True
    # When True and batch_size > 1, each batch contains samples of the same type.
    # Type rotation is balanced (round-robin). balanced_shuffle and other settings
    # still control the within-type ordering.
    batches_per_type: bool = False
    # Crash recovery
    auto_resume_on_crash: bool = True
    max_crash_retries: int = 3
    crash_cooldown_seconds: int = 10
    # Checkpoint management
    keep_last_n_checkpoints: int = 5
    keep_best_n_checkpoints: int = 3
    # Adaptive sampling
    adaptive_sampling: bool = False
    adaptive_zero_threshold: int = 5
    adaptive_perfect_threshold: int = 8
    adaptive_retry_interval: int = 50
    # Gradient checkpointing
    grad_checkpoint: bool = True  # Trade compute for memory on transformer layers
    # Metal GPU timeout safety — truncate sequences over this length in gradient step
    max_grad_seq_len: int = 1024
    # Prompt length estimate for metal_budget watchdog check. Default 900 covers
    # Qwen-Thinking chat template (verbose). Set to ~150-212 for exam/MCQ types.
    metal_est_prompt_tokens: int = 900
    # GPU cooldown (seconds) between per-sequence gradient computations
    gpu_cooldown: float = 1.0
    # Cooldown between rollout generations (seconds)
    rollout_cooldown: float = 0.5
    # Parallel rollout workers. 1 = serial (default). N > 1 = N threads; GPU ops
    # serialized via _ROLLOUT_LOCK; cooldown sleeps and reward computation overlap.
    rollout_workers: int = 1
    # "thread" = shared model, serialized GPU (default). "process" = one model copy per worker.
    rollout_worker_mode: str = "thread"
    # Reset optimizer momentum every N iterations. 0 = disabled.
    optimizer_reset_every: int = 0
    # Cooldown between generation and gradient phases (seconds)
    phase_cooldown: float = 2.0
    # Scaffold curriculum
    scaffold_end_progress: float = 0.8  # Progress at which scaffold reaches 0
    scaffold_initial_ratio: float = 1.0  # Starting ratio at progress=0
    scaffold_decay_rate: float = 1.0  # Multiplier on progress (>1 = faster decay)
    scaffold_target: str = "both"  # thinking | answer | both
    # Scaffold probe (eval at fixed scaffold ratios)
    scaffold_probe_ratios: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    scaffold_probe_samples: int = 20
    # Adaptive scaffold boundary discovery: Thompson Sampling + curve fitting
    optimize_boundaries: bool = False  # Enable/disable boundary optimization
    boundary_exploration_samples: int = 50  # Linear sweep before Thompson Sampling
    boundary_difficulty_buckets: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    boundary_reward_threshold: float = 0.5  # Reward >= threshold is "effective"
    dynamic_boundaries: bool = True  # Tighten boundaries as training progresses
    boundary_tightening_rate: float = 0.3  # How much to tighten per 100 iterations
    # Optimizer & LR schedule (dict fields pass through from YAML)
    optimizer: dict[str, Any] = field(default_factory=dict)
    lr_schedule: dict[str, Any] = field(default_factory=dict)
    # Logging
    log_token_ids: bool = False  # Include raw token ID arrays in rollouts.jsonl segments
    verbose_grad: bool = False  # Print per-sequence truncation details during gradient step
    # Checkpoint options
    save_optimizer: bool = True  # Save optimizer state with each checkpoint
    # WandB
    wandb_project: str | None = None
    wandb_log_full_model: bool = False  # Upload full model weights to WandB (large!)
    # Vision-Language Model support
    vision: VisionConfig = field(default_factory=VisionConfig)
    # CKA drift probe — auto-wired in _cmd_train when set.
    # Run `mlx-grpo inspect model_path --feature probe_save --probe-base base.npz` first.
    cka_probe_base: str | None = None  # path to base_acts.npz; None = probe disabled
    cka_probe_prompt: str = ""  # prompt used for activation collection at eval time
    # Conditional rules: pattern/lambda-based multipliers for gradient and reward
    conditional_rules: tuple[ConditionalRuleConfig, ...] = field(default_factory=tuple)
    # Prompt masking strategy: "mask" (default) excludes prompt from loss; "no_mask" includes it
    # Overridable per type via type_overrides
    prompt_mask_strategy: str = "mask"  # mask | no_mask
    # PyTorch device mapping: "gpu" (default) keeps model on GPU for training speed
    # "auto" allows CPU offloading on OOM (slow but fits on 8GB), "cpu" for CPU-only inference
    pytorch_device_map: str = "gpu"  # gpu | auto | cpu
    # Curriculum prompt masking: when enabled, first pass includes prompt in loss,
    # repeats exclude it to prevent overfitting to repeated prompt phrasings
    mask_prompt_on_repeats: bool = True
    # Adaptive prompt masking: when enabled, mask prompt only if model fails.
    # If model gets correct answer, unmask and move to next sample.
    # Overrides mask_prompt_on_repeats when model performance is checked.
    adaptive_prompt_masking: bool = False
    # Logging configuration with file + console handlers
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Training mode: "grpo" (standard GRPO loss) or "gar" (Guided Activation Reinforcement)
    mode: str = "grpo"  # grpo | gar
    # GAR-specific: cosine angle dead-zone threshold in degrees
    gar_drift_threshold_deg: float = 5.0
    # GAR-specific: L2-normalize activations before angle computation
    gar_normalize: bool = True
    # Jaccard similarity token boosting — boost loss on matching target/generated
    jaccard_boost: JaccardBoostConfig = field(default_factory=JaccardBoostConfig)
    # Dynamic per-layer gradient scaling based on angle drift
    dynamic_gradient_scaling: DynamicGradientScalingConfig = field(
        default_factory=DynamicGradientScalingConfig
    )


@dataclass(slots=True, frozen=True)
class GenerationConfig:
    """Generation defaults (overridable per type)."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    backend: str = (
        "mlx"  # mlx or pytorch for training; mlx, pytorch, vllm, llamacpp for eval/rollouts
    )
    # Sampler defaults
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    # XTC (eXclusion-based Token Culling) sampler — MLX only
    xtc_probability: float = 0.0  # 0.0 = disabled
    xtc_threshold: float = 0.1
    min_tokens_to_keep: int = 1  # Min tokens kept after top-p/top-k/xtc
    # KV cache (MLX only)
    kv_bits: int | None = None
    kv_group_size: int = 64
    max_kv_size: int | None = None
    quantized_kv_start: int = 0  # Prefill tokens before KV quantization kicks in
    # Speculative decoding
    draft_model_path: str | None = None
    # Prefill step size (MLX only) — tokens per prefill chunk
    prefill_step_size: int = 512
    # Global system prompt default (overridden by type_overrides or sample metadata)
    system_prompt: str = ""
    # Loss token truncation — generate max_tokens but only use used_tokens for loss
    used_tokens: int | None = None  # None = use all generated tokens
    truncate_from: str = "prefix"  # prefix | postfix | middle
    truncate_marker: str = "..."  # Text inserted at truncation boundaries
    truncate_marker_masked: bool = True  # Whether marker tokens are excluded from loss
    snap_to_boundary: bool = False  # Snap truncation to sentence endings
    # Intra-group parallelism: 0=serial, >0=parallel when >= group_size
    pipeline_batch_size: int = 0


@dataclass(slots=True, frozen=True)
class GradientConfig:
    """Gradient-level training controls."""

    train_layers: str | None = None
    thinking_gradient_weight: float = 1.0
    answer_gradient_weight: float = 1.0
    max_grad_norm: float = 1.0  # Gradient clipping threshold (0 = disabled)
    sft_anchor_enabled: bool = False
    sft_anchor_lr_multiplier: float = 0.1
    gradient_alignment_mode: str = "none"  # none | interpolate | project
    gradient_alignment_weight: float = 0.5


@dataclass(slots=True, frozen=True)
class MonitorConfig:
    """Training health monitoring."""

    enabled: bool = True
    kl_warning: float = 0.04
    kl_critical: float = 0.08
    reward_warning: float = 0.3
    stop_on_critical: bool = False


@dataclass(slots=True, frozen=True)
class SpikeConfig:
    """Spike-guard — detect and mitigate sudden KL / loss spikes."""

    # Detection — absolute thresholds
    kl_threshold: float = 0.5  # KL above this → spike
    loss_threshold: float | None = None  # loss above this → spike (None = disabled)
    # Detection — delta from rolling-window mean
    kl_delta: float | None = None  # KL exceeds mean by this → spike (None = disabled)
    loss_delta: float | None = None  # loss exceeds mean by this → spike (None = disabled)
    window: int = 20  # rolling window size for baseline
    min_window: int = 5  # min steps before delta checks activate
    # Action on spike
    action: str = "skip"  # skip | revert | shutdown
    max_consecutive: int = 5  # consecutive spikes before forced shutdown
    # Logging
    log_file: str | None = None  # JSON-lines file for skipped samples


@dataclass(slots=True, frozen=True)
class DynamicViewsConfig:
    """Dynamic multi-view scaffold: N equal-fraction prefix views + 1 naked view.

    n_additional=1 → 2 total views (1 scaffold + naked).
    n_additional=3 → 4 total views (3 scaffold at 1/3, 2/3, 3/3 fractions + naked).

    completion options:
      model_closes — inject prefix scaffold, model must reach </think> itself.
      model_opens  — (future) model generates thinking, scaffold provides answer start.
    """

    n_additional: int = 1  # scaffold views beyond the naked view
    target: str = "thinking"  # thinking | answer | both
    completion: str = "model_closes"  # model_closes | model_opens
    max_scaffold_tokens: int = 512  # token cap on prefix injected per view
    mask_system_prompt: bool = False  # exclude system prompt tokens from loss (all views)
    mask_user_prompt: bool = False  # exclude user prompt tokens from loss (all views)
    system_prompt_masked_views: tuple[int, ...] = field(
        default_factory=tuple
    )  # view indices to mask system prompt (overrides global mask_system_prompt)


@dataclass(slots=True, frozen=True)
class TokenISConfig:
    """Token-level importance sampling — penalizes verbosity via gradient weighting."""

    mode: str = "none"  # none | token | segment | scaffold_proximity
    decay_base: float = 0.5  # token: w(t) = decay_base^(t/n), 1.0 = no decay
    length_exponent: float = 0.5  # token: ls = (ref_length/n)^exponent, 0.0 = no scale
    ref_length: int = 256  # token: reference completion length (tokens)
    thinking_weight: float = 0.5  # segment: weight for thinking-tagged tokens
    answer_weight: float = 2.0  # segment: answer token weight (uniform fallback)
    correct_answer_weight: float = (
        1.0  # multiply answer-token gradient when rollout reward >= answer_weight_threshold
    )
    incorrect_answer_weight: float = (
        1.0  # multiply answer-token gradient when rollout reward < answer_weight_threshold
    )
    answer_weight_threshold: float = 0.5  # reward cutoff separating correct from incorrect
    proximity_decay: float = 0.9  # scaffold_proximity: weight at dist=0 (near) when invert=True
    invert_proximity: bool = False  # True = far tokens get MORE gradient (ICL handles near tokens)
    proximity_scale: float = (
        1.0  # Multiply all proximity weights by this. >1.0 upweights above baseline.
    )
    naked_apply_scale: bool = (
        True  # apply proximity_scale to naked (no-scaffold) members; False = flat 1.0
    )
    proximity_member_split: float = (
        1.0  # fraction of scaffolded members using near-proximity (0.0=all far, 1.0=all near)
    )
    proximity_chunk_split: float = (
        1.0  # within a member: fraction of scaffold chunks using near-proximity (0.0=all far)
    )
    proximity_decay_far: float | None = (
        None  # proximity_decay override for far members; None = same as proximity_decay
    )


@dataclass(slots=True, frozen=True)
class ConditionalRuleConfig:
    """Pattern/lambda-based rule for gradient and reward multipliers.

    Rules can match via regex pattern or lambda condition, then apply
    per-segment gradient multiplier and per-result reward multiplier.
    """

    gradient: str  # float literal, lambda, or "ClassName.method"
    reward: str  # float literal, lambda, or "ClassName.method"
    pattern: str = ""  # regex or literal string (ignored if when is set)
    check: str = "regex"  # "regex" | "contains"
    when: str = ""  # lambda or "ClassName.method" (preferred over pattern)
    target_segments: str = "generated"  # "generated" | "scaffolded" | "both"

    def __post_init__(self) -> None:
        """Validate rule configuration."""
        if not self.pattern and not self.when:
            raise TypeError("ConditionalRuleConfig requires 'pattern' or 'when'")
        if self.check not in ("regex", "contains"):
            raise ValueError(f"check must be 'regex' or 'contains', got {self.check!r}")
        if self.target_segments not in ("generated", "scaffolded", "both"):
            raise ValueError(
                f"target_segments must be 'generated'|'scaffolded'|'both', "
                f"got {self.target_segments!r}"
            )


def build_conditional_rules(data: list[dict[str, Any]] | None) -> tuple[ConditionalRuleConfig, ...]:
    """Build ConditionalRuleConfig from YAML list of dicts.

    Args:
        data: List of rule dicts from YAML (or None).

    Returns:
        Tuple of ConditionalRuleConfig objects. Empty tuple if data is None.
    """
    if data is None:
        return ()
    if not isinstance(data, list):
        return ()
    rules = []
    for rule_dict in data:
        if isinstance(rule_dict, dict):
            rule = _build_dataclass(ConditionalRuleConfig, rule_dict)
            rules.append(rule)
    return tuple(rules)


def build_jaccard_boost_config(data: dict[str, Any] | None) -> JaccardBoostConfig:
    """Build JaccardBoostConfig from YAML dict.

    Args:
        data: Dict from YAML jaccard_boost section (or None).

    Returns:
        JaccardBoostConfig object. Defaults if data is None.
    """
    if data is None:
        return JaccardBoostConfig()
    if not isinstance(data, dict):
        return JaccardBoostConfig()
    return _build_dataclass(JaccardBoostConfig, data)


def build_logging_config(data: dict[str, Any] | None) -> LoggingConfig:
    """Build LoggingConfig from YAML dict.

    Args:
        data: Dict with console_level, file_level, log_file, log_dir (or None).

    Returns:
        LoggingConfig object with defaults if data is None.
    """
    if data is None or not isinstance(data, dict):
        return LoggingConfig()
    return _build_dataclass(LoggingConfig, data)


@dataclass(slots=True)
class Config:
    """Complete configuration — composed from sub-configs."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    dynamic_gradient_scaling: DynamicGradientScalingConfig | None = None
    moe: MoEConfig | None = None
    importance_sampling: TokenISConfig | None = None
    spike_guard: SpikeConfig | None = None
    dynamic_views: DynamicViewsConfig | None = None
    escalation: EscalationConfig | None = None
    datasets: dict[str, Any] = field(default_factory=dict)
    type_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _resolve_type_extends(
    overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve Docker Compose-style 'extends' chains in type_overrides.

    Each type may declare ``extends: <parent_type>`` to inherit all parent
    settings, then override specific fields on top.  Inheritance is transitive
    (A extends B extends C) and the 'extends' key is consumed (not forwarded).
    Circular references raise ValueError.

    Example YAML::

        type_overrides:
          math:
            thinking_budget: 512
          math_hard:
            extends: math          # inherits thinking_budget: 512
            thinking_budget: 1024  # overrides just this field
    """
    resolved: dict[str, dict[str, Any]] = {}

    def _resolve(name: str, chain: tuple[str, ...]) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name not in overrides:
            return {}
        if name in chain:
            raise ValueError(f"Circular extends in type_overrides: {' → '.join(chain)} → {name}")
        raw = dict(overrides[name])
        parent_name = raw.pop("extends", None)
        if parent_name:
            parent = _resolve(str(parent_name), chain + (name,))
            raw = {**parent, **raw}  # parent is base; child wins on conflict
        resolved[name] = raw
        return raw

    for type_name in overrides:
        _resolve(type_name, ())
    return resolved


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file.

    Handles:
    1. Parsing YAML
    2. Creating TrainingConfig, GenerationConfig, etc.
    3. Dynamically creating TaskTypes from ``types:`` section
    4. Registering custom rewards from ``rewards:`` section
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Parse optional vision config block
    vision_raw = raw.get("vision")
    vision_cfg = (
        _build_dataclass(VisionConfig, vision_raw) if vision_raw is not None else VisionConfig()
    )

    training_raw = dict(raw.get("training", {}))
    training_raw["vision"] = vision_cfg
    # Parse optional conditional_rules block (top-level key)
    training_raw["conditional_rules"] = build_conditional_rules(raw.get("conditional_rules"))
    # Parse optional jaccard_boost block (top-level key)
    training_raw["jaccard_boost"] = build_jaccard_boost_config(raw.get("jaccard_boost"))
    # Parse optional logging block (nested in training or top-level)
    logging_raw = raw.get("training", {}).get("logging") or raw.get("logging")
    training_raw["logging"] = build_logging_config(logging_raw)
    # Parse optional dynamic_gradient_scaling block (nested in training or top-level)
    dgs_raw = raw.get("training", {}).get("dynamic_gradient_scaling") or raw.get(
        "dynamic_gradient_scaling"
    )
    training_raw["dynamic_gradient_scaling"] = (
        _build_dataclass(DynamicGradientScalingConfig, dgs_raw)
        if dgs_raw is not None
        else DynamicGradientScalingConfig()
    )
    training_cfg = _build_dataclass(TrainingConfig, training_raw)

    # Build scaffold config: prefer top-level ``scaffold:`` key, fall back
    # to legacy ``training.scaffold_*`` flat fields for backward compat.
    scaffold_raw = raw.get("scaffold")
    if scaffold_raw is None:
        # Legacy: pull from training flat fields
        t = raw.get("training", {})
        legacy = {
            k.removeprefix("scaffold_"): v
            for k, v in t.items()
            if k.startswith("scaffold_")
            and k != "scaffold_probe_ratios"
            and k != "scaffold_probe_samples"
        }
        scaffold_cfg = build_scaffold_config(legacy if legacy else None)
    else:
        scaffold_cfg = build_scaffold_config(scaffold_raw)

    # Per-type config overrides (tune builtin knobs without replacing the type)
    type_overrides: dict[str, dict[str, Any]] = {}
    for name, overrides in raw.get("type_overrides", {}).items():
        if isinstance(overrides, dict):
            type_overrides[name] = overrides
    type_overrides = _resolve_type_extends(type_overrides)

    # Parse optional MoE config block
    moe_cfg: MoEConfig | None = None
    moe_raw = raw.get("moe")
    if moe_raw is not None:
        gate_cfg = _build_dataclass(GateConfig, moe_raw.get("gate", {}))
        routing_cfg = _build_dataclass(RoutingConfig, moe_raw.get("routing", {}))
        expert_types = tuple(
            _build_dataclass(ExpertTypeMappingConfig, et) for et in moe_raw.get("expert_types", [])
        )
        _top = {k: v for k, v in moe_raw.items() if k not in ("gate", "routing", "expert_types")}
        # Convert list lora_targets → tuple
        if "lora_targets" in _top and isinstance(_top["lora_targets"], list):
            _top["lora_targets"] = tuple(_top["lora_targets"])
        moe_cfg = MoEConfig(**_top, gate=gate_cfg, routing=routing_cfg, expert_types=expert_types)

    # Parse optional token-level importance sampling block
    is_cfg: TokenISConfig | None = None
    is_raw = raw.get("importance_sampling")
    if is_raw is not None:
        is_cfg = _build_dataclass(TokenISConfig, is_raw)

    # Parse optional spike guard block
    spike_cfg: SpikeConfig | None = None
    spike_raw = raw.get("spike_guard")
    if spike_raw is not None:
        spike_cfg = _build_dataclass(SpikeConfig, spike_raw)

    # Parse optional dynamic views block
    dv_cfg: DynamicViewsConfig | None = None
    dv_raw = raw.get("dynamic_views")
    if dv_raw is not None:
        dv_cfg = _build_dataclass(DynamicViewsConfig, dv_raw)

    # Parse optional escalation block
    esc_cfg: EscalationConfig | None = None
    esc_raw = raw.get("escalation")
    if esc_raw is not None:
        esc_cfg = build_escalation_config(esc_raw)

    # Extract dynamic_gradient_scaling for top-level convenience access
    dgs_cfg = training_cfg.dynamic_gradient_scaling

    config = Config(
        training=training_cfg,
        generation=_build_dataclass(GenerationConfig, raw.get("generation", {})),
        gradient=_build_dataclass(GradientConfig, raw.get("gradient", {})),
        monitor=_build_dataclass(MonitorConfig, raw.get("monitor", {})),
        scaffold=scaffold_cfg,
        dynamic_gradient_scaling=dgs_cfg,
        moe=moe_cfg,
        importance_sampling=is_cfg,
        spike_guard=spike_cfg,
        dynamic_views=dv_cfg,
        escalation=esc_cfg,
        datasets=raw.get("datasets", {}),
        type_overrides=type_overrides,
        raw=raw,
    )

    # Validate GAR mode constraints
    if training_cfg.mode == "gar":
        if training_cfg.group_size < 2:
            raise ValueError(
                f"GAR mode requires group_size >= 2 (got {training_cfg.group_size}). "
                "GAR needs at least one scaffolded + one naked rollout per group."
            )
        if training_cfg.lora_layers is None:
            raise ValueError(
                "GAR mode requires lora_layers to be set (not None). "
                "Specify which layers to capture activations from."
            )

    # Metal GPU time-budget checks (warns when config risks Metal watchdog / exit 134)
    from .metal_budget import check_escalation_metal_budget, check_metal_budget

    check_metal_budget(config)
    check_escalation_metal_budget(config)

    # Validate per_group_config constraints
    if scaffold_cfg.per_group_config is not None:
        pgc = scaffold_cfg.per_group_config
        gs = training_cfg.group_size
        if len(pgc.views) != gs:
            raise ValueError(
                f"scaffold.per_group_config.views has {len(pgc.views)} entries "
                f"but training.group_size={gs}; they must match"
            )
        sps = training_cfg.samples_per_step
        if sps != 1:
            raise ValueError(f"samples_per_step ({sps}) must be 1 when using per_group_config")

    # Register YAML-defined types
    for name, type_cfg in raw.get("types", {}).items():
        if isinstance(type_cfg, dict):
            create_yaml_type(name, type_cfg)

    # Register YAML-defined rewards
    for name, reward_def in raw.get("rewards", {}).items():
        if isinstance(reward_def, str):
            fn = parse_lambda(reward_def)
            register_lambda_reward(name, fn)

    return config


def _build_dataclass(cls, data: dict[str, Any]):
    """Build a frozen dataclass from a dict, ignoring unknown keys.

    Coerces values to match field types (e.g., YAML may parse '2e-6' as str).
    """
    fields = cls.__dataclass_fields__
    unknown = [k for k in data if k not in fields]
    if unknown:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "%s: ignoring unknown keys %s",
            cls.__name__,
            unknown,
        )
    filtered = {}
    for k, v in data.items():
        if k not in fields:
            continue
        # Coerce string values to float/int when the field expects a number
        if isinstance(v, str):
            field_type = fields[k].type
            if field_type in ("float", float) or "float" in str(field_type):
                with contextlib.suppress(ValueError):
                    v = float(v)
            elif field_type in ("int", int) or "int" in str(field_type):
                with contextlib.suppress(ValueError):
                    v = int(v)
        filtered[k] = v
    return cls(**filtered)


def merge_configs(*paths: str | Path) -> Config:
    """Merge multiple YAML configs (later files override earlier)."""
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(merged, data)
    # Write merged to temp and load
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(merged, f)
        return load_config(f.name)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: pip install pyyaml
# Usage: config = load_config("config.yml")
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
