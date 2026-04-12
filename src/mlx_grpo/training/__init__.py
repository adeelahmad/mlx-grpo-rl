"""GRPO Training Loop — the orchestrator that ties everything together.

Phases per step:
  1. Sample from dataset
  2. Generate G rollouts (the "group") via executor
  3. Score each rollout with reward function
  4. Compute GRPO advantages (group-relative normalization)
  5. Forward pass through model → loss → backward → update

Phase 5 is the critical part. It MUST run the forward pass inside
MLX's value_and_grad so the computation graph is traced. The
mlx_ops module handles this correctly.
"""

from __future__ import annotations

import signal
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlx_grpo.config.loader import GenerationConfig, TrainingConfig
from mlx_grpo.data import DatasetLoader, Sample
from mlx_grpo.engine.callbacks import CallbackRegistry
from mlx_grpo.engine.context import BatchContext
from mlx_grpo.engine.executor import RolloutExecutor
from mlx_grpo.engine.result import RolloutResult
from mlx_grpo.rewards.base import BaseReward
from mlx_grpo.scaffold import ScaffoldConfig, build_scaffold_config, get_view_scaffold_ratio
from mlx_grpo.types.base import TaskType

from .parallel import (
    _BOUNDARY_OPTIMIZER_LOCK,
    _PROMPT_MASK_STATE_LOCK,
    _ROLLOUT_LOCK,
    dispatch_parallel,
)


def _extract_prefix_lines(text: str, n: int) -> str | None:
    """Extract first n non-empty lines from a thinking trace.

    Strips the leading <think> token (the model's generation prompt already
    contains it). Returns None if the text has no usable lines.
    """
    t = text.lstrip()
    if t.startswith("<think>"):
        t = t[len("<think>") :]
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if not lines:
        return None
    return "\n".join(lines[:n]) + "\n"


def _prefix_from_thinking(text: str, n_tokens: int, backend: Any) -> str:
    """Extract first n_tokens of the thinking content (before </think>) for escalation prefix.

    Stops at </think> so the injected prefix never includes the closing tag or the boxed
    answer — which would cause the model to generate tokens AFTER the answer and never
    produce a fresh </think> stop sequence.
    """
    t = text.lstrip()
    if t.startswith("<think>"):
        t = t[len("<think>") :]
    # Only use the thinking content — stop before </think>
    think_end = t.find("</think>")
    if think_end != -1:
        t = t[:think_end]
    tokens = backend.encode(t)
    if n_tokens <= 0 or not tokens:
        return ""
    return backend.decode(tokens[: min(n_tokens, len(tokens))])


def _metal_clear_cache() -> None:
    """Release Metal GPU buffer pool between heavy operations.

    Frees unused Metal buffers to prevent macOS's interactivity watchdog
    timeout (kIOGPUCommandBufferCallbackErrorImpactingInteractivity).

    Note: gc.collect() was intentionally removed — MLX arrays are reference-
    counted and freed when they go out of scope. gc.collect() only helps with
    cycles and costs ~100ms per call, adding 500ms+ overhead per iteration.
    """
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def _metal_full_reset(model: Any = None) -> None:
    """Full Metal sync + cache clear for use between sample-repeat reps.

    Unlike _metal_clear_cache() which only frees the buffer pool, this
    evaluates the actual model parameters to drain the gradient step's
    lazy computation graph from Metal's command queue, then clears cache.

    mx.eval(mx.array(0)) is NOT sufficient: it syncs a trivial constant
    but leaves pending parameter updates in-flight. The next generation
    then collides with those in-flight buffers and hits the 8s watchdog.
    """
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        if model is not None:
            # Force-evaluate all model parameters — drains gradient step residuals
            params = [v for _, v in tree_flatten(model.parameters()) if isinstance(v, mx.array)]
            if params:
                mx.eval(*params)
        else:
            mx.eval(mx.array(0))
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def _collect_memory() -> tuple[float, float]:
    """Return (active_metal_mb, rss_mb). Non-MLX platforms return (0, rss)."""
    import resource

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    active_mb = 0.0
    try:
        import mlx.core as mx

        active_mb = mx.metal.get_active_memory() / (1024 * 1024)
    except (ImportError, AttributeError):
        pass
    return active_mb, rss_mb


def _snapshot_trainable_weights(model: Any) -> dict[str, Any]:
    """Snapshot trainable weights for angle computation (MLX or PyTorch)."""
    try:
        # Try MLX first
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten

        snap = {}
        for k, v in tree_flatten(model.trainable_parameters()):
            # Force true copy via numpy (MLX arrays are otherwise lazy/shared)
            v_f32 = v.astype(mx.float32)
            mx.eval(v_f32)
            # Convert to numpy and back — guarantees independent copy
            np_array = np.array(v_f32)
            snap[k] = mx.array(np_array)
        mx.eval(*snap.values())
        return snap
    except (ImportError, AttributeError):
        # Fall back to PyTorch
        try:
            snap = {}
            for k, v in model.named_parameters():
                if v.requires_grad:
                    snap[k] = v.detach().float().clone()
            return snap
        except Exception:
            return {}


def _compute_weight_angles(model: Any, baseline: dict[str, Any]) -> dict[str, float]:
    """Cosine angle (degrees) between current and initial trainable weights, grouped by layer.

    Works for both MLX and PyTorch backends. Returns {layer_key: degrees} or {} on failure.
    """
    try:
        import math

        # Detect backend by checking model type, not just whether MLX is installed
        is_mlx = hasattr(model, "parameters") is False and hasattr(model, "state")
        # PyTorch models have .parameters() method; MLX models have .state attribute
        # Better check: look for torch.nn.Module or mlx.nn.Module
        try:
            import torch

            is_torch = isinstance(model, torch.nn.Module)
        except (ImportError, TypeError):
            is_torch = False

        if is_torch or (not is_mlx and is_torch):
            return _compute_weight_angles_torch(model, baseline, math)

        # Try MLX if not PyTorch (and MLX is available)
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            return _compute_weight_angles_mlx(model, baseline, math, mx, tree_flatten)
        except (ImportError, AttributeError):
            pass

        return {}

    except Exception as e:
        import logging

        logging.getLogger(__name__).debug("Failed to compute weight angles: %s", e)
        return {}


def _compute_weight_angles_mlx(
    model: Any, baseline: dict[str, Any], math: Any, mx: Any, tree_flatten: Any
) -> dict[str, float]:
    """MLX implementation of weight angle computation."""
    layer_ops: dict[str, list[tuple[Any, Any, Any]]] = {}
    for k, v in tree_flatten(model.trainable_parameters()):
        if k not in baseline:
            continue
        cur = v.astype(mx.float32).reshape(-1)
        ini = baseline[k].reshape(-1)
        if cur.shape != ini.shape:
            continue
        parts = k.split(".")
        # "layers.20" as the key — concise, matches layer numbering
        layer_key = parts[2] if len(parts) > 2 and parts[1] == "layers" else ".".join(parts[:3])
        layer_ops.setdefault(layer_key, []).append(
            (mx.sum(cur * ini), mx.sum(cur * cur), mx.sum(ini * ini))
        )

    if not layer_ops:
        return {}

    flat = [x for triples in layer_ops.values() for triple in triples for x in triple]
    mx.eval(*flat)

    result: dict[str, float] = {}
    for lk, triples in layer_ops.items():
        tot_dot = sum(float(d) for d, _, _ in triples)
        tot_nc = math.sqrt(sum(float(nc) for _, nc, _ in triples))
        tot_ni = math.sqrt(sum(float(ni) for _, _, ni in triples))
        if tot_nc < 1e-12 or tot_ni < 1e-12:
            continue
        cos = max(-1.0, min(1.0, tot_dot / (tot_nc * tot_ni)))
        result[lk] = math.degrees(math.acos(cos))
    return result


def _compute_weight_angles_torch(
    model: Any, baseline: dict[str, Any], math: Any
) -> dict[str, float]:
    """PyTorch implementation of weight angle computation."""
    import torch

    layer_ops: dict[str, list[tuple[float, float, float]]] = {}
    for k, v in model.named_parameters():
        if k not in baseline or not v.requires_grad:
            continue
        cur = v.detach().float().reshape(-1)
        ini = baseline[k].reshape(-1)
        if cur.shape != ini.shape:
            continue
        parts = k.split(".")
        # "layers.20" as the key — concise, matches layer numbering
        layer_key = parts[2] if len(parts) > 2 and parts[1] == "layers" else ".".join(parts[:3])

        dot = float(torch.dot(cur, ini).item())
        norm_cur = float(torch.norm(cur).item()) ** 2
        norm_ini = float(torch.norm(ini).item()) ** 2
        layer_ops.setdefault(layer_key, []).append((dot, norm_cur, norm_ini))

    if not layer_ops:
        return {}

    result: dict[str, float] = {}
    for lk, triples in layer_ops.items():
        tot_dot = sum(d for d, _, _ in triples)
        tot_nc = math.sqrt(sum(nc for _, nc, _ in triples))
        tot_ni = math.sqrt(sum(ni for _, _, ni in triples))
        if tot_nc < 1e-12 or tot_ni < 1e-12:
            continue
        cos = max(-1.0, min(1.0, tot_dot / (tot_nc * tot_ni)))
        result[lk] = math.degrees(math.acos(cos))
    return result


# ── Metrics ──


@dataclass
class StepMetrics:
    """Metrics from a single training step."""

    iteration: float
    loss: float = 0.0
    mean_reward: float = 0.0
    reward_std: float = 0.0
    kl_divergence: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    generated_tokens: int = 0
    elapsed_sec: float = 0.0
    generation_sec: float = 0.0
    peak_memory_mb: float = 0.0
    rss_mb: float = 0.0
    learning_rate: float = 0.0
    rewards: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleGroupResult:
    """Result from processing one sample's rollout group inside _run_sample_group."""

    group_rollouts: list  # list[RolloutResult]
    rewards: list  # list[float]
    advantages: list  # list[float]
    mean_reward: float
    prompts: list  # list[str]
    answers: list  # list[str]
    targets: list  # list[str]
    pending_events: list  # list[tuple[str, StepMetrics]] — replayed in main thread
    tui_ctx: tuple  # (merged_type_config, reward_fn, prompt_tokens, callbacks)
    generation_sec: float = 0.0  # Wall-clock time spent on model generation only


# ── Hooks ──

StepHook = Callable[["Trainer", StepMetrics], None]


# ── Trainer ──


class Trainer:
    """GRPO training loop orchestrator.

    Two modes:
      1. Full training (model + ref_model + optimizer provided):
         Forward pass → GRPO loss → backward → weight update
      2. Rollout-only (no model): Generates rollouts and computes
         rewards, useful for testing/eval/reward debugging
    """

    def __init__(
        self,
        config: TrainingConfig,
        dataset: DatasetLoader,
        backend: Any,
        model: Any = None,
        ref_model: Any = None,
        tokenizer: Any = None,
        optimizer: Any = None,
        max_completion_tokens: int = 0,
        scaffold: ScaffoldConfig | None = None,
        max_grad_norm: float = 1.0,
        used_tokens: int | None = None,
        truncate_from: str = "prefix",
        truncate_marker: str = "...",
        truncate_marker_masked: bool = True,
        snap_to_boundary: bool = False,
        type_overrides: dict[str, dict[str, Any]] | None = None,
        system_prompt: str = "",
        is_cfg: Any = None,
        dv_cfg: Any = None,
        esc_cfg: Any = None,
        gen_defaults: dict[str, Any] | None = None,
        generation_config: GenerationConfig | None = None,
    ):
        self.config = config
        self.dataset = dataset
        self.backend = backend
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self._gen_defaults = gen_defaults or {}
        self._gen_cfg = generation_config or GenerationConfig()
        self.executor = (
            RolloutExecutor(backend, gen_defaults=gen_defaults) if backend is not None else None
        )

        # Inject tokenizer into backend for deferred initialization
        if (
            self.tokenizer is not None
            and self.backend is not None
            and hasattr(self.backend, "_tokenizer")
        ):
            self.backend._tokenizer = self.tokenizer
        self.max_completion_tokens = max_completion_tokens
        self._global_scaffold = scaffold or ScaffoldConfig()
        self._max_grad_norm = max_grad_norm
        self._used_tokens = used_tokens
        self._truncate_from = truncate_from
        self._truncate_marker = truncate_marker
        self._truncate_marker_masked = truncate_marker_masked
        self._snap_to_boundary = snap_to_boundary
        self._marker_ids: dict[str, list[int]] = {}
        self._type_overrides = type_overrides or {}
        self._system_prompt = system_prompt
        self._is_cfg = is_cfg
        self._dv_cfg = dv_cfg
        self._esc_cfg = esc_cfg

        # Gradient ops — created lazily when model is available
        self._mlx_ops_available = False
        self._torch_ops_available = False
        self._gar_mode = getattr(config, "mode", "grpo") == "gar"
        self._torch_device = "cpu"
        self._accum_grads: Any = None
        self._accum_count: int = 0
        self._accum_count_this_iter: int = 0  # Tracks steps accumulated in this iteration
        self._step_accum_mode: bool = False
        self._setup_gradient_ops()

        self._hooks: dict[str, list[StepHook]] = {
            "before_step": [],
            "after_step": [],
            "before_rollout": [],
            "after_rollout": [],
            "on_group": [],
            "on_gradient": [],
            "on_rollout_start": [],
            "on_checkpoint": [],
            "on_eval": [],
        }
        self._iteration = 0
        self._best_reward = float("-inf")
        self._recent_losses: deque[float] = deque(maxlen=10)
        self._recent_rewards: deque[float] = deque(maxlen=10)
        self._reward_baseline: float = 0.5  # EMA baseline for uniform-group fallback
        self.stop_requested: bool = False  # Cooperative stop for TUI graceful shutdown

        # GAR mode state
        self._gar_last_metrics: dict[str, Any] = {}

        # Optional: spike guard (set by caller after construction via set_spike_guard)
        self._spike_guard: Any = None

        # Optional: adaptive sampling
        self._adaptive: Any = None
        if getattr(self.config, "adaptive_sampling", False):
            from ..data.adaptive_sampler import AdaptiveSampler

            self._adaptive = AdaptiveSampler(
                dataset=self.dataset,
                zero_threshold=getattr(self.config, "adaptive_zero_threshold", 5),
                perfect_threshold=getattr(self.config, "adaptive_perfect_threshold", 8),
                retry_interval=getattr(self.config, "adaptive_retry_interval", 50),
            )

        # Optional: process-based rollout workers — assigned externally by CLI in process mode
        self._process_pool: Any = None

        # VLM: load processor and freeze vision layers when vision config is active
        self._processor: Any = None
        _vision_cfg = getattr(config, "vision", None)
        if _vision_cfg is not None and _vision_cfg.is_enabled():
            from ..training.vlm_loader import freeze_vision_layers, load_processor

            self._processor = load_processor(_vision_cfg)
            if model is not None:
                freeze_vision_layers(model, _vision_cfg)

        # Optional: conditional rules engine for pattern/lambda-based multipliers
        _conditional_rules = getattr(config, "conditional_rules", ())
        if _conditional_rules:
            from ..engine.conditional_rules import ConditionalRulesEngine

            self._conditional_rules_engine: Any = ConditionalRulesEngine(_conditional_rules)
        else:
            self._conditional_rules_engine = None

        # Optional: scaffold boundary optimization
        self._boundary_optimizer: Any = None
        self._difficulty_estimator: Any = None
        if getattr(self.config, "optimize_boundaries", False):
            from ..training.prompt_difficulty_estimator import PromptDifficultyEstimator
            from ..training.scaffold_boundary_optimizer import ScaffoldBoundaryOptimizer

            self._boundary_optimizer = ScaffoldBoundaryOptimizer(
                n_exploration_samples=getattr(self.config, "boundary_exploration_samples", 50),
                difficulty_buckets=getattr(
                    self.config, "boundary_difficulty_buckets", [0.25, 0.5, 0.75]
                ),
                reward_threshold=getattr(self.config, "boundary_reward_threshold", 0.5),
            )
            self._difficulty_estimator = PromptDifficultyEstimator()

        # Force answer flag for repeat-training
        self._force_answer_this_rep: bool = False
        # Current repeat index for curriculum masking (0 = first pass, 1+ = repeats)
        self._current_repeat_index: int = 0

        # Adaptive diverse scaffold region params (step 5)
        self._scaffold_region_offset_chars: int = 0
        self._scaffold_region_length_chars: int = 0
        self._scaffold_chunk_count_override: int = 0
        self._scaffold_is_fallback: bool = False
        self._scaffold_coverage_state: Any = None
        self._scaffold_last_dataset_row: int = -1

    def _setup_gradient_ops(self) -> None:  # pragma: no cover
        """Check if MLX or PyTorch gradient ops are available.

        Backend-aware: if the backend is 'pytorch', skip MLX entirely
        (even if MLX is installed) to avoid the MLX path claiming ops
        when CUDA training is intended.
        """
        if self.model is not None and self.optimizer is not None:
            _backend_name = self.backend.name if self.backend else None

            # Try MLX first — but ONLY if the backend is MLX (not pytorch)
            if _backend_name != "pytorch":
                try:
                    from mlx_grpo.loss.mlx_ops import (
                        _tree_add,
                        _tree_scale,
                        apply_grads,
                        compute_grads,
                        extract_sequences_from_rollouts,
                        gradient_step,
                        sft_gradient_step,
                    )

                    self._gradient_step_fn = gradient_step
                    self._compute_grads_fn = compute_grads
                    self._apply_grads_fn = apply_grads
                    self._extract_fn = extract_sequences_from_rollouts
                    self._tree_add = _tree_add
                    self._tree_scale = _tree_scale
                    self._sft_step_fn = sft_gradient_step
                    self._mlx_ops_available = True
                    self._accum_grads = None
                    self._accum_count = 0
                    return
                except ImportError:
                    self._mlx_ops_available = False

            # Try PyTorch
            try:
                from mlx_grpo.loss.mlx_ops import extract_sequences_from_rollouts
                from mlx_grpo.loss.torch_ops import (
                    apply_grads as torch_apply_grads,
                )
                from mlx_grpo.loss.torch_ops import (
                    compute_grads as torch_compute_grads,
                )
                from mlx_grpo.loss.torch_ops import (
                    gradient_step as torch_gradient_step,
                )
                from mlx_grpo.loss.torch_ops import (
                    sft_gradient_step as torch_sft_step,
                )

                self._torch_gradient_step_fn = torch_gradient_step
                self._torch_compute_grads_fn = torch_compute_grads
                self._torch_apply_grads_fn = torch_apply_grads
                self._torch_sft_step_fn = torch_sft_step
                self._extract_fn = extract_sequences_from_rollouts
                self._torch_ops_available = True
                self._torch_accum_count = 0
                # Detect device from model
                self._torch_device = str(next(self.model.parameters()).device)
                # CUDA/ROCm: disable Metal rollout lock (allows parallel generation)
                if self._torch_device.startswith("cuda"):
                    from .parallel import set_rollout_lock_enabled

                    set_rollout_lock_enabled(False)
            except Exception as _e:
                self._torch_ops_available = False
                with open("/tmp/_grpo_grad_debug.log", "a") as _f:
                    import traceback

                    _f.write(f"TORCH OPS IMPORT FAILED: {type(_e).__name__}: {_e}\n")
                    traceback.print_exc(file=_f)

    # ── System prompt resolution ──

    def _resolve_system_prompt(self, sample: Sample) -> str | None:
        """Resolve system_prompt with explicit-only priority chain.

        Priority: type_overrides > sample metadata > None.
        Returns None when no system prompt should be applied.

        Note: TaskType.chat_params().system_prompt and global self._system_prompt
        are intentionally excluded to ensure system prompts are explicit in YAML
        or JSONL, not implicit defaults.
        """
        overrides = self._type_overrides.get(sample.type_name) or self._type_overrides.get(
            TaskType._aliases.get(sample.type_name, sample.type_name), {}
        )
        return overrides.get("system_prompt") or sample.metadata.get("system_prompt") or None

    # ── Hook system ──

    def on(self, event: str, hook: StepHook) -> Trainer:
        """Register a hook. Returns self for chaining."""
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}. Valid: {list(self._hooks)}")
        self._hooks[event].append(hook)
        return self

    def _fire(self, event: str, metrics: StepMetrics) -> None:
        for hook in self._hooks.get(event, []):
            hook(self, metrics)

    def set_spike_guard(self, guard: Any) -> None:
        """Attach a SpikeGuard instance to intercept spikes after gradient steps."""
        self._spike_guard = guard

    # ── Main loop ──

    def train(self, resume_from: int | float = 0) -> None:
        """Run training loop."""
        resume_from = int(resume_from)  # float if resumed from a sub-step
        self._iteration = resume_from
        self._last_metrics: StepMetrics | None = None
        consecutive_failures = 0

        # Snapshot initial trainable weights for angle-drift logging
        # NOTE: Weight angle logging only works for MLX backend (uses mx operations)
        self._weight_angle_baseline: dict[str, Any] | None = None
        self._last_angle_avg: float | None = None
        self._baseline_snapshot_iter: int | None = None  # Track when baseline was taken
        _wa_every = getattr(self.config, "log_weight_angle_every", 0)
        if _wa_every > 0 and self.model is not None:
            self._weight_angle_baseline = _snapshot_trainable_weights(self.model)
            self._baseline_snapshot_iter = resume_from
            print(f"[angle_baseline] Snapshot taken at resume_from={resume_from}")

        # Register signal handlers for graceful shutdown
        def _signal_handler(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            print(f"\n  Received signal: {sig_name}")
            self.stop_requested = True

        original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
        original_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            for iteration in range(resume_from, self.config.iterations):
                if self.stop_requested:
                    self._graceful_shutdown(sig_name="STOP REQUESTED (SIGTERM)")
                    break
                self._iteration = iteration
                t0 = time.perf_counter()

                _post_step_done = False
                try:
                    sps = getattr(self.config, "samples_per_step", 1)
                    sr = getattr(self.config, "sample_repeat", 1)
                    if sr > 1 or sr < 0:
                        # Each repeat is a complete step: train + all post-step hooks.
                        # -1: until any non-forced member correct; -2: until naked correct
                        _sr_batch = self._sample_batch()
                        _sr_type = _sr_batch[0].type_name if _sr_batch else ""
                        _sr_ov = self._type_overrides.get(_sr_type, {})
                        sr = _sr_ov.get("sample_repeat", sr)
                        _sr_threshold = _sr_ov.get(
                            "sample_repeat_threshold",
                            getattr(self.config, "sample_repeat_threshold", 1.0),
                        )
                        _sr_max = _sr_ov.get(
                            "sample_repeat_max",
                            getattr(self.config, "sample_repeat_max", 20),
                        )
                        _sr_kl_thresh = _sr_ov.get(
                            "sample_repeat_kl_threshold",
                            getattr(self.config, "sample_repeat_kl_threshold", 0.08),
                        )
                        _sr_base = iteration  # integer base — shared across all reps
                        _sr_dataset_row = getattr(self.dataset, "_index", -1)
                        _sr_rep = 0
                        _sr_unforced_rep = 0  # counts only non-forced reps (for limits/stops)
                        _sr_reason = "max_reached"
                        _fa_thresh = getattr(
                            self.config,
                            "force_answer_on_every_other_repeat_after_repeat_number",
                            0,
                        )
                        _adaptive_masking = getattr(self.config, "adaptive_prompt_masking", False)
                        _model_succeeded = False  # Track if model got answer right
                        # Naked repeat phases: zero-scaffold reps before/after normal
                        _naked_before = _sr_ov.get(
                            "before_repeat_naked_repeats",
                            getattr(self.config, "before_repeat_naked_repeats", 0),
                        )
                        _naked_after = _sr_ov.get(
                            "after_repeat_naked_repeats",
                            getattr(self.config, "after_repeat_naked_repeats", 0),
                        )
                        self._in_naked_after_phase = False
                        self._naked_before = _naked_before
                        self._naked_after = _naked_after
                        _sr_after_count = 0
                        # Set up step-level accumulation if enabled
                        _accum_over_step = getattr(self.config, "accumulate_over_step", False)
                        if _accum_over_step:
                            self._step_accum_mode = True
                            self._accum_grads = None
                            self._accum_count = 0
                            self._accum_count_this_iter = 0  # Track accumulated steps this iter
                        while True:
                            _rep_t0 = time.perf_counter()
                            # Determine if this is a forced-answer repeat
                            _is_force_rep = (
                                _fa_thresh > 0
                                and _sr_rep >= _fa_thresh
                                and (_sr_rep - _fa_thresh) % 2 == 0
                            )
                            self._force_answer_this_rep = _is_force_rep
                            # Curriculum masking: first pass (naked) excludes prompt mask,
                            # repeats (2nd+) apply prompt mask to prevent overfitting
                            self._current_repeat_index = _sr_unforced_rep
                            # Adaptive diverse scaffold region plan
                            _esc_div = (
                                getattr(getattr(self._esc_cfg, "diverse", None), "enabled", False)
                                if self._esc_cfg
                                else False
                            )
                            if _esc_div:
                                from .scaffold_region_planner import (
                                    CoverageState,
                                    compute_region_plan,
                                    extract_thinking_text,
                                    scale_chunks_for_density,
                                )

                                # Get current sample row ID
                                _cur_row = (
                                    _sr_batch[0].metadata.get("_row_index", -1) if _sr_batch else -1
                                )
                                # Reset coverage state on new sample or if not yet initialized
                                if (
                                    _cur_row != self._scaffold_last_dataset_row
                                    or self._scaffold_coverage_state is None
                                ):
                                    _th = (
                                        extract_thinking_text(_sr_batch[0].answer or "")
                                        if _sr_batch
                                        else ""
                                    )
                                    self._scaffold_coverage_state = CoverageState(
                                        thinking_text_len=len(_th),
                                        covered_ranges=[],
                                        repeat_count=0,
                                    )
                                    self._scaffold_last_dataset_row = _cur_row
                                # Compute region plan for this repeat
                                _div_cfg = self._esc_cfg.diverse
                                _plan = compute_region_plan(
                                    thinking_len=self._scaffold_coverage_state.thinking_text_len,
                                    repeat_index=_sr_unforced_rep,
                                    num_regions=_div_cfg.num_regions,
                                    density_decay=_div_cfg.density_decay,
                                    coverage_threshold=_div_cfg.coverage_threshold,
                                    state=self._scaffold_coverage_state,
                                )
                                self._scaffold_region_offset_chars = _plan.offset_chars
                                self._scaffold_region_length_chars = _plan.length_chars
                                self._scaffold_is_fallback = _plan.is_fallback
                                # Diminishing density: compute chunk override
                                _horiz = getattr(self._esc_cfg, "horizontal", None)
                                if _horiz is not None:
                                    _base = getattr(_horiz, "max_members", 4)
                                    self._scaffold_chunk_count_override = scale_chunks_for_density(
                                        _base,
                                        _sr_unforced_rep,
                                        _div_cfg.density_decay,
                                        _div_cfg.min_chunks,
                                    )
                            else:
                                # Diverse disabled: reset all to defaults
                                self._scaffold_region_offset_chars = 0
                                self._scaffold_region_length_chars = 0
                                self._scaffold_chunk_count_override = 0
                                self._scaffold_is_fallback = False
                            # Show "rep N in progress" on sub-bar before the rollout starts
                            if hasattr(self, "_monitor") and self._monitor is not None:
                                self._monitor.mark_repeat_start(_sr_rep + 1, sr)
                            metrics = self._train_step(_sr_base, batch=_sr_batch)
                            _sr_rep += 1
                            if not _is_force_rep:
                                _sr_unforced_rep += 1
                            # Fractional step number: 1421.1, 1421.2, …
                            _step_iter = _sr_base + round(_sr_rep * 0.1, 10)
                            metrics.iteration = _step_iter
                            self._iteration = _step_iter
                            # Update coverage state after this repeat completes
                            _esc_div = (
                                getattr(getattr(self._esc_cfg, "diverse", None), "enabled", False)
                                if self._esc_cfg
                                else False
                            )
                            if _esc_div and self._scaffold_coverage_state is not None:
                                from .scaffold_region_planner import update_coverage

                                _off = self._scaffold_region_offset_chars
                                _len = (
                                    self._scaffold_region_length_chars
                                    or self._scaffold_coverage_state.thinking_text_len
                                )
                                update_coverage(self._scaffold_coverage_state, _off, _len)
                                self._scaffold_coverage_state.repeat_count += 1
                            # Check stop conditions (phase-aware for naked repeats)
                            _sr_done = False
                            _in_before_naked = (
                                _naked_before > 0 and _sr_unforced_rep <= _naked_before
                            )
                            if self._in_naked_after_phase:
                                # After-naked phase: count reps, stop when done
                                _sr_after_count += 1
                                if _sr_after_count >= _naked_after:
                                    _sr_done, _sr_reason = True, "after_naked_done"
                            elif _in_before_naked:
                                # Before-naked phase: skip normal stop conditions
                                pass
                            elif sr > 1 and _sr_unforced_rep >= _naked_before + sr:
                                _sr_done, _sr_reason = True, "fixed"
                            elif sr in (-1, -2) and _sr_rep >= _sr_max + _naked_before:
                                _sr_done, _sr_reason = True, "max_reached"
                            elif sr == -1 and not _is_force_rep:
                                # Stop when any non-forced member is genuinely correct.
                                _rollouts = metrics.extras.get("rollouts", [])
                                if _rollouts and any(
                                    r.metadata.get("_esc_reward", r.metadata.get("_reward", -1.0))
                                    >= _sr_threshold
                                    for r in _rollouts
                                    if not r.metadata.get("_view_is_naked")
                                    and not r.metadata.get("_esc_forced_answer")
                                ):
                                    _sr_done, _sr_reason = True, "non_forced_correct"
                                elif not _rollouts:
                                    _vs = metrics.extras.get("view_stats", {})
                                    if any(
                                        max(v["rewards"], default=0.0) >= _sr_threshold
                                        for vn, v in _vs.items()
                                        if not vn.startswith("forced")
                                    ):
                                        _sr_done, _sr_reason = True, "non_forced_correct"
                            elif sr == -2 and not _is_force_rep:
                                # Stop when naked rollout is genuinely correct.
                                _rollouts = metrics.extras.get("rollouts", [])
                                if _rollouts and any(
                                    r.metadata.get("_esc_reward", r.metadata.get("_reward", -1.0))
                                    >= _sr_threshold
                                    for r in _rollouts
                                    if r.metadata.get("_view_is_naked")
                                ):
                                    _sr_done, _sr_reason = True, "naked_correct"
                                elif not _rollouts:
                                    _vs = metrics.extras.get("view_stats", {})
                                    if any(
                                        max(v["rewards"], default=0.0) >= _sr_threshold
                                        for v in _vs.values()
                                        if v.get("is_naked")
                                    ):
                                        _sr_done, _sr_reason = True, "naked_correct"
                            # Transition to after-naked phase instead of stopping
                            if (
                                _sr_done
                                and _naked_after > 0
                                and not self._in_naked_after_phase
                                and _sr_reason != "kl_break"
                            ):
                                self._in_naked_after_phase = True
                                _sr_after_count = 0
                                _sr_done = False
                                _sr_reason = ""
                            # KL breaker: model already overfit on this sample
                            if (
                                not _sr_done
                                and _sr_kl_thresh > 0.0
                                and metrics.kl_divergence > _sr_kl_thresh
                            ):
                                _sr_done, _sr_reason = True, "kl_break"
                                print(
                                    f"  [repeat KL break] iter {_step_iter:.1f} "
                                    f"KL={metrics.kl_divergence:.4f} > {_sr_kl_thresh:.4f}"
                                )
                            # Regions exhausted: all diverse regions covered
                            if (
                                not _sr_done
                                and _esc_div
                                and _sr_unforced_rep
                                >= getattr(self._esc_cfg.diverse, "num_regions", 999)
                            ):
                                _sr_done, _sr_reason = True, "regions_exhausted"
                            # Adaptive masking: mask prompt only if model failed
                            # If success (non_forced_correct, naked_correct), unmask and advance
                            if _adaptive_masking and not _is_force_rep:
                                _model_succeeded = _sr_reason in (
                                    "non_forced_correct",
                                    "naked_correct",
                                )
                                if _model_succeeded:
                                    # Model got it right! Keep unmask and advance to next sample
                                    self._current_repeat_index = 0  # Reset for next sample
                                    _sr_done = True
                                    _sr_reason = "adaptive_success"
                                else:
                                    # Model failed. Enable masking for next repeat if not done.
                                    if not _sr_done:
                                        self._current_repeat_index = 1
                            # Stamp repeat metadata into extras for CSV/JSONL/WandB
                            metrics.extras["repeat_idx"] = _sr_rep
                            metrics.extras["repeat_total"] = _sr_rep if _sr_done else None
                            if _naked_before or _naked_after:
                                if _sr_unforced_rep <= _naked_before:
                                    metrics.extras["naked_phase"] = "before"
                                elif self._in_naked_after_phase:
                                    metrics.extras["naked_phase"] = "after"
                                else:
                                    metrics.extras["naked_phase"] = "normal"
                            metrics.extras["repeat_done_reason"] = _sr_reason if _sr_done else ""
                            metrics.extras["dataset_row"] = _sr_dataset_row
                            # Curriculum masking: log whether prompt was masked this sample
                            with _PROMPT_MASK_STATE_LOCK:
                                metrics.extras["prompt_masked"] = getattr(
                                    self, "_prompt_masked_this_sample", False
                                )
                            # Every rep is a full step — fire all post-step hooks now
                            self._do_post_step(_step_iter, metrics, _rep_t0)
                            if _sr_done:
                                self._force_answer_this_rep = False  # reset after repeat loop
                                break
                            # Between reps: full Metal sync + free before next generation.
                            # del metrics drops Python refs so MLX ref-counts hit zero.
                            # _metal_full_reset() evaluates model params to drain gradient
                            # step residuals from Metal's command queue, then clears cache.
                            # Without this, residuals collide with the next generation's
                            # allocations and trigger the 8s Metal watchdog.
                            del metrics
                            _metal_full_reset(self.model)
                            _rep_cd = getattr(self.config, "phase_cooldown", 2.0)
                            if _rep_cd > 0:
                                time.sleep(_rep_cd)
                        # Flush accumulated gradients if step-level accumulation was enabled
                        if _accum_over_step:
                            self._flush_step_accum()
                            self._step_accum_mode = False
                        _post_step_done = True
                    elif sps <= 1:
                        metrics = self._train_step(iteration)
                        metrics.extras.setdefault("repeat_idx", 1)
                        _ds_idx = getattr(self.dataset, "_index", 0)
                        metrics.extras.setdefault("dataset_row", max(_ds_idx - 1, 0))
                    else:
                        # Repeat same batch with fresh rollouts each time
                        batch = self._sample_batch()
                        _accum_over_step = getattr(self.config, "accumulate_over_step", False)
                        if _accum_over_step:
                            self._step_accum_mode = True
                            self._accum_grads = None
                            self._accum_count = 0
                        for _rep in range(sps):
                            metrics = self._train_step(iteration, batch=batch)
                        if _accum_over_step:
                            self._flush_step_accum()
                            self._step_accum_mode = False
                    consecutive_failures = 0
                except RuntimeError as e:
                    if "Command buffer execution failed" in str(e) or "METAL" in str(e):
                        consecutive_failures += 1
                        _metal_clear_cache()
                        print(
                            f"  [Metal GPU timeout at iter {iteration}, "
                            f"recovery {consecutive_failures}/3]"
                        )
                        if consecutive_failures >= 3:
                            raise RuntimeError("3 consecutive Metal GPU timeouts — aborting") from e
                        continue  # Skip to next iteration
                    raise

                if not _post_step_done:
                    self._do_post_step(iteration, metrics, t0)
        except KeyboardInterrupt:
            self._graceful_shutdown(sig_name="KEYBOARD INTERRUPT (Ctrl+C)")
            raise
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGTERM, original_sigterm)
            signal.signal(signal.SIGINT, original_sigint)
            if self._process_pool is not None:
                self._process_pool.shutdown()

    def _sample_batch(self) -> list[Sample]:
        """Draw a batch of samples from the dataset."""
        bs = self.config.batch_size
        if self._adaptive is not None:
            batch = []
            for _ in range(bs):
                _, sample = self._adaptive.next()
                batch.append(sample)
            return batch
        if getattr(self.config, "batches_per_type", False) and bs > 1:
            return self.dataset.next_batch_same_type(bs)
        return self.dataset.next_batch(bs)

    def _do_post_step(self, iteration: float, metrics: StepMetrics, t0: float) -> None:
        """Full post-step: timing, hooks, reporting, checkpoint, eval, optimizer reset, cache."""
        metrics.elapsed_sec = time.perf_counter() - t0
        _gen_sec = metrics.generation_sec if metrics.generation_sec > 0 else metrics.elapsed_sec
        metrics.tokens_per_sec = metrics.generated_tokens / max(_gen_sec, 0.001)
        metrics.peak_memory_mb, metrics.rss_mb = _collect_memory()
        if self._process_pool is not None and self._process_pool._last_metal_mb > 0:
            metrics.peak_memory_mb = self._process_pool._last_metal_mb
        self._last_metrics = metrics
        # Compute weight angles before firing hooks so monitor/WandB see them this step
        # NOTE: Only for MLX backend (PyTorch has no MLX-specific weight operations)
        _wa_every = getattr(self.config, "log_weight_angle_every", 0)
        if (
            _wa_every > 0
            and self.model is not None
            and getattr(self, "_weight_angle_baseline", None) is not None
        ):
            _iiter_pre = int(iteration)
            if (_iiter_pre + 1) % _wa_every == 0:
                _angles = _compute_weight_angles(self.model, self._weight_angle_baseline)
                if _angles:
                    metrics.extras["weight_angles"] = _angles
                    _cur_avg = sum(_angles.values()) / len(_angles)
                    # DEBUG: Log baseline state at each computation
                    if _iiter_pre % 2 == 0:  # Log every other step to reduce spam
                        _baseline_keys = list(self._weight_angle_baseline.keys())[
                            :2
                        ]  # First 2 keys
                        print(
                            f"[angle_compute] iter={iteration:.1f}, avg={_cur_avg:.6f}°, "
                            f"baseline_taken_at={self._baseline_snapshot_iter}"
                        )
                    if self._last_angle_avg is not None:
                        metrics.extras["weight_angle_delta"] = _cur_avg - self._last_angle_avg
                    self._last_angle_avg = _cur_avg

                    # Compute dynamic gradient scaling multipliers
                    _dgs_cfg = getattr(self.config, "dynamic_gradient_scaling", None)
                    if _dgs_cfg and _dgs_cfg.enabled:
                        from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler

                        _layer_names = [f"L{20 + i}" for i in range(16)]
                        _angle_drifts = [_angles.get(ln, 0.0) for ln in _layer_names]
                        _scaler = DynamicGradientScaler(
                            angle_drift_threshold=_dgs_cfg.angle_drift_threshold,
                            min_multiplier=_dgs_cfg.min_multiplier,
                            max_multiplier=_dgs_cfg.max_multiplier,
                        )
                        _mults = _scaler.compute_multipliers(_angle_drifts)
                        metrics.extras["gradient_multipliers"] = _mults
        self._fire("after_step", metrics)
        # Use integer part for modular checks (fractional sub-steps share base iteration)
        _iiter = int(iteration)
        steps_per_report = getattr(self.config, "steps_per_report", 1)
        if steps_per_report and (_iiter + 1) % steps_per_report == 0:
            self._report_step(iteration, self.config.iterations, metrics)
        if self._process_pool is None:
            if self.config.save_every and (_iiter + 1) % self.config.save_every == 0:
                self._fire("on_checkpoint", metrics)
            if self.config.eval_every and (_iiter + 1) % self.config.eval_every == 0:
                self._fire("on_eval", metrics)
        _reset_every = getattr(self.config, "optimizer_reset_every", 0)
        if _reset_every > 0 and (_iiter + 1) % _reset_every == 0 and self._process_pool is None:
            self._reset_optimizer()
        _eval_cmd = getattr(self.config, "eval_command", "")
        _eval_every = getattr(self.config, "eval_command_every", 0)
        if _eval_cmd and _eval_every > 0 and (_iiter + 1) % _eval_every == 0:
            self._run_eval_command(_eval_cmd, _iiter + 1, metrics)
        _metal_clear_cache()
        _iter_cd = getattr(self.config, "phase_cooldown", 2.0)
        if _iter_cd > 0:
            time.sleep(_iter_cd)

    def _run_eval_command(
        self, cmd: str, iteration: int, metrics: StepMetrics | None = None
    ) -> None:
        """Run a shell command and write timestamped output to its own log file."""
        import os
        import subprocess
        from datetime import datetime
        from pathlib import Path

        log_dir = Path(getattr(self.config, "eval_log_dir", "logs/eval"))
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"eval_iter{iteration:07d}_{ts}.log"
        print(f"\n  [eval_command] iter {iteration}: running → {cmd}")
        print(f"  [eval_command] log → {log_path}")
        env = os.environ.copy()
        env["EVAL_ITER"] = str(iteration)
        if metrics is not None:
            if metrics.loss is not None:
                env["EVAL_AVG_LOSS"] = f"{metrics.loss:.6f}"
            if metrics.kl_divergence is not None:
                env["EVAL_KL"] = f"{metrics.kl_divergence:.6f}"
        with open(log_path, "w") as fh:
            fh.write(f"# iter={iteration}  cmd={cmd}  started={ts}\n\n")
            fh.flush()
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                fh.write(f"\n# exit_code={result.returncode}\n")
            except Exception as exc:  # noqa: BLE001
                fh.write(f"\n# ERROR: {exc}\n")
                print(f"  [eval_command] failed: {exc}")
                return
        print(f"  [eval_command] done (exit={result.returncode})")

    def _train_step(
        self,
        iteration: int,
        batch: list[Sample] | None = None,
    ) -> StepMetrics:
        """Single training step: rollouts → rewards → gradient update.

        Supports batch_size > 1: fetches multiple samples, generates G
        rollouts per sample, normalizes advantages per-prompt, then
        flattens everything into a single gradient step.
        """
        gs = self.config.group_size

        # Sample selection: use provided batch or draw fresh
        sample_indices: list[int] = []
        if batch is None:
            if self._adaptive is not None:
                batch = []
                for _ in range(self.config.batch_size):
                    idx, sample = self._adaptive.next()
                    batch.append(sample)
                    sample_indices.append(idx)
            elif getattr(self.config, "batches_per_type", False) and self.config.batch_size > 1:
                batch = self.dataset.next_batch_same_type(self.config.batch_size)
            else:
                batch = self.dataset.next_batch(self.config.batch_size)

        self._fire("before_rollout", StepMetrics(iteration=iteration))

        all_rollouts: list[RolloutResult] = []
        all_advantages: list[float] = []
        all_rewards: list[float] = []
        all_prompts: list[str] = []
        all_answers: list[str] = []
        all_targets: list[str] = []

        if self._process_pool is not None:
            _reset_every = getattr(self.config, "optimizer_reset_every", 0)
            _do_reset = _reset_every > 0 and (iteration + 1) % _reset_every == 0
            sample_results, _grad_result = self._process_pool.dispatch(
                batch, iteration, self._reward_baseline, reset_optimizer=_do_reset
            )
        else:
            _grad_result = None
            _workers = getattr(self.config, "rollout_workers", 1)
            sample_results = dispatch_parallel(
                lambda s: self._run_sample_group(s, iteration),
                batch,
                max_workers=_workers,
                on_error=lambda i, e: print(f"  [warn] Skipping batch item {i}: {e}"),
            )

        _last_tui_ctx: tuple | None = None
        _total_gen_sec = 0.0
        for _sample, sgr in zip(batch, sample_results, strict=False):
            if sgr is None:
                continue
            for evt, mtr in sgr.pending_events:
                self._fire(evt, mtr)
            self._reward_baseline = 0.99 * self._reward_baseline + 0.01 * sgr.mean_reward
            all_rollouts.extend(sgr.group_rollouts)
            all_advantages.extend(sgr.advantages)
            all_rewards.extend(sgr.rewards)
            all_prompts.extend(sgr.prompts)
            all_answers.extend(sgr.answers)
            all_targets.extend(sgr.targets)
            _last_tui_ctx = sgr.tui_ctx
            _total_gen_sec += sgr.generation_sec
        # sample = last batch item, used for TUI re-run context below
        sample = batch[-1] if batch else None

        # Update adaptive sampler with mean reward per sample
        if self._adaptive is not None and sample_indices:
            for bi, idx in enumerate(sample_indices):
                group_rewards = all_rewards[bi * gs : (bi + 1) * gs]
                if group_rewards:
                    mean_r = sum(group_rewards) / len(group_rewards)
                    self._adaptive.update(idx, mean_r)

        self._fire("after_rollout", StepMetrics(iteration=iteration))

        # TUI edit request: re-run a member with user-edited scaffold before gradient step
        _bridge = getattr(self, "_tui_bridge", None)
        if (
            _bridge is not None
            and all_rollouts
            and not _bridge.edit_queue.empty()
            and _last_tui_ctx
        ):
            _edit = _bridge.edit_queue.get_nowait()
            _tc_config, _tc_reward_fn, _tc_prompt_tokens, _tc_callbacks = _last_tui_ctx
            _edited = self._rerun_with_scaffold(
                _edit,
                sample,
                _tc_config,
                _tc_reward_fn,
                _tc_prompt_tokens,
                iteration,
                _tc_callbacks,
            )
            if _edited is not None:
                _edit_reward = float(_edited.metadata.get("_esc_reward", 0.0))
                all_rollouts.append(_edited)
                all_advantages.append(_edit_reward)
                all_rewards.append(_edit_reward)

        # GPU phase break: generation → gradient
        try:
            import mlx.core as _mx

            _mx.synchronize()
        except (ImportError, AttributeError):
            pass
        _metal_clear_cache()
        _phase_cd = getattr(self.config, "phase_cooldown", 2.0)
        if _phase_cd > 0:
            time.sleep(_phase_cd)

        # ── Gradient update (all B*G rollouts at once) ──
        self._fire(
            "on_gradient",
            StepMetrics(
                iteration=iteration,
                extras={"phase": "gradient"},
            ),
        )
        # MoE: inject dominant task type for supervised routing blend
        if hasattr(self.model, "current_task_type"):
            from collections import Counter

            dominant = Counter(s.type_name for s in batch).most_common(1)[0][0]
            self.model.current_task_type = dominant

        # Spike guard — snapshot weights before gradient step
        if self._spike_guard is not None and self.model is not None:
            self._spike_guard.snapshot(self.model)

        if self._process_pool is not None:
            # Worker 0 handled the gradient step; pull metrics from dispatch result
            loss = _grad_result.loss if _grad_result else 0.0
            kl = _grad_result.kl if _grad_result else 0.0
            grad_norm = _grad_result.grad_norm if _grad_result else 0.0
        elif not all_rollouts:
            print("  [warn] All samples in batch disabled — skipping gradient step")
            return StepMetrics(iteration=iteration)
        else:
            loss, kl, grad_norm = self._do_gradient_step(all_rollouts, all_advantages)
            _metal_clear_cache()

        # Spike guard — check after gradient step; restore and skip if triggered
        if self._spike_guard is not None and self._process_pool is None:
            _sg_info = {
                "prompt": batch[0].prompt[:200] if batch else "",
                "type": batch[0].type_name if batch else "",
            }
            from .spike_guard import SpikeShutdownError

            try:
                _sg_action = self._spike_guard.check(loss, kl, iteration, _sg_info)
            except SpikeShutdownError:
                raise
            if _sg_action in ("skip", "revert"):
                if _sg_action == "revert":
                    self._spike_guard.restore_checkpoint(self.model, self.optimizer)
                else:
                    self._spike_guard.restore_snapshot(self.model)
                self._spike_guard.reset_accum(self)
                return StepMetrics(
                    iteration=iteration,
                    extras={"spike_skipped": True, "spike_action": _sg_action},
                )

        # ── Optional SFT anchor step ──
        if (
            self._process_pool is None
            and (self._mlx_ops_available or self._torch_ops_available)
            and getattr(self.config, "sft_anchor_enabled", False)
            and self.tokenizer is not None
        ):
            sft_every = getattr(self.config, "sft_anchor_every", 1)
            if sft_every and (iteration + 1) % sft_every == 0:
                self._sft_anchor_step(batch)

        total_tokens = sum(r.generated_token_count for r in all_rollouts)
        mean_r = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        std_r = (
            (sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
            if all_rewards
            else 0.0
        )

        # Per-type stats for reporting/logging
        ut = self._used_tokens
        type_stats: dict[str, dict[str, Any]] = {}
        for i, rollout in enumerate(all_rollouts):
            tn = rollout.type_name or "unknown"
            if tn not in type_stats:
                type_stats[tn] = {
                    "rewards": [],
                    "advantages": [],
                    "gen_tokens": 0,
                    "trained_tokens": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "count": 0,
                }
            ts = type_stats[tn]
            ts["rewards"].append(all_rewards[i])
            ts["advantages"].append(all_advantages[i])
            gc = rollout.generated_token_count
            ts["gen_tokens"] += gc
            ts["trained_tokens"] += min(gc, ut) if ut else gc
            ts["correct"] += 1 if all_rewards[i] > 0.5 else 0
            ts["incorrect"] += 1 if all_rewards[i] <= 0.5 else 0
            ts["count"] += 1

        type_names = list(dict.fromkeys(r.type_name or "unknown" for r in all_rollouts))

        trained_tokens = sum(ts["trained_tokens"] for ts in type_stats.values())
        scaffold_tokens = sum(
            sum(len(s.tokens) for s in r.segments if s.source == "injected" and s.tag == "scaffold")
            for r in all_rollouts
        )

        _accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        _accum_this_iter = (
            getattr(self.config, "accumulate_over_step", False) and self._accum_count_this_iter > 0
        )
        extras: dict[str, Any] = {
            "rollouts": all_rollouts,
            "advantages": all_advantages,
            "prompts": all_prompts,
            "answers": all_answers,
            "targets": all_targets,
            "type_stats": type_stats,
            "type_names": type_names,
            "group_size": gs,
            "total_views": len(all_rollouts),
            "trained_tokens": trained_tokens,
            "scaffold_tokens": scaffold_tokens,
            "accum_count": self._accum_count_this_iter if _accum_this_iter else self._accum_count,
            "accum_steps": _accum_steps,
            # Show how many repeats were accumulated in this iteration
            "accum_over_repeats": self._accum_count_this_iter,
        }
        if self._adaptive is not None:
            extras["pool_sizes"] = self._adaptive.pool_sizes

        # MoE routing metrics
        if hasattr(self.model, "get_routing_mode"):
            extras["moe_routing_mode"] = self.model.get_routing_mode()
            extras["moe_temperature"] = self.model.get_temperature()
            extras["moe_step"] = self.model.current_step

        # Per-view stats for multi-view contrastive scaffold logging
        view_stats: dict[str, dict[str, Any]] = {}
        for i, rollout in enumerate(all_rollouts):
            vn = rollout.metadata.get("_view_name")
            if vn is not None:
                if vn not in view_stats:
                    view_stats[vn] = {
                        "rewards": [],
                        "advantages": [],
                        "scaffold_ratio": rollout.metadata.get("_view_scaffold_ratio", 0.0),
                        "is_naked": rollout.metadata.get("_view_is_naked", False),
                    }
                view_stats[vn]["rewards"].append(all_rewards[i])
                view_stats[vn]["advantages"].append(all_advantages[i])
        if view_stats:
            extras["view_stats"] = view_stats

        # IS weight per-rollout summary — consumed by WandB hook and CSV logger.
        # Process pool: weights are stamped in gradient worker; pull stats from GradientResult.
        # Thread mode: weights are stamped in _do_gradient_step (above) on all_rollouts.
        if self._process_pool is not None and _grad_result is not None:
            _is_rollout_stats: list[dict[str, float]] = _grad_result.is_weight_stats
        else:
            _is_rollout_stats = []
            for r in all_rollouts:
                _iws: list[float] = r.metadata.get("_is_weights") or []
                if _iws:
                    _is_rollout_stats.append(
                        {"mean": sum(_iws) / len(_iws), "min": min(_iws), "max": max(_iws)}
                    )
        if _is_rollout_stats:
            extras["is_weight_stats"] = _is_rollout_stats

        # GAR metrics from last gradient step
        if self._gar_mode and hasattr(self, "_gar_last_metrics") and self._gar_last_metrics:
            extras["gar_loss"] = self._gar_last_metrics.get("loss", 0.0)
            extras["gar_angles"] = self._gar_last_metrics.get("per_layer_angles", {})
            extras["gar_scaffold_fraction"] = self._gar_last_metrics.get("scaffold_fraction", 0.0)

        # LLM judge stats (collect from cached reward instances)
        if hasattr(self, "_reward_cache"):
            from mlx_grpo.rewards.llm_judge import LLMJudgeReward

            for rw in self._reward_cache.values():
                if isinstance(rw, LLMJudgeReward) and rw.judge_logger is not None:
                    extras["judge_stats"] = rw.judge_logger.step_stats()
                    break

        # Extract current learning rate
        lr = 0.0
        if self._process_pool is not None and _grad_result is not None:
            lr = _grad_result.learning_rate
        elif self.optimizer is not None:
            try:
                _lr = self.optimizer.learning_rate
                lr = float(_lr.item()) if hasattr(_lr, "item") else float(_lr)
            except (AttributeError, TypeError):
                pass

        metrics = StepMetrics(
            iteration=iteration,
            loss=loss,
            mean_reward=mean_r,
            reward_std=std_r,
            kl_divergence=kl,
            grad_norm=grad_norm,
            generated_tokens=total_tokens,
            learning_rate=lr,
            rewards=all_rewards,
            extras=extras,
            generation_sec=_total_gen_sec,
        )

        # Reset this iter's accumulation counter after metrics are built
        self._accum_count_this_iter = 0

        return metrics

    def _rerun_with_scaffold(
        self,
        edit: Any,
        sample: Sample,
        type_config: dict[str, Any],
        reward_fn: BaseReward,
        prompt_tokens: list[int] | None,
        iteration: int,
        callbacks: Any,
    ) -> RolloutResult | None:
        """Re-run _vertical_rollout with user-edited scaffold from the TUI pause panel."""
        from .escalation import _vertical_rollout  # noqa: PLC0415

        total_iters = getattr(self.config, "iterations", 1)
        _bridge = getattr(self, "_tui_bridge", None)
        seg_cb = _bridge.make_segment_cb(edit.member_idx) if _bridge else None
        batch_ctx = BatchContext(
            iteration=iteration,
            total_iters=total_iters,
            group_index=edit.member_idx,
            group_size=-1,
            scaffold=ScaffoldConfig(mode="prefix"),
            custom={
                "segments": edit.scaffold_segs,
                "vert_cfg": None,
                "N_eff": len(edit.scaffold_segs),
                "B_eff": 64,
                "b_ext": 0,
                "answer_prefix_text": "",
                "answer_prefix_toks": 0,
                "segment_cb": seg_cb,
            },
        )
        result = self.executor.execute(
            _vertical_rollout,
            prompt=sample.prompt,
            prompt_tokens=prompt_tokens,
            answer=sample.answer,
            type_name=sample.type_name,
            type_config=type_config,
            batch=batch_ctx,
            callbacks=callbacks,
            sample=sample.metadata,
            max_completion_tokens=self.max_completion_tokens,
            prompt_cache=None,
        )
        ground_truth = sample.metadata.get("ground_truth", sample.answer)
        reward = reward_fn.compute(result, ground_truth)
        result.metadata.update(
            {
                "_esc_member_idx": edit.member_idx,
                "_esc_allocated_chunks": len(edit.scaffold_segs),
                "_esc_n_eff": len(edit.scaffold_segs),
                "_esc_reward": reward,
                "_view_name": f"edited_{edit.member_idx}",
                "_view_is_naked": False,
            }
        )
        return result

    def _run_sample_group(
        self,
        sample: Sample,
        iteration: int,
    ) -> SampleGroupResult | None:
        """Generate and score all rollouts for one sample. Thread-safe via _ROLLOUT_LOCK.

        Returns None when the sample should be skipped (disabled type, image-only prompt).
        pending_events holds buffered on_rollout_start / on_group fire calls to be
        replayed in the main thread after dispatch_parallel completes.
        """
        gs = self.config.group_size
        pending_events: list[tuple[str, StepMetrics]] = []

        # Resolve task type + overrides
        task_type = TaskType.get(sample.type_name)
        overrides = self._type_overrides.get(sample.type_name) or self._type_overrides.get(
            TaskType._aliases.get(sample.type_name, sample.type_name), {}
        )
        merged_type_config = (
            {**task_type.config(), **overrides} if overrides else task_type.config()
        )
        # Set prompt_mask_strategy from training config if not overridden per-type
        if "prompt_mask_strategy" not in merged_type_config:
            merged_type_config["prompt_mask_strategy"] = self.config.prompt_mask_strategy
        # Curriculum masking: fixed or adaptive
        _use_curriculum_masking = getattr(self.config, "mask_prompt_on_repeats", True)
        _use_adaptive_masking = getattr(self.config, "adaptive_prompt_masking", False)
        if _use_curriculum_masking and not _use_adaptive_masking:
            # Fixed curriculum: first pass (naked) uses no_mask, repeats use mask
            # This lets the model learn prompt-reasoning association on first pass,
            # then specialize on reasoning/answer-only on repeats to avoid overfitting
            if self._current_repeat_index > 0:
                merged_type_config["prompt_mask_strategy"] = "mask"
        elif _use_adaptive_masking:
            # Adaptive curriculum: mask only if _current_repeat_index > 0 (model failed)
            if self._current_repeat_index > 0:
                merged_type_config["prompt_mask_strategy"] = "mask"
            else:
                merged_type_config["prompt_mask_strategy"] = "no_mask"
        # Track whether prompt is masked for this sample
        with _PROMPT_MASK_STATE_LOCK:
            self._prompt_masked_this_sample = (
                merged_type_config.get("prompt_mask_strategy") == "mask"
            )
        if not merged_type_config.get("enabled", True):
            return None
        if "![" in (sample.prompt or ""):
            return None

        reward_fn = self._get_reward(task_type, merged_type_config)
        callbacks = self._build_callbacks(task_type)
        system_prompt = self._resolve_system_prompt(sample)
        chat_messages = [
            *([{"role": "system", "content": system_prompt}] if system_prompt else []),
            {"role": "user", "content": sample.prompt},
        ]

        # Pre-tokenize via chat template
        prompt_tokens: list[int] | None = None
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            _cp = task_type.chat_params()
            _tpl_kw: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": False,
            }
            if _cp.enable_thinking:
                _tpl_kw["enable_thinking"] = True
                _tb = merged_type_config.get("thinking_budget") or _cp.thinking_budget
                if _tb is not None:
                    _tpl_kw["thinking_budget"] = int(_tb)
            try:
                prompt_tokens = self.tokenizer.apply_chat_template(chat_messages, **_tpl_kw)
            except TypeError:
                prompt_tokens = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            if _cp.enable_thinking and prompt_tokens:
                _tail = self.tokenizer.decode(prompt_tokens[-5:])
                if "<think>" not in _tail:
                    _think_toks = self.tokenizer.encode("<think>", add_special_tokens=False)
                    prompt_tokens = list(prompt_tokens) + _think_toks

        # Compute system_prompt_token_count for per-view masking
        _system_prompt_token_count = 0
        if system_prompt and prompt_tokens and self.tokenizer is not None:
            try:
                # Tokenize user-only messages (no system prompt)
                _user_only_msgs = [{"role": "user", "content": sample.prompt}]
                _tpl_kw_user = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_dict": False,
                }
                if _cp.enable_thinking:
                    _tpl_kw_user["enable_thinking"] = True
                    _tb = merged_type_config.get("thinking_budget") or _cp.thinking_budget
                    if _tb is not None:
                        _tpl_kw_user["thinking_budget"] = int(_tb)
                try:
                    _user_only_tokens = self.tokenizer.apply_chat_template(
                        _user_only_msgs, **_tpl_kw_user
                    )
                except TypeError:
                    _user_only_tokens = self.tokenizer.apply_chat_template(
                        _user_only_msgs,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                _user_only_tokens = list(_user_only_tokens) if _user_only_tokens else []
                # Account for <think> token if added
                if _cp.enable_thinking and prompt_tokens:
                    _tail = self.tokenizer.decode(prompt_tokens[-5:])
                    if "<think>" in _tail:
                        _user_only_tokens = _user_only_tokens[:-1]
                _system_prompt_token_count = len(prompt_tokens) - len(_user_only_tokens)
            except Exception:
                # If tokenization fails, fall back to 0
                _system_prompt_token_count = 0

        # Build merged scaffold (type overrides global defaults)
        type_scaffold_raw = merged_type_config.get("scaffold")
        merged_scaffold = build_scaffold_config(type_scaffold_raw, defaults=self._global_scaffold)

        # Naked repeat override: force zero scaffold during naked phases.
        _nb = getattr(self, "_naked_before", 0)
        _na = getattr(self, "_naked_after", 0)
        _rep_idx = self._current_repeat_index
        _is_naked = False
        if (
            (_nb > 0 or _na > 0)
            and (_nb + _na) > 0
            and (_rep_idx < _nb or getattr(self, "_in_naked_after_phase", False))
        ):
            _is_naked = True
        if _is_naked:
            merged_scaffold = ScaffoldConfig(
                mode=merged_scaffold.mode,
                initial_ratio=0.0,
            )

        if iteration == 0 and (merged_scaffold.levels or merged_scaffold.max_scaffolded_tokens):
            print(f"{'=' * 60}\nMulti-Curriculum Scaffold\n{'=' * 60}")
            if merged_scaffold.levels:
                pcts = [f"{int(v * 100)}%" for v in merged_scaffold.levels]
                print(f"  Levels per group: {pcts}")
            if merged_scaffold.max_scaffolded_tokens:
                print(f"  Max scaffolded tokens: {merged_scaffold.max_scaffolded_tokens}")
            print(f"  Group size: {gs}\n{'=' * 60}")

        # Prefill KV cache once for the group (MLX backend only)
        _pcache = None
        _be = self.backend
        _kv_quantized = getattr(_be, "_kv_bits", None) is not None
        if (
            gs > 1
            and prompt_tokens
            and self.model is not None
            and not _kv_quantized
            and _be.name == "mlx"
        ):
            try:
                from mlx_grpo.backends.cache import prefill_prompt_cache

                _pcache = prefill_prompt_cache(
                    self.model,
                    prompt_tokens,
                    kv_group_size=getattr(_be, "_kv_group_size", 64),
                    max_kv_size=getattr(_be, "_max_kv_size", None),
                    prefill_step_size=getattr(_be, "_prefill_step_size", 512),
                )
            except (ImportError, TypeError, AttributeError):
                _pcache = None

        _answer_count = 0
        if merged_scaffold.dynamic_max_tokens and sample.answer and self.tokenizer:
            _answer_count = len(self.tokenizer.encode(sample.answer))

        pgc = merged_scaffold.per_group_config
        _progress = iteration / max(self.config.iterations, 1)

        # Estimate difficulty and select scaffold ratio if boundary optimization enabled
        _selected_scaffold_ratio = None
        _sample_difficulty = None
        if self._boundary_optimizer is not None and self._difficulty_estimator is not None:
            _sample_difficulty = self._difficulty_estimator.estimate(
                prompt=sample.prompt or "",
                answer=sample.answer or "",
                sample_type=sample.type_name,
                metadata=sample.metadata,
            )
            _selected_scaffold_ratio = self._boundary_optimizer.select_scaffold_ratio(
                sample_type=sample.type_name,
                difficulty=_sample_difficulty,
            )
            # Override merged_scaffold initial_ratio with optimizer's selection
            merged_scaffold = ScaffoldConfig(
                max_tokens=merged_scaffold.max_tokens,
                mode=merged_scaffold.mode,
                target=merged_scaffold.target,
                initial_ratio=_selected_scaffold_ratio,
                decay_rate=merged_scaffold.decay_rate,
                end_progress=merged_scaffold.end_progress,
                min_ratio=merged_scaffold.min_ratio,
                levels=merged_scaffold.levels,
                max_scaffolded_tokens=merged_scaffold.max_scaffolded_tokens,
                dynamic_max_tokens=merged_scaffold.dynamic_max_tokens,
                per_group_config=merged_scaffold.per_group_config,
            )

        if iteration == 0 and pgc is not None:
            print(f"{'=' * 60}\nMulti-View Contrastive Scaffold\n{'=' * 60}")
            for vi, vw in enumerate(pgc.views):
                flag = "(naked)" if vw.scaffold_position == "none" else f"ep={vw.end_progress}"
                print(
                    f"  [{vi}] {vw.name}: pos={vw.scaffold_position}"
                    f" -> {vw.truncate_from}/{vw.used_tokens}tok {flag}"
                )
            print("=" * 60)

        # Resolve forced answer text if this is a forced repeat
        _force_ans = getattr(self, "_force_answer_this_rep", False)
        _force_ans_text = ""
        if _force_ans and sample.answer:
            _force_ans_text = sample.metadata.get("ground_truth", sample.answer)

        # Select rollout strategy
        _use_adaptive = (
            sample.type_name == "math"
            and merged_type_config.get("view_strategy") == "adaptive_escalation"
            and hasattr(task_type, "escalation_rollout")
        )
        _use_escalation = (
            not _use_adaptive
            and self._esc_cfg is not None
            and getattr(self._esc_cfg, "enabled", False)
        )
        _use_dynamic = self._dv_cfg is not None and not _use_adaptive and not _use_escalation

        # Naked repeat phase: bypass multi-view strategies, run full group with zero scaffold
        if _is_naked:
            _use_dynamic = False
            _use_escalation = False
            _use_adaptive = False
            _phase = "before" if _rep_idx < _nb else "after"
            print(f"  [naked {_phase}] rep={_rep_idx} — {gs} members, zero scaffold")

        group_rollouts: list[RolloutResult] = []
        _gen_t0 = time.perf_counter()

        if _use_escalation:
            from .escalation import run_escalation  # noqa: PLC0415
            from .escalation_plan import merge_esc_cfg  # noqa: PLC0415

            # Per-type escalation overrides: type_overrides.<type>.escalation: {...}
            _esc_for_type = merge_esc_cfg(self._esc_cfg, merged_type_config.get("escalation"))
            _esc_tb = merged_type_config.get("thinking_budget", 512)
            _esc_ab = merged_type_config.get("answer_max_tokens", 32)
            _esc_rollout = merged_type_config.get("rollout_max_tokens", _esc_tb + _esc_ab)
            _esc_max_tokens = max(self.max_completion_tokens, _esc_rollout)
            _reward_fn = self._get_reward(task_type, merged_type_config)
            group_rollouts = run_escalation(
                trainer=self,
                sample=sample,
                task_type=task_type,
                merged_type_config=merged_type_config,
                reward_fn=_reward_fn,
                prompt_tokens=prompt_tokens,
                prompt_cache=_pcache,
                iteration=iteration,
                max_completion_tokens=_esc_max_tokens,
                esc_cfg=_esc_for_type,
                pending_events=pending_events,
                force_all_answers=_force_ans_text,
                region_offset_chars=getattr(self, "_scaffold_region_offset_chars", 0),
                region_length_chars=getattr(self, "_scaffold_region_length_chars", 0),
                chunk_count_override=getattr(self, "_scaffold_chunk_count_override", 0),
                fallback=getattr(self, "_scaffold_is_fallback", False),
            )
            # Set system message and masking flags on all results
            for result in group_rollouts:
                result.system_message = system_prompt
                result.system_prompt_token_count = _system_prompt_token_count
        elif _use_adaptive:
            group_rollouts = self._adaptive_math_rollouts(
                sample=sample,
                task_type=task_type,
                merged_type_config=merged_type_config,
                prompt_tokens=prompt_tokens,
                prompt_cache=_pcache,
                iteration=iteration,
                max_completion_tokens=self.max_completion_tokens,
                pending_events=pending_events,
            )
            # Set system message and masking flags on all results
            for result in group_rollouts:
                result.system_message = system_prompt
                result.system_prompt_token_count = _system_prompt_token_count
        elif _use_dynamic:
            group_rollouts = self._dynamic_view_rollouts(
                sample=sample,
                task_type=task_type,
                merged_type_config=merged_type_config,
                prompt_tokens=prompt_tokens,
                prompt_cache=_pcache,
                iteration=iteration,
                max_completion_tokens=self.max_completion_tokens,
                pending_events=pending_events,
            )
            # Set system message and masking flags on all results
            for view_idx, result in enumerate(group_rollouts):
                result.system_message = system_prompt
                result.system_prompt_token_count = _system_prompt_token_count
                # Set prompt_mask_strategy from merged_type_config
                result.prompt_mask_strategy = merged_type_config.get("prompt_mask_strategy", "mask")
                # Get masking flags from dynamic views config
                if self._dv_cfg is not None:
                    _masked_views = getattr(self._dv_cfg, "system_prompt_masked_views", ())
                    if _masked_views:
                        # Per-view masking: only mask specified views
                        result.mask_system_prompt = view_idx in _masked_views
                    else:
                        # Global masking: apply to all views if system_prompt_masked_views is empty
                        result.mask_system_prompt = getattr(
                            self._dv_cfg, "mask_system_prompt", False
                        )
                    result.mask_user_prompt = getattr(self._dv_cfg, "mask_user_prompt", False)
        else:
            # Standard group loop with Metal-safe locking
            _cd = getattr(self.config, "rollout_cooldown", 0.5)
            _pipeline_bs = getattr(self._gen_cfg, "pipeline_batch_size", 0)
            _has_batch_gen = hasattr(self.backend, "generate_batch") and callable(
                getattr(self.backend, "generate_batch", None)
            )
            _use_parallel = not _is_naked and (
                (_pipeline_bs > 0 and _pipeline_bs >= gs) or (_has_batch_gen and gs > 1)
            )

            if _use_parallel:
                # Parallel member execution via ThreadPoolExecutor
                _pcache = None  # Skip KV cache prefill — members diverge due to scaffold

                from concurrent.futures import ThreadPoolExecutor, as_completed

                # Create batch collector for generate_batch() fusion when backend supports it
                _collector = None
                if _has_batch_gen:
                    from mlx_grpo.engine.batch_collector import BatchCollector

                    _collector = BatchCollector(self.backend, group_size=gs)

                def _run_member(g: int) -> tuple[int, RolloutResult, list[tuple[str, StepMetrics]]]:
                    """Run a single group member and buffer its events."""
                    member_events: list[tuple[str, StepMetrics]] = []

                    if pgc is not None:
                        view = pgc.views[g]
                        _view_ratio = get_view_scaffold_ratio(view, _progress)
                        group_scaffold = ScaffoldConfig(
                            mode=view.scaffold_position,
                            target=pgc.target,
                            initial_ratio=_view_ratio,
                            end_progress=2.0,
                            decay_rate=0.0,
                            min_ratio=0.0,
                            max_scaffolded_tokens=merged_scaffold.max_scaffolded_tokens,
                        )
                    else:
                        group_scaffold = merged_scaffold
                        _view_ratio = 0.0

                    batch_ctx = BatchContext(
                        iteration=iteration,
                        total_iters=self.config.iterations,
                        group_index=g,
                        group_size=gs,
                        scaffold=group_scaffold,
                        custom={"forced_answer_text": _force_ans_text} if _force_ans_text else {},
                    )
                    type_max_tokens = merged_type_config.get(
                        "max_tokens", self.max_completion_tokens
                    )
                    _tb = merged_type_config.get("thinking_budget", 0)
                    _ab = merged_type_config.get("answer_max_tokens", 128)
                    _tc = merged_type_config.get("thinking_budget_ceiling", 4096)
                    _hard_cap = getattr(self.config, "max_grad_seq_len", 1200)
                    if _tb > 0:
                        type_max_tokens = max(type_max_tokens, min(_tb + _ab, _tc + _ab, _hard_cap))
                    _dyn_max = group_scaffold.dynamic_budget(
                        _progress, type_max_tokens, _answer_count, g
                    )

                    member_events.append(
                        (
                            "on_rollout_start",
                            StepMetrics(
                                iteration=iteration,
                                extras={
                                    "prompt": sample.prompt[:300],
                                    "type_name": sample.type_name,
                                    "group_index": g + 1,
                                    "group_size": gs,
                                },
                            ),
                        )
                    )

                    _pre_inject: str | None = None
                    if (
                        pgc is not None
                        and view.prefix_lines
                        and sample.answer
                        and view.scaffold_position not in ("none", "prefix")
                    ):
                        _pre_inject = _extract_prefix_lines(sample.answer, view.prefix_lines)

                    _exec_kwargs: dict[str, Any] = dict(
                        prompt=sample.prompt,
                        prompt_tokens=prompt_tokens,
                        answer=sample.answer,
                        type_name=sample.type_name,
                        type_config=merged_type_config,
                        batch=batch_ctx,
                        callbacks=callbacks,
                        sample=sample.metadata,
                        max_completion_tokens=_dyn_max,
                        prompt_cache=None,
                        pre_inject_text=_pre_inject,
                    )
                    if _collector is not None:
                        _exec_kwargs["batch_collector"] = _collector

                    if _collector is not None:
                        # Batch collector coordinates Metal access — no lock needed
                        result = self.executor.execute(task_type.rollout, **_exec_kwargs)
                        try:
                            import mlx.core as _mx

                            _mx.synchronize()
                        except (ImportError, AttributeError):
                            pass
                        _metal_clear_cache()
                    else:
                        with _ROLLOUT_LOCK:
                            result = self.executor.execute(task_type.rollout, **_exec_kwargs)
                            try:
                                import mlx.core as _mx

                                _mx.synchronize()
                            except (ImportError, AttributeError):
                                pass
                            _metal_clear_cache()

                    result.system_message = system_prompt
                    result.system_prompt_token_count = _system_prompt_token_count
                    # Set prompt_mask_strategy from merged_type_config
                    result.prompt_mask_strategy = merged_type_config.get(
                        "prompt_mask_strategy", "mask"
                    )
                    # Set masking flags for per-view or global masking
                    if pgc is not None and self._dv_cfg is not None:
                        # Multi-view scaffold with dynamic views config
                        _masked_views = getattr(self._dv_cfg, "system_prompt_masked_views", ())
                        if _masked_views:
                            result.mask_system_prompt = g in _masked_views
                        else:
                            result.mask_system_prompt = getattr(
                                self._dv_cfg, "mask_system_prompt", False
                            )
                        result.mask_user_prompt = getattr(self._dv_cfg, "mask_user_prompt", False)
                    elif self._dv_cfg is not None:
                        # Dynamic views (fallback for safety)
                        _masked_views = getattr(self._dv_cfg, "system_prompt_masked_views", ())
                        if _masked_views:
                            result.mask_system_prompt = g in _masked_views
                        else:
                            result.mask_system_prompt = getattr(
                                self._dv_cfg, "mask_system_prompt", False
                            )
                        result.mask_user_prompt = getattr(self._dv_cfg, "mask_user_prompt", False)
                    _lo = {
                        k: merged_type_config[k]
                        for k in (
                            "beta",
                            "epsilon",
                            "epsilon_high",
                            "max_grad_norm",
                            "used_tokens",
                            "truncate_from",
                            "truncate_marker",
                            "truncate_marker_masked",
                            "snap_to_boundary",
                            "prompt_mask_strategy",
                        )
                        if k in merged_type_config
                    }
                    if group_scaffold.dynamic_max_tokens:
                        _base_used = _lo.get("used_tokens", self._used_tokens) or 0
                        if _base_used:
                            _dyn_used = merged_scaffold.dynamic_used(
                                _progress, _base_used, _answer_count, g
                            )
                            _lo["used_tokens"] = _dyn_used
                    if _lo:
                        result.metadata["_loss_overrides"] = _lo
                    if pgc is not None:
                        _vlo = result.metadata.get("_loss_overrides", {})
                        _vlo["used_tokens"] = view.used_tokens
                        _vlo["truncate_from"] = view.truncate_from
                        result.metadata["_loss_overrides"] = _vlo
                        result.metadata["_view_name"] = view.name
                        result.metadata["_view_scaffold_ratio"] = _view_ratio
                        result.metadata["_view_is_naked"] = view.scaffold_position == "none"
                    if "importance_sampling" in merged_type_config:
                        from mlx_grpo.training.token_importance import (
                            build_is_config,  # noqa: PLC0415
                        )

                        result.metadata["_is_override"] = build_is_config(
                            self._is_cfg, merged_type_config["importance_sampling"]
                        )
                    if prompt_tokens and self.tokenizer is not None:
                        # Decode only the unmasked portion of the prompt
                        # If both system and user are masked, prompt_text is empty
                        # Otherwise, include the tokens that are used for loss
                        _prompt_to_decode = prompt_tokens
                        if (
                            result.mask_system_prompt or result.mask_user_prompt
                        ) and _system_prompt_token_count > 0:
                            # Selectively slice based on masking flags
                            _sys_count = min(_system_prompt_token_count, len(prompt_tokens))
                            _tokens_to_use = []
                            if not result.mask_system_prompt:
                                # Include system tokens
                                _tokens_to_use.extend(prompt_tokens[:_sys_count])
                            if not result.mask_user_prompt:
                                # Include user tokens
                                _tokens_to_use.extend(prompt_tokens[_sys_count:])
                            _prompt_to_decode = _tokens_to_use
                        result.prompt_text = self.tokenizer.decode(_prompt_to_decode)

                    member_events.append(
                        (
                            "on_group",
                            StepMetrics(
                                iteration=iteration,
                                extras={
                                    "phase": "rollout",
                                    "group_index": g + 1,
                                    "group_size": gs,
                                    "type_name": sample.type_name,
                                },
                            ),
                        )
                    )

                    if _cd > 0:
                        time.sleep(_cd)

                    return g, result, member_events

                # Execute all members concurrently and collect results in order.
                # We store events per-index so replay order matches member order,
                # not as_completed order (Bug #1 fix: event ordering).
                raw: list[RolloutResult | None] = [None] * gs
                raw_events: list[list[tuple[str, StepMetrics]]] = [[] for _ in range(gs)]
                with ThreadPoolExecutor(max_workers=gs) as executor:
                    futures = {executor.submit(_run_member, g): g for g in range(gs)}
                    for fut in as_completed(futures):
                        # Bug #2 fix: catch per-member exceptions so one failure
                        # doesn't crash the entire training step.
                        try:
                            g, result, evts = fut.result()
                            raw[g] = result
                            raw_events[g] = evts
                        except Exception as _member_exc:  # noqa: BLE001
                            import logging  # noqa: PLC0415

                            _g_idx = futures[fut]
                            logging.getLogger(__name__).warning(
                                "Pipeline member %d failed: %s", _g_idx, _member_exc
                            )
                # Replay events in member order (0, 1, 2, …) regardless of
                # completion order — maintains deterministic event sequencing.
                for _evt_list in raw_events:
                    pending_events.extend(_evt_list)

                # Bug #3 fix: filter failed members, skip sample if too few
                # succeed (need ≥2 for meaningful advantage computation).
                group_rollouts = [r for r in raw if r is not None]
                if len(group_rollouts) < 2:
                    import logging  # noqa: PLC0415

                    logging.getLogger(__name__).warning(
                        "Only %d/%d members succeeded for sample '%s'; skipping.",
                        len(group_rollouts),
                        gs,
                        sample.prompt[:60],
                    )
                    return None
            else:
                # Sequential member execution (existing behavior)
                for g in range(gs):
                    if pgc is not None:
                        view = pgc.views[g]
                        _view_ratio = get_view_scaffold_ratio(view, _progress)
                        group_scaffold = ScaffoldConfig(
                            mode=view.scaffold_position,
                            target=pgc.target,
                            initial_ratio=_view_ratio,
                            end_progress=2.0,
                            decay_rate=0.0,
                            min_ratio=0.0,
                            max_scaffolded_tokens=merged_scaffold.max_scaffolded_tokens,
                        )
                    else:
                        group_scaffold = merged_scaffold
                        _view_ratio = 0.0
                    batch_ctx = BatchContext(
                        iteration=iteration,
                        total_iters=self.config.iterations,
                        group_index=g,
                        group_size=gs,
                        scaffold=group_scaffold,
                        custom={"forced_answer_text": _force_ans_text} if _force_ans_text else {},
                    )
                    type_max_tokens = merged_type_config.get(
                        "max_tokens", self.max_completion_tokens
                    )
                    _tb = merged_type_config.get("thinking_budget", 0)
                    _ab = merged_type_config.get("answer_max_tokens", 128)
                    _tc = merged_type_config.get("thinking_budget_ceiling", 4096)
                    _hard_cap = getattr(self.config, "max_grad_seq_len", 1200)
                    if _tb > 0:
                        type_max_tokens = max(type_max_tokens, min(_tb + _ab, _tc + _ab, _hard_cap))
                    _dyn_max = group_scaffold.dynamic_budget(
                        _progress, type_max_tokens, _answer_count, g
                    )
                    pending_events.append(
                        (
                            "on_rollout_start",
                            StepMetrics(
                                iteration=iteration,
                                extras={
                                    "prompt": sample.prompt[:300],
                                    "type_name": sample.type_name,
                                    "group_index": g + 1,
                                    "group_size": gs,
                                },
                            ),
                        )
                    )
                    _pre_inject: str | None = None
                    if (
                        pgc is not None
                        and view.prefix_lines
                        and sample.answer
                        and view.scaffold_position not in ("none", "prefix")
                    ):
                        _pre_inject = _extract_prefix_lines(sample.answer, view.prefix_lines)

                    with _ROLLOUT_LOCK:
                        result = self.executor.execute(
                            task_type.rollout,
                            prompt=sample.prompt,
                            prompt_tokens=prompt_tokens,
                            answer=sample.answer,
                            type_name=sample.type_name,
                            type_config=merged_type_config,
                            batch=batch_ctx,
                            callbacks=callbacks,
                            sample=sample.metadata,
                            max_completion_tokens=_dyn_max,
                            prompt_cache=_pcache,
                            pre_inject_text=_pre_inject,
                        )
                        try:
                            import mlx.core as _mx

                            _mx.synchronize()
                        except (ImportError, AttributeError):
                            pass
                        _metal_clear_cache()

                    result.system_message = system_prompt
                    result.system_prompt_token_count = _system_prompt_token_count
                    # Set prompt_mask_strategy from merged_type_config
                    result.prompt_mask_strategy = merged_type_config.get(
                        "prompt_mask_strategy", "mask"
                    )
                    # Set masking flags for per-view or global masking
                    if pgc is not None and self._dv_cfg is not None:
                        # Multi-view scaffold with dynamic views config
                        _masked_views = getattr(self._dv_cfg, "system_prompt_masked_views", ())
                        if _masked_views:
                            result.mask_system_prompt = g in _masked_views
                        else:
                            result.mask_system_prompt = getattr(
                                self._dv_cfg, "mask_system_prompt", False
                            )
                        result.mask_user_prompt = getattr(self._dv_cfg, "mask_user_prompt", False)
                    elif self._dv_cfg is not None:
                        # Dynamic views (fallback for safety)
                        _masked_views = getattr(self._dv_cfg, "system_prompt_masked_views", ())
                        if _masked_views:
                            result.mask_system_prompt = g in _masked_views
                        else:
                            result.mask_system_prompt = getattr(
                                self._dv_cfg, "mask_system_prompt", False
                            )
                        result.mask_user_prompt = getattr(self._dv_cfg, "mask_user_prompt", False)
                    _lo = {
                        k: merged_type_config[k]
                        for k in (
                            "beta",
                            "epsilon",
                            "epsilon_high",
                            "max_grad_norm",
                            "used_tokens",
                            "truncate_from",
                            "truncate_marker",
                            "truncate_marker_masked",
                            "snap_to_boundary",
                            "prompt_mask_strategy",
                        )
                        if k in merged_type_config
                    }
                    if group_scaffold.dynamic_max_tokens:
                        _base_used = _lo.get("used_tokens", self._used_tokens) or 0
                        if _base_used:
                            _dyn_used = merged_scaffold.dynamic_used(
                                _progress, _base_used, _answer_count, g
                            )
                            _lo["used_tokens"] = _dyn_used
                    if _lo:
                        result.metadata["_loss_overrides"] = _lo
                    if pgc is not None:
                        _vlo = result.metadata.get("_loss_overrides", {})
                        _vlo["used_tokens"] = view.used_tokens
                        _vlo["truncate_from"] = view.truncate_from
                        result.metadata["_loss_overrides"] = _vlo
                        result.metadata["_view_name"] = view.name
                        result.metadata["_view_scaffold_ratio"] = _view_ratio
                        result.metadata["_view_is_naked"] = view.scaffold_position == "none"
                    if "importance_sampling" in merged_type_config:
                        from mlx_grpo.training.token_importance import (
                            build_is_config,  # noqa: PLC0415
                        )

                        result.metadata["_is_override"] = build_is_config(
                            self._is_cfg, merged_type_config["importance_sampling"]
                        )
                    if prompt_tokens and self.tokenizer is not None:
                        # Decode only the unmasked portion of the prompt
                        # If both system and user are masked, prompt_text is empty
                        # Otherwise, include the tokens that are used for loss
                        _prompt_to_decode = prompt_tokens
                        if (
                            result.mask_system_prompt or result.mask_user_prompt
                        ) and _system_prompt_token_count > 0:
                            # Selectively slice based on masking flags
                            _sys_count = min(_system_prompt_token_count, len(prompt_tokens))
                            _tokens_to_use = []
                            if not result.mask_system_prompt:
                                # Include system tokens
                                _tokens_to_use.extend(prompt_tokens[:_sys_count])
                            if not result.mask_user_prompt:
                                # Include user tokens
                                _tokens_to_use.extend(prompt_tokens[_sys_count:])
                            _prompt_to_decode = _tokens_to_use
                        result.prompt_text = self.tokenizer.decode(_prompt_to_decode)
                    group_rollouts.append(result)
                    pending_events.append(
                        (
                            "on_group",
                            StepMetrics(
                                iteration=iteration,
                                extras={
                                    "phase": "rollout",
                                    "group_index": g + 1,
                                    "group_size": gs,
                                    "type_name": sample.type_name,
                                },
                            ),
                        )
                    )
                    if _cd > 0:
                        time.sleep(_cd)

        del _pcache
        # Full Metal sync: drain all pending GPU ops BEFORE reward computation
        # (which may block on Gemini API for 5-10s). Without this, stale Metal
        # command buffers from generation collide with the gradient step.
        try:
            import mlx.core as _mx

            _mx.synchronize()
        except (ImportError, AttributeError):
            pass
        _metal_clear_cache()

        # Set prompt_text for all rollouts that don't have it yet (escalation, dynamic views)
        if prompt_tokens and self.tokenizer is not None:
            _prompt_text = self.tokenizer.decode(prompt_tokens)
            for result in group_rollouts:
                if not result.prompt_text:
                    result.prompt_text = _prompt_text

        # Add metadata for masking and system prompt source
        # Determine system prompt source
        _sys_prompt_source = "none"
        if system_prompt:
            _overrides = self._type_overrides.get(sample.type_name) or self._type_overrides.get(
                TaskType._aliases.get(sample.type_name, sample.type_name), {}
            )
            if _overrides.get("system_prompt"):
                _sys_prompt_source = "yaml_type"
            elif sample.metadata.get("system_prompt"):
                _sys_prompt_source = "sample_jsonl"

        # Tag naked-phase rollouts so sr==-2 stop condition recognizes them
        if _is_naked:
            for result in group_rollouts:
                result.metadata["_view_is_naked"] = True

        # Compute total trained tokens for each result
        for result in group_rollouts:
            # Track masking state
            result.metadata["_system_prompt_masked"] = result.mask_system_prompt
            result.metadata["_user_prompt_masked"] = result.mask_user_prompt
            result.metadata["_system_prompt_source"] = _sys_prompt_source
            result.metadata["_system_prompt_token_count"] = result.system_prompt_token_count

            # Count trained tokens = all tokens with non-zero loss weight
            trained_token_count = len([w for w in result.loss_mask if w > 0.0])
            result.metadata["_trained_token_count"] = trained_token_count

        # Mark forced rollouts (for non-escalation paths; escalation handles its own)
        if _force_ans_text and not _use_escalation:
            for result in group_rollouts:
                result.metadata["_esc_forced_answer"] = True

        _generation_sec = time.perf_counter() - _gen_t0

        # Score rollouts with reward function
        ground_truth = sample.metadata.get("ground_truth", sample.answer)
        if _use_escalation:
            rewards = [
                r.metadata.get("_esc_shaped_reward", r.metadata.get("_esc_reward", 0.0))
                for r in group_rollouts
            ]
        else:
            rewards = [reward_fn.compute(r, ground_truth) for r in group_rollouts]
        # Stamp raw reward on each rollout for answer gradient weighting (pre-scaling).
        for _r, _rw in zip(group_rollouts, rewards, strict=False):
            _r.metadata.setdefault("_reward", _rw)
        _rs = merged_type_config.get("reward_scaling", 1.0)
        if _rs != 1.0:
            rewards = [r * _rs for r in rewards]

        # Observe reward for boundary optimizer (if enabled)
        if (
            self._boundary_optimizer is not None
            and _selected_scaffold_ratio is not None
            and _sample_difficulty is not None
            and rewards
        ):
            mean_reward = sum(rewards) / len(rewards)
            with _BOUNDARY_OPTIMIZER_LOCK:
                self._boundary_optimizer.observe_reward(
                    sample_type=sample.type_name,
                    difficulty=_sample_difficulty,
                    ratio=_selected_scaffold_ratio,
                    reward=mean_reward,
                )

        # Apply Jaccard boosting (before conditional rules)
        _jb_cfg = self.config.jaccard_boost
        if _jb_cfg.enabled:
            from ..engine.jaccard import boost_weights_by_jaccard

            target_text = ground_truth or sample.answer or ""
            target_tokens = target_text.lower().split()
            for result in group_rollouts:
                # Extract tokens from the specified segment
                _seg_text = ""
                if _jb_cfg.apply_segment in ("answer", "both"):
                    for seg in result.segments:
                        if seg.tag and "answer" in seg.tag.lower():
                            _seg_text = seg.text or ""
                            break
                if not _seg_text and _jb_cfg.apply_segment in ("thinking", "both"):
                    for seg in result.segments:
                        if seg.tag and "thinking" in seg.tag.lower():
                            _seg_text = seg.text or ""
                            break
                if not _seg_text:
                    _seg_text = result.completion_text

                gen_tokens = _seg_text.lower().split()
                boosted = boost_weights_by_jaccard(
                    target_tokens,
                    gen_tokens,
                    result.loss_mask,
                    {
                        "boost_factor": _jb_cfg.boost_factor,
                        "min_jaccard": _jb_cfg.min_jaccard,
                        "exclude_patterns": list(_jb_cfg.exclude_patterns),
                        "stop_words": list(_jb_cfg.stop_words),
                    },
                )
                # Store boosted mask in metadata for use during loss computation
                result.metadata["_jaccard_boosted_mask"] = boosted

        # Apply conditional rules — modifies loss_weight and reward_multiplier in-place
        if self._conditional_rules_engine is not None:
            answer_correct = ground_truth is not None
            for g_idx, (result, reward) in enumerate(zip(group_rollouts, rewards, strict=False)):
                self._conditional_rules_engine.apply_to_result(
                    result,
                    reward=reward,
                    answer_correct=answer_correct,
                    is_naked=result.metadata.get("_view_is_naked", False),
                    group_index=g_idx,
                    sample_index=0,
                    prompt_text=result.prompt_text or sample.prompt,
                )

        # Per-prompt advantage normalization
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        lt = getattr(self.config, "loss_type", "grpo")
        # Skip when all rollouts are perfect: GRPO/BNPO get zero advantages (no signal).
        # DR-GRPO uses baseline subtraction → advantages = r - baseline ≠ 0 even here,
        # so skipping is wrong — it discards valid reinforcement gradient.
        if rewards and all(r >= 1.0 for r in rewards) and lt != "dr_grpo":
            return None
        # DR-GRPO: absolute baseline advantage — r - EMA_baseline.
        # When all rollouts fail (rewards ≈ 0, baseline ≈ 0.2) this gives strong
        # negative advantages (~-0.2) → high loss → correct gradient direction.
        # Group normalization (r - mean_r) collapses to ≈0 when all rewards identical.
        if lt == "dr_grpo":
            advantages = [r - self._reward_baseline for r in rewards]
        elif std_r > 1e-4:
            # BUG FIX: Compute advantages per-view-type (naked vs scaffolded)
            # to prevent naked members from getting negative gradients due to
            # scaffolded members' higher rewards. Naked members are harder but
            # still deserve positive signal when they improve.
            advantages = []
            naked_indices = [
                i for i, r in enumerate(group_rollouts) if r.metadata.get("_view_is_naked")
            ]
            scaffolded_indices = [
                i for i, r in enumerate(group_rollouts) if not r.metadata.get("_view_is_naked")
            ]

            # If mixed views, compute separate stats for each type
            if naked_indices and scaffolded_indices:
                # Scaffolded group
                scaffolded_rewards = [rewards[i] for i in scaffolded_indices]
                s_mean = sum(scaffolded_rewards) / len(scaffolded_rewards)
                s_var = sum((r - s_mean) ** 2 for r in scaffolded_rewards) / len(scaffolded_rewards)
                s_std = max(float(s_var**0.5), 1e-8)

                # Naked group
                naked_rewards = [rewards[i] for i in naked_indices]
                n_mean = sum(naked_rewards) / len(naked_rewards)
                n_var = sum((r - n_mean) ** 2 for r in naked_rewards) / len(naked_rewards)
                n_std = max(float(n_var**0.5), 1e-8)

                # Build advantages per type
                advantages = [0.0] * len(rewards)
                for _i, idx in enumerate(scaffolded_indices):
                    advantages[idx] = (rewards[idx] - s_mean) / s_std
                for _i, idx in enumerate(naked_indices):
                    advantages[idx] = (rewards[idx] - n_mean) / n_std
            else:
                # No mixed views, use global normalization
                advantages = [(r - mean_r) / std_r for r in rewards]
        else:
            advantages = [mean_r - self._reward_baseline] * len(rewards)

        # Correct rollouts must always get positive advantage.
        # DR-GRPO baseline drift and GRPO group normalization can both
        # produce negative advantages for correct answers — this kills
        # the training signal entirely (model learns nothing from success).
        _correct_floor = 0.1
        for i, r in enumerate(rewards):
            if r >= 0.5 and advantages[i] < _correct_floor:
                advantages[i] = _correct_floor

        return SampleGroupResult(
            group_rollouts=group_rollouts,
            rewards=rewards,
            advantages=advantages,
            mean_reward=mean_r,
            prompts=[sample.prompt] * len(group_rollouts),
            answers=[sample.answer or ""] * len(group_rollouts),
            targets=[ground_truth or ""] * len(group_rollouts),
            pending_events=pending_events,
            tui_ctx=(merged_type_config, reward_fn, prompt_tokens, callbacks),
            generation_sec=_generation_sec,
        )

    def _adaptive_math_rollouts(
        self,
        sample: Sample,
        task_type: TaskType,
        merged_type_config: dict[str, Any],
        prompt_tokens: list[int] | None,
        prompt_cache: Any,
        iteration: int,
        max_completion_tokens: int,
        pending_events: list | None = None,
    ) -> list[RolloutResult]:
        """Adaptive escalation for math: 2–6 views depending on difficulty.

        Runs up to max_escalations attempts with increasing prefix lengths.
        Stops early when model lands (stop_sequence). Always appends a naked
        view. All attempts go into group_rollouts for advantage computation.
        """
        cfg = merged_type_config
        # The global max_completion_tokens is often a small per-call Metal
        # safety value (e.g. 20). For thinking-heavy types, use thinking_budget
        # + answer_max_tokens as the actual rollout budget (take the larger).
        _tb = cfg.get("thinking_budget", 2048)
        _ab = cfg.get("answer_max_tokens", 128)
        _ceil = cfg.get("thinking_budget_ceiling", 4096)
        # max_grad_seq_len limits gradient computation, not generation.
        # Per-type rollout_max_tokens lets math generate longer than the gradient
        # seq limit — safe when used_tokens handles gradient truncation.
        _grad_cap = getattr(self.config, "max_grad_seq_len", 1200)
        _hard_cap = cfg.get("rollout_max_tokens") or _grad_cap
        effective_max_tokens = max(
            max_completion_tokens,
            min(_tb + _ab, _ceil + _ab, _hard_cap),
        )
        escalation_step = cfg.get("escalation_step_tokens", 128)
        max_escalations = cfg.get("max_escalations", 5)
        postfix_len = cfg.get("postfix_tokens", 128)
        naked_count = max(1, int(cfg.get("naked_count", 1)))
        _naked_temps_raw = cfg.get("naked_temperatures", None)
        if isinstance(_naked_temps_raw, int | float):
            naked_temps: list[float] = [float(_naked_temps_raw)] * naked_count
        elif isinstance(_naked_temps_raw, list) and _naked_temps_raw:
            naked_temps = [float(t) for t in _naked_temps_raw]
            # pad to naked_count by repeating the last value
            while len(naked_temps) < naked_count:
                naked_temps.append(naked_temps[-1])
        else:
            _base_temp = float(cfg.get("temperature", 0.7))
            naked_temps = [_base_temp] * naked_count
        rollout_fn = getattr(task_type, "escalation_rollout", task_type.rollout)
        callbacks = self._build_callbacks(task_type)
        target_text = sample.answer or ""
        group_rollouts: list[RolloutResult] = []
        landed_at = -1
        _cd = getattr(self.config, "rollout_cooldown", 0.5)

        # Pre-compute target token count (thinking content only, before </think>).
        # Skip attempts where prefix_len >= thinking tokens — injecting the full
        # thinking trace leaves the model nothing to generate.
        # Crucially we stop at </think> so the count excludes the closing tag and the
        # \boxed{answer} tokens (which must not appear in the injected prefix either).
        _t = target_text.lstrip()
        if _t.startswith("<think>"):
            _t = _t[len("<think>") :]
        _think_end = _t.find("</think>")
        if _think_end != -1:
            _t = _t[:_think_end]
        _target_tok_count = len(self.backend.encode(_t)) if _t else 0

        for attempt_idx in range(max_escalations):
            prefix_len = attempt_idx * escalation_step
            if prefix_len > 0 and _target_tok_count > 0 and prefix_len >= _target_tok_count:
                break  # Prefix would cover entire target — no generation possible
            prefix_text = _prefix_from_thinking(target_text, prefix_len, self.backend)

            # Final attempt: inject postfix so model sees start+end of thinking.
            # Use thinking content only (stop at </think>) — if the postfix included
            # </think>\boxed{answer}, the model would continue generating tokens after
            # the answer and never produce the </think> stop sequence itself.
            postfix_text: str | None = None
            if attempt_idx == max_escalations - 1:
                t = target_text.lstrip()
                if t.startswith("<think>"):
                    t = t[len("<think>") :]
                _pe = t.find("</think>")
                if _pe != -1:
                    t = t[:_pe]
                toks = self.backend.encode(t)
                if len(toks) > postfix_len:
                    postfix_text = self.backend.decode(toks[-postfix_len:])

            batch_ctx = BatchContext(
                iteration=iteration,
                total_iters=self.config.iterations,
                group_index=attempt_idx,
                group_size=max_escalations + naked_count,
                scaffold=ScaffoldConfig(mode="prefix" if prefix_len > 0 else "none"),
                custom={"postfix_text": postfix_text} if postfix_text else {},
            )
            with _ROLLOUT_LOCK:
                result = self.executor.execute(
                    rollout_fn,
                    prompt=sample.prompt,
                    prompt_tokens=prompt_tokens,
                    answer=sample.answer,
                    type_name=sample.type_name,
                    type_config=merged_type_config,
                    batch=batch_ctx,
                    callbacks=callbacks,
                    sample=sample.metadata,
                    max_completion_tokens=effective_max_tokens,
                    prompt_cache=prompt_cache,
                    pre_inject_text=prefix_text or None,
                )
                try:
                    import mlx.core as _mx

                    _mx.synchronize()
                except (ImportError, AttributeError):
                    pass
                _metal_clear_cache()
            result.metadata["_escalation_attempt"] = attempt_idx
            result.metadata["_escalation_prefix_len"] = prefix_len
            result.metadata["_view_name"] = f"escalation_{attempt_idx}"
            result.metadata["_view_is_naked"] = False
            group_rollouts.append(result)
            _am_ev = StepMetrics(
                iteration=iteration,
                extras={
                    "phase": "rollout",
                    "group_index": attempt_idx + 1,
                    "group_size": max_escalations + naked_count,
                    "type_name": sample.type_name,
                },
            )
            if pending_events is not None:
                pending_events.append(("on_group", _am_ev))
            else:
                self._fire("on_group", _am_ev)
            if _cd > 0:
                time.sleep(_cd)

            if result.metadata.get("thinking_stop_reason") == "stop_sequence":
                # Model closed </think> cleanly → no more prefix needed.
                # Even if this attempt scores reward=0, the naked view below
                # provides contrastive signal. Zero-advantage fallback in
                # _train_step (EMA baseline) handles the uniform-reward case.
                landed_at = attempt_idx
                break

        # Run naked_count naked views — full rollout() with no scaffold injection.
        # Each gets its own temperature for variance; initial_ratio=0.0 suppresses
        # math rollout's internal curriculum scaffold.
        total_members = max_escalations + naked_count
        for nk_idx in range(naked_count):
            nk_temp = naked_temps[nk_idx]
            naked_cfg = {**merged_type_config, "temperature": nk_temp}
            naked_batch = BatchContext(
                iteration=iteration,
                total_iters=self.config.iterations,
                group_index=len(group_rollouts),
                group_size=total_members,
                scaffold=ScaffoldConfig(mode="none", initial_ratio=0.0),
            )
            with _ROLLOUT_LOCK:
                naked_result = self.executor.execute(
                    task_type.rollout,
                    prompt=sample.prompt,
                    prompt_tokens=prompt_tokens,
                    answer=sample.answer,
                    type_name=sample.type_name,
                    type_config=naked_cfg,
                    batch=naked_batch,
                    callbacks=callbacks,
                    sample=sample.metadata,
                    max_completion_tokens=effective_max_tokens,
                    prompt_cache=prompt_cache,
                )
                try:
                    import mlx.core as _mx

                    _mx.synchronize()
                except (ImportError, AttributeError):
                    pass
                _metal_clear_cache()
            view_name = "naked" if naked_count == 1 else f"naked_{nk_idx}"
            naked_result.metadata["_view_name"] = view_name
            naked_result.metadata["_view_is_naked"] = True
            naked_result.metadata["_escalation_attempt"] = len(group_rollouts)
            naked_result.metadata["_naked_index"] = nk_idx
            naked_result.metadata["_naked_temperature"] = nk_temp
            group_rollouts.append(naked_result)
            if _cd > 0:
                time.sleep(_cd)

        n_att = len(group_rollouts) - naked_count
        naked_ok = all(
            r.metadata.get("thinking_stop_reason") == "stop_sequence"
            for r in group_rollouts[-naked_count:]
        )
        print(
            f"[math/escalation] attempts={n_att} landed_at={landed_at} "
            f"naked_count={naked_count} naked={'✓' if naked_ok else '✗'}"
        )
        return group_rollouts

    def _dynamic_view_rollouts(
        self,
        sample: Sample,
        task_type: TaskType,
        merged_type_config: dict[str, Any],
        prompt_tokens: list[int] | None,
        prompt_cache: Any,
        iteration: int,
        max_completion_tokens: int,
        pending_events: list | None = None,
    ) -> list[RolloutResult]:
        """Dynamic views: N equal-fraction prefix-scaffold views + 1 naked.

        For n_additional=N views, view k (k=0..N-1) injects the first
        (k+1)/N fraction of the target thinking trace as a masked prefix.
        The model must then close </think> itself and produce an answer.
        Recovery fires automatically when thinking hits max_tokens without
        producing </think> (existing RolloutContext mechanism).

        The naked view always runs last to provide unscaffolded contrast.
        """
        dv = self._dv_cfg
        if dv.completion == "model_opens":
            raise NotImplementedError(
                "dynamic_views completion='model_opens' is not yet implemented. "
                "Use completion='model_closes' (default) — the model generates thinking "
                "freely and must reach </think> before answering."
            )
        # Warn on unrecognized targets; "thinking", "answer", "both" are all implemented
        if dv.target not in ("thinking", "answer", "both"):
            import warnings

            warnings.warn(
                f"dynamic_views target='{dv.target}' is not recognized. "
                "Supported values: 'thinking', 'answer', 'both'. Falling back to 'thinking'.",
                stacklevel=4,
            )
        n = dv.n_additional
        group_size = n + 1
        callbacks = self._build_callbacks(task_type)
        _cd = getattr(self.config, "rollout_cooldown", 0.5)
        group_rollouts: list[RolloutResult] = []

        # Per-type effective token budget — overrides global max_completion_tokens
        # so that type_overrides.mcq.thinking_budget (350) + answer_max_tokens (24)
        # is not silently capped by generation.max_tokens (64).
        _tb = merged_type_config.get("thinking_budget", 512)
        _ab = merged_type_config.get("answer_max_tokens", 32)
        _rollout_max = merged_type_config.get("rollout_max_tokens", _tb + _ab)
        effective_max_tokens = max(max_completion_tokens, _rollout_max)

        # Tokenize the scaffold source based on target field
        target_text = sample.answer or ""
        if dv.target == "answer":
            # Use only the answer portion (text after </think>)
            _think_end = target_text.find("</think>")
            t = (
                target_text[_think_end + len("</think>") :].lstrip()
                if _think_end != -1
                else target_text
            )
        else:
            # "thinking" (default) or "both": strip opening <think> tag, keep rest
            t = target_text.lstrip()
            if t.startswith("<think>"):
                t = t[len("<think>") :]
        target_tokens = self.backend.encode(t)
        n_target = len(target_tokens)

        if iteration == 0:
            print(
                f"{'=' * 60}\nDynamic Multi-View Scaffold ({n} scaffold + 1 naked)\n"
                f"  target={dv.target}  completion={dv.completion}\n"
                f"  target_tokens={n_target}  max_scaffold={dv.max_scaffold_tokens}\n"
                f"{'=' * 60}"
            )

        for k in range(n):
            fraction = (k + 1) / n
            n_scaffold = min(int(fraction * n_target), dv.max_scaffold_tokens)
            prefix_text = self.backend.decode(target_tokens[:n_scaffold]) if n_scaffold > 0 else ""
            batch_ctx = BatchContext(
                iteration=iteration,
                total_iters=self.config.iterations,
                group_index=k,
                group_size=group_size,
                # initial_ratio=0.0 suppresses internal curriculum scaffold in
                # task type rollouts (MCQ/math check ratio > min_ratio=0.05).
                # pre_inject_text handles the actual prefix injection.
                scaffold=ScaffoldConfig(mode="none", initial_ratio=0.0),
            )
            with _ROLLOUT_LOCK:
                result = self.executor.execute(
                    task_type.rollout,
                    prompt=sample.prompt,
                    prompt_tokens=prompt_tokens,
                    answer=sample.answer,
                    type_name=sample.type_name,
                    type_config=merged_type_config,
                    batch=batch_ctx,
                    callbacks=callbacks,
                    sample=sample.metadata,
                    max_completion_tokens=effective_max_tokens,
                    prompt_cache=prompt_cache,
                    pre_inject_text=prefix_text or None,
                )
                try:
                    import mlx.core as _mx

                    _mx.synchronize()
                except (ImportError, AttributeError):
                    pass
                _metal_clear_cache()
            result.metadata.update(
                {
                    "_view_name": f"dynamic_{k}",
                    "_view_fraction": fraction,
                    "_scaffold_tokens": n_scaffold,
                    "_view_is_naked": False,
                }
            )
            group_rollouts.append(result)
            landed = result.metadata.get("thinking_stop_reason") == "stop_sequence"
            print(
                f"  [dynamic] view={k + 1}/{n} frac={fraction:.2f} "
                f"scaffold={n_scaffold}tok landed={'✓' if landed else '✗'}"
            )
            _dv_ev = StepMetrics(
                iteration=iteration,
                extras={
                    "phase": "rollout",
                    "group_index": k + 1,
                    "group_size": group_size,
                    "type_name": sample.type_name,
                },
            )
            if pending_events is not None:
                pending_events.append(("on_group", _dv_ev))
            else:
                self._fire("on_group", _dv_ev)
            if _cd > 0:
                time.sleep(_cd)

        # Naked view — always last, no scaffold
        naked_batch = BatchContext(
            iteration=iteration,
            total_iters=self.config.iterations,
            group_index=n,
            group_size=group_size,
            scaffold=ScaffoldConfig(mode="none", initial_ratio=0.0),
        )
        with _ROLLOUT_LOCK:
            naked_result = self.executor.execute(
                task_type.rollout,
                prompt=sample.prompt,
                prompt_tokens=prompt_tokens,
                answer=sample.answer,
                type_name=sample.type_name,
                type_config=merged_type_config,
                batch=naked_batch,
                callbacks=callbacks,
                sample=sample.metadata,
                max_completion_tokens=effective_max_tokens,
                prompt_cache=prompt_cache,
            )
            try:
                import mlx.core as _mx

                _mx.synchronize()
            except (ImportError, AttributeError):
                pass
            _metal_clear_cache()
        naked_result.metadata.update({"_view_name": "naked", "_view_is_naked": True})
        group_rollouts.append(naked_result)
        naked_ok = naked_result.metadata.get("thinking_stop_reason") == "stop_sequence"
        print(f"  [dynamic] naked landed={'✓' if naked_ok else '✗'}")
        if _cd > 0:
            time.sleep(_cd)
        return group_rollouts

    def _reset_optimizer(self) -> None:
        """Clear optimizer momentum state (Adam m1/m2).

        Resets accumulated first and second moments to zero so the next
        gradient step starts from scratch. Useful for periodic restarts
        (see optimizer_reset_every config key).
        """
        if self.optimizer is None:
            return
        if hasattr(self.optimizer, "state"):
            self.optimizer.state = {}
            print(f"  [optimizer] state reset at iteration {self._iteration}")

    def _do_gradient_step(
        self,
        rollouts: list[RolloutResult],
        advantages: list[float],
    ) -> tuple[float, float, float]:
        """Compute loss and update model weights.

        Returns (loss, kl_divergence, grad_norm).

        When model+optimizer are available: runs MLX gradient step
        (forward pass inside value_and_grad → proper autograd).

        When not available (testing/eval): computes loss analytically
        from pre-computed log probs, or returns reward-based pseudo-loss.

        GAR mode: bypasses GRPO loss entirely, uses activation drift loss.
        """
        # ── Validate backend supports training ──
        if self.backend and self.backend.name == "vllm":
            raise NotImplementedError(
                "vLLM backend is inference-only and does not support training. "
                "Use backend: 'pytorch' or backend: 'mlx' for GRPO training. "
                "vLLM excels at rollout generation and can be used for evaluation."
            )

        # ── GAR mode: activation drift loss ──
        if self._gar_mode:
            return self._gar_gradient_step(rollouts)

        # ── Path A: Full MLX training (model + optimizer available) ──
        if (
            self._mlx_ops_available
            and self.backend
            and self.backend.name in ("mlx", "mlx_batch_generate")
        ):
            return self._mlx_gradient_step(rollouts, advantages)

        # ── Path A2: Full PyTorch training (model + optimizer available) ──
        if self._torch_ops_available and self.backend and self.backend.name == "pytorch":
            return self._torch_gradient_step(rollouts, advantages)

        # ── Debug: log which path is taken ──
        with open("/tmp/_grpo_grad_debug.log", "a") as _f:
            _f.write(
                f"GRAD DISPATCH: torch_ops={self._torch_ops_available} "
                f"backend={self.backend.name if self.backend else None} "
                f"model={type(self.model).__name__} "
                f"n_rollouts={len(rollouts)} n_adv={len(advantages)}\n"
            )

        # ── Path B: Analytical loss (log probs available but no model) ──
        log_probs = [r.log_probs for r in rollouts if r.log_probs]
        if log_probs:
            return self._analytical_loss_step(rollouts, advantages)

        # ── Path C: Reward-only tracking (testing/mock backend) ──
        return self._reward_only_step(advantages)

    def _get_marker_ids(self, text: str | None = None) -> list[int]:
        """Tokenize a truncation marker and cache. Defaults to global marker."""
        marker = text if text is not None else self._truncate_marker
        if marker not in self._marker_ids:
            if self.tokenizer is not None:
                self._marker_ids[marker] = self.tokenizer.encode(marker)
            else:
                self._marker_ids[marker] = []
        return self._marker_ids[marker]

    def _truncate_completions(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        masks: list[list[bool]],
        limits: list[int | None] | None = None,
        modes: list[str] | None = None,
        markers: list[str] | None = None,
        marker_maskeds: list[bool] | None = None,
        snaps: list[bool] | None = None,
    ) -> tuple[list[list[int]], list[int], list[list[bool]]]:
        """Truncate completion tokens to used_tokens for loss calculation.

        Keeps prompt intact. Replaces truncated portions with marker tokens.
        When snap_to_boundary=True, adjusts cut point to nearest sentence end.

        Args:
            limits: Per-sequence used_tokens (None = use global).
            modes: Per-sequence truncate_from (None = use global).
            markers: Per-sequence marker text (None = use global).
            marker_maskeds: Per-sequence marker masking (None = use global).
            snaps: Per-sequence snap_to_boundary (None = use global).
        """
        global_limit = self._used_tokens
        if limits is None and global_limit is None:
            return sequences, prompt_lengths, masks

        out_seqs, out_pls, out_masks = [], [], []

        for i, (seq, pl, mask) in enumerate(
            zip(sequences, prompt_lengths, masks, strict=True),
        ):
            limit = limits[i] if limits is not None else global_limit
            mode = modes[i] if modes is not None else self._truncate_from
            marker_text = markers[i] if markers is not None else None
            masked = (
                marker_maskeds[i] if marker_maskeds is not None else self._truncate_marker_masked
            )

            if limit is None:
                out_seqs.append(seq)
                out_pls.append(pl)
                out_masks.append(mask)
                continue

            comp_len = len(seq) - pl
            if comp_len <= limit:
                out_seqs.append(seq)
                out_pls.append(pl)
                out_masks.append(mask)
                continue

            ell = self._get_marker_ids(marker_text)
            mm = not masked  # mask value: True = included in loss, False = excluded
            prompt = seq[:pl]
            comp = seq[pl:]
            cmask = mask[pl:]
            snap = snaps[i] if snaps is not None else self._snap_to_boundary

            if mode == "postfix":
                keep = max(limit - len(ell), 0)
                if snap and self.tokenizer and 0 < keep < comp_len:
                    keep = self._snap_cut(comp, keep, direction="after")
                new_comp = ell + comp[-keep:] if keep else ell[:limit]
                new_mask = [mm] * len(ell) + cmask[-keep:] if keep else [mm] * limit
            elif mode == "middle":
                # Excise the middle: keep prefix + suffix, one marker between them.
                # Rationale: preserves start-of-reasoning context AND final answer for loss,
                # while dropping the middle chunk that carries least gradient signal.
                elen = len(ell)
                keep = max(limit - elen, 0)
                keep_pre = keep // 2
                keep_suf = keep - keep_pre
                if snap and self.tokenizer:
                    if 0 < keep_pre < comp_len:
                        keep_pre = self._snap_cut(comp, keep_pre, direction="before")
                    if 0 < keep_suf < comp_len:
                        keep_suf = self._snap_cut(comp, keep_suf, direction="after")
                suf_start = comp_len - keep_suf if keep_suf else comp_len
                new_comp = comp[:keep_pre] + ell + comp[suf_start:]
                new_mask = cmask[:keep_pre] + [mm] * elen + cmask[suf_start:]
            else:  # prefix (default)
                keep = max(limit - len(ell), 0)
                if snap and self.tokenizer and 0 < keep < comp_len:
                    keep = self._snap_cut(comp, keep, direction="before")
                new_comp = comp[:keep] + ell
                new_mask = cmask[:keep] + [mm] * len(ell)

            out_seqs.append(prompt + new_comp)
            out_pls.append(pl)
            out_masks.append(mask[:pl] + new_mask)

        return out_seqs, out_pls, out_masks

    def _snap_cut(
        self,
        comp: list[int],
        keep: int,
        direction: str = "before",
        window: int = 40,
    ) -> int:
        """Snap a token cut point to the nearest sentence boundary.

        direction="before": walk backward from keep, find last sentence end.
        direction="after": for postfix — snap start of kept portion forward.
        Returns adjusted keep count.
        """
        if direction == "before":
            lo = max(1, keep - window)
            for i in range(keep, lo - 1, -1):
                text = self.tokenizer.decode(comp[:i]).rstrip(" ")
                if text and text[-1] in ".!?\n":
                    return i
        else:
            # Postfix: snap the start (comp_len - keep) forward to a boundary
            start = len(comp) - keep
            hi = min(len(comp), start + window)
            for i in range(start, hi):
                text = self.tokenizer.decode(comp[:i]).rstrip(" ")
                if text and text[-1] in ".!?\n":
                    return len(comp) - i
        return keep

    @staticmethod
    def _extract_segment_masks(
        rollouts: list[RolloutResult],
    ) -> tuple[list[list[bool]], list[list[bool]], list[list[bool]]]:
        """Extract per-completion-token answer/thinking/scaffold boolean masks.

        Used by token-level IS to weight gradients by segment type and
        scaffold proximity. Scaffold mask is True for INJECTED segments
        (the scaffold tokens themselves, which are already masked from loss).
        """
        from mlx_grpo.engine.result import SegmentSource

        answer_masks: list[list[bool]] = []
        thinking_masks: list[list[bool]] = []
        scaffold_masks: list[list[bool]] = []
        for r in rollouts:
            n = len(r.completion_tokens)
            ans_mask = [False] * n
            think_mask = [False] * n
            scaf_mask = [False] * n
            idx = 0
            for s in r.segments:
                if s.source == SegmentSource.PROMPT:
                    continue
                # Scaffold mask: only content scaffold injections (tag="scaffold_*").
                # Answer-framing injections (answer_prefix, closing "}", </think>)
                # are excluded — they'd make all thinking tokens appear "far from scaffold".
                is_content_scaffold = s.source == SegmentSource.INJECTED and (
                    s.tag or ""
                ).startswith("scaffold")
                for _ in s.tokens:
                    if idx < n:
                        if s.tag == "answer":
                            ans_mask[idx] = True
                        elif s.tag == "thinking":
                            think_mask[idx] = True
                        if is_content_scaffold:
                            scaf_mask[idx] = True
                        idx += 1
            answer_masks.append(ans_mask)
            thinking_masks.append(think_mask)
            scaffold_masks.append(scaf_mask)
        return answer_masks, thinking_masks, scaffold_masks

    def _mlx_gradient_step(  # pragma: no cover
        self,
        rollouts: list[RolloutResult],
        advantages: list[float],
    ) -> tuple[float, float, float]:
        """Real MLX gradient step with gradient accumulation support.

        When gradient_accumulation_steps > 1, computes grads and accumulates
        them, only applying the optimizer update on the boundary step.
        """
        sequences, prompt_lengths, masks = self._extract_fn(rollouts, self.backend)

        # ── Per-type loss overrides (beta, epsilon, used_tokens, etc.) ──
        eps = getattr(self.config, "epsilon", 0.0)
        eps_high = getattr(self.config, "epsilon_high", None)
        _has_lo = any(r.metadata.get("_loss_overrides") for r in rollouts)
        if _has_lo:
            _los = [r.metadata.get("_loss_overrides", {}) for r in rollouts]
            betas = [lo.get("beta", self.config.beta) for lo in _los]
            epsilons = [lo.get("epsilon", eps) for lo in _los]
            eps_highs = [lo.get("epsilon_high", eps_high) for lo in _los]
            grad_norms = [lo.get("max_grad_norm", self._max_grad_norm) for lo in _los]
            ut_lims = [lo.get("used_tokens", self._used_tokens) for lo in _los]
            ut_modes = [lo.get("truncate_from", self._truncate_from) for lo in _los]
            ut_markers = [lo.get("truncate_marker", self._truncate_marker) for lo in _los]
            ut_mm = [lo.get("truncate_marker_masked", self._truncate_marker_masked) for lo in _los]
            ut_snap = [lo.get("snap_to_boundary", self._snap_to_boundary) for lo in _los]
        else:
            betas, epsilons, eps_highs = self.config.beta, eps, eps_high
            grad_norms = None
            ut_lims = ut_modes = ut_markers = ut_mm = ut_snap = None

        sequences, prompt_lengths, masks = self._truncate_completions(
            sequences,
            prompt_lengths,
            masks,
            limits=ut_lims,
            modes=ut_modes,
            markers=ut_markers,
            marker_maskeds=ut_mm,
            snaps=ut_snap,
        )

        if not sequences or not any(any(m) for m in masks):
            return 0.0, 0.0, 0.0

        # ── Token-level IS weights (computed after truncation for accurate length) ──
        _caw = getattr(self._is_cfg, "correct_answer_weight", 1.0) if self._is_cfg else 1.0
        _iaw = getattr(self._is_cfg, "incorrect_answer_weight", 1.0) if self._is_cfg else 1.0
        _has_answer_w = _caw != 1.0 or _iaw != 1.0
        _has_is = (
            self._is_cfg is not None
            or any(r.metadata.get("_is_override") for r in rollouts)
            or any(r.metadata.get("_esc_weights") for r in rollouts)
            or _has_answer_w
        )
        is_seg_weights: list[list[float] | None] | None = None
        if _has_is:
            from dataclasses import replace as _dc_replace

            from mlx_grpo.training.token_importance import compute_is_weights

            ans_masks, think_masks, scaf_masks = self._extract_segment_masks(rollouts)
            _sw: list[list[float] | None] = []

            # Per-rollout near/far proximity assignment for proximity_member_split < 1.0.
            # Among scaffolded members within each group, the first n_near use near-proximity
            # (default); the remainder use far-proximity (invert_proximity=True).
            _pms = getattr(self._is_cfg, "proximity_member_split", 1.0) if self._is_cfg else 1.0
            _gs = getattr(self.config, "group_size", 1)
            _use_far: list[bool] = [False] * len(rollouts)
            if _pms < 1.0:
                for _g0 in range(0, len(rollouts), _gs):
                    _g1 = min(_g0 + _gs, len(rollouts))
                    _scaf_idxs = [
                        _ii
                        for _ii in range(_g0, _g1)
                        if any((scaf_masks[_ii] or [])[: len(masks[_ii])])
                    ]
                    _n_near = round(len(_scaf_idxs) * _pms) if _scaf_idxs else 0
                    for _jj, _idx in enumerate(_scaf_idxs):
                        _use_far[_idx] = _jj >= _n_near

            def _neutralize_seg_answer(cfg: Any) -> Any:
                """When reward-based answer weighting is active alongside mode=segment,
                neutralize the uniform answer_weight so the correctness-based pass
                is the sole answer multiplier (prevents unintended double application).
                """
                if _has_answer_w and cfg.mode == "segment" and cfg.answer_weight != 1.0:
                    return _dc_replace(cfg, answer_weight=1.0)
                return cfg

            for i, rollout in enumerate(rollouts):
                esc_w = rollout.metadata.get("_esc_weights")
                if esc_w is not None:
                    n = len(masks[i])
                    combined: list[float] = esc_w[:n]
                    # Combine with IS config when set — multiply element-wise so both
                    # mechanisms contribute. Naked member (esc_w=[1.0]*n, no scaffold →
                    # proximity=[1.0]*n) produces no change; scaffolded members get
                    # tri_factor × proximity_scale weights.
                    _is_cfg_esc = rollout.metadata.get("_is_override") or self._is_cfg
                    if _is_cfg_esc is not None and _is_cfg_esc.mode != "none":
                        if _use_far[i] and _is_cfg_esc.mode == "scaffold_proximity":
                            _pd_f = getattr(_is_cfg_esc, "proximity_decay_far", None)
                            _is_cfg_esc = _dc_replace(
                                _is_cfg_esc,
                                invert_proximity=True,
                                **{"proximity_decay": _pd_f} if _pd_f is not None else {},
                            )
                        prox = compute_is_weights(
                            n,
                            _neutralize_seg_answer(_is_cfg_esc),
                            ans_masks[i],
                            think_masks[i],
                            scaf_masks[i],
                        )
                        if prox is not None:
                            combined = [e * p for e, p in zip(combined, prox, strict=False)]
                    _sw.append(combined)
                    continue
                is_cfg = rollout.metadata.get("_is_override") or self._is_cfg
                if is_cfg is None or is_cfg.mode == "none":
                    _sw.append(None)
                else:
                    n = len(masks[i])
                    _active_cfg = is_cfg
                    if _use_far[i] and is_cfg.mode == "scaffold_proximity":
                        _pd_f = getattr(is_cfg, "proximity_decay_far", None)
                        _active_cfg = _dc_replace(
                            is_cfg,
                            invert_proximity=True,
                            **{"proximity_decay": _pd_f} if _pd_f is not None else {},
                        )
                    _sw.append(
                        compute_is_weights(
                            n,
                            _neutralize_seg_answer(_active_cfg),
                            ans_masks[i],
                            think_masks[i],
                            scaf_masks[i],
                        )
                    )
            # ── Reward-based answer token weighting ──────────────────────────────
            # Multiply answer-segment tokens by correct_answer_weight or
            # incorrect_answer_weight depending on the rollout's raw reward.
            # Runs on top of any IS/esc weights already in _sw (multiplicative).
            if _has_answer_w:
                _awt = getattr(self._is_cfg, "answer_weight_threshold", 0.5)
                for i, rollout in enumerate(rollouts):
                    rw = rollout.metadata.get("_reward")
                    if rw is None:
                        continue
                    w = _caw if rw >= _awt else _iaw
                    if w == 1.0:
                        continue
                    n = len(masks[i])
                    base = _sw[i] if _sw[i] is not None else [1.0] * n
                    _sw[i] = [v * w if ans_masks[i][j] else v for j, v in enumerate(base[:n])]
            is_seg_weights = None if all(w is None for w in _sw) else _sw
            # Stamp computed weights onto metadata so rollout_logger captures them.
            if is_seg_weights is not None:
                for i, rollout in enumerate(rollouts):
                    w = is_seg_weights[i]
                    if w is not None:
                        rollout.metadata["_is_weights"] = w

        accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        lt = getattr(self.config, "loss_type", "grpo")
        msl = getattr(self.config, "max_grad_seq_len", 1024)
        gc = getattr(self.config, "gpu_cooldown", 1.0)
        vg = getattr(self.config, "verbose_grad", False)
        if vg and is_seg_weights is not None:
            _scale = getattr(self._is_cfg, "proximity_scale", 1.0) if self._is_cfg else 1.0
            _mode = getattr(self._is_cfg, "mode", "?") if self._is_cfg else "?"
            for i, w in enumerate(is_seg_weights):
                if w:
                    mn, mx_w = min(w), max(w)
                    mean_w = sum(w) / len(w)
                    print(
                        f"  [is_weights] rollout {i}: n={len(w)}"
                        f"  min={mn:.3f}  mean={mean_w:.3f}  max={mx_w:.3f}"
                        f"  scale={_scale}  mode={_mode}"
                    )

        # MoE: balance entropy aux loss (maximize entropy = minimize -entropy)
        aux_fn = None
        if hasattr(self.model, "_last_gate_logits"):
            try:
                import mlx.core as _mx

                _bw = self.model._moe_cfg.routing.balance_loss_weight

                def aux_fn(m, _bw=_bw):
                    if m._last_gate_logits is None:
                        return _mx.array(0.0)
                    w = _mx.softmax(m._last_gate_logits, axis=-1)
                    entropy = -_mx.sum(w * _mx.log(w + 1e-8))
                    return -entropy * _bw  # maximize entropy
            except (ImportError, AttributeError):
                aux_fn = None

        if accum_steps <= 1:
            # No accumulation — single step
            result = self._gradient_step_fn(
                model=self.model,
                ref_model=self.ref_model,
                optimizer=self.optimizer,
                sequences=sequences,
                prompt_lengths=prompt_lengths,
                masks=masks,
                advantages=advantages,
                beta=betas,
                max_grad_norm=self._max_grad_norm,
                segment_weights=is_seg_weights,
                epsilon=epsilons,
                epsilon_high=eps_highs,
                loss_type=lt,
                max_seq_len=msl,
                gpu_cooldown=gc,
                per_seq_grad_norm=grad_norms,
                aux_loss_fn=aux_fn,
                verbose=vg,
            )
            # MoE: advance schedule counter after weights are updated
            if hasattr(self.model, "step"):
                self.model.step()
            return result.loss, result.kl_divergence, result.grad_norm

        # Gradient accumulation: compute grads, accumulate, apply on boundary
        loss, grads, grad_norm, _, _, kl = self._compute_grads_fn(
            model=self.model,
            ref_model=self.ref_model,
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            masks=masks,
            advantages=advantages,
            beta=betas,
            max_grad_norm=self._max_grad_norm,
            segment_weights=is_seg_weights,
            epsilon=epsilons,
            epsilon_high=eps_highs,
            loss_type=lt,
            max_seq_len=msl,
            gpu_cooldown=gc,
            per_seq_grad_norm=grad_norms,
            aux_loss_fn=aux_fn,
        )

        if not grads:  # All sequences skipped (adv=0, too long, etc.)
            return loss, kl, grad_norm
        if self._accum_grads is None:
            self._accum_grads = grads
        else:
            self._accum_grads = self._tree_add(self._accum_grads, grads)
            # Eval accumulated grads to prevent lazy chain growth across steps
            from mlx_grpo.loss.mlx_ops import _tree_eval

            _tree_eval(self._accum_grads)
        self._accum_count += 1
        if self._step_accum_mode:
            self._accum_count_this_iter += 1  # Track for this iteration

        # Apply on boundary (skip if in step-accumulation mode — will flush externally)
        if self._accum_count >= accum_steps and not self._step_accum_mode:
            avg_grads = self._tree_scale(self._accum_grads, 1.0 / self._accum_count)
            self._apply_grads_fn(self.model, self.optimizer, avg_grads)
            self._accum_grads = None
            self._accum_count = 0
            # MoE: advance schedule counter after weights are updated
            if hasattr(self.model, "step"):
                self.model.step()

        return loss, kl, grad_norm

    def _flush_step_accum(self) -> None:
        """Apply and reset accumulated step gradients (called at end of rep loop).

        Used when accumulate_over_step=True to defer the optimizer update until
        all sub-steps (reps) of an outer iteration are complete.
        """
        if self._accum_count > 0 and self._accum_grads is not None:
            avg_grads = self._tree_scale(self._accum_grads, 1.0 / self._accum_count)
            self._apply_grads_fn(self.model, self.optimizer, avg_grads)
            print(f"  Flushed {self._accum_count} gradient step(s) accumulated over repeats")
            self._accum_grads = None
            self._accum_count = 0
            # NOTE: keep _accum_count_this_iter for metrics reporting (reset after metrics capture)
            if hasattr(self.model, "step"):
                self.model.step()

    def _torch_gradient_step(  # pragma: no cover
        self,
        rollouts: list[RolloutResult],
        advantages: list[float],
    ) -> tuple[float, float, float]:
        """PyTorch gradient step with gradient accumulation support."""
        sequences, prompt_lengths, masks = self._extract_fn(rollouts, self.backend)
        sequences, prompt_lengths, masks = self._truncate_completions(
            sequences,
            prompt_lengths,
            masks,
        )

        # Debug: trace gradient step entry
        with open("/tmp/_grpo_grad_debug.log", "a") as _f:
            _f.write(
                f"_torch_gradient_step: n_seq={len(sequences)} "
                f"n_adv={len(advantages)} "
                f"any_mask={any(any(m) for m in masks) if masks else False} "
                f"seq_lens={[len(s) for s in sequences[:5]]} "
                f"mask_lens={[len(m) for m in masks[:5]]} "
                f"mask_active={[sum(m) for m in masks[:5]]}\n"
            )

        if not sequences or not any(any(m) for m in masks):
            with open("/tmp/_grpo_grad_debug.log", "a") as _f:
                _f.write("EARLY RETURN: no valid sequences or all masks empty!\n")
            return 0.0, 0.0, 0.0

        eps = getattr(self.config, "epsilon", 0.0)
        eps_high = getattr(self.config, "epsilon_high", None)
        accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        device = self._torch_device

        if accum_steps <= 1:
            result = self._torch_gradient_step_fn(
                model=self.model,
                ref_model=self.ref_model,
                optimizer=self.optimizer,
                sequences=sequences,
                prompt_lengths=prompt_lengths,
                masks=masks,
                advantages=advantages,
                beta=self.config.beta,
                max_grad_norm=self._max_grad_norm,
                epsilon=eps,
                epsilon_high=eps_high,
                device=device,
            )
            return result.loss, result.kl_divergence, result.grad_norm

        # Gradient accumulation: compute grads, accumulate, apply on boundary
        loss, grad_norm, _, _, kl = self._torch_compute_grads_fn(
            model=self.model,
            ref_model=self.ref_model,
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            masks=masks,
            advantages=advantages,
            beta=self.config.beta,
            epsilon=eps,
            epsilon_high=eps_high,
            device=device,
        )
        self._torch_accum_count += 1

        if self._torch_accum_count >= accum_steps:
            # Average grads accumulated by backward() across accum steps
            if self._torch_accum_count > 1:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.div_(self._torch_accum_count)
            self._torch_apply_grads_fn(self.model, self.optimizer)
            self._torch_accum_count = 0

        return loss, kl, grad_norm

    def _analytical_loss_step(  # pragma: no cover
        self,
        rollouts: list[RolloutResult],
        advantages: list[float],
    ) -> tuple[float, float, float]:
        """Compute loss analytically from pre-computed log probs.

        Used when log probs are available (e.g., from vLLM with logprobs)
        but we're not doing MLX gradient updates.
        """
        from mlx_grpo.loss import bnpo_loss, dr_grpo_loss, grpo_loss

        log_probs = [r.log_probs for r in rollouts]
        ref_log_probs = [r.ref_log_probs or r.log_probs for r in rollouts]
        # Use boosted masks if available (from Jaccard boosting), else use default loss_mask
        masks = [r.metadata.get("_jaccard_boosted_mask") or r.loss_mask for r in rollouts]

        loss_type = getattr(self.config, "loss_type", "grpo")

        if loss_type == "dr_grpo":
            result = dr_grpo_loss(
                log_probs, ref_log_probs, advantages, masks, beta=self.config.beta
            )
        elif loss_type == "bnpo":
            result = bnpo_loss(log_probs, ref_log_probs, advantages, masks, beta=self.config.beta)
        else:
            result = grpo_loss(log_probs, ref_log_probs, advantages, masks, beta=self.config.beta)

        return result.loss, result.kl_divergence, 0.0

    def _reward_only_step(
        self,
        advantages: list[float],
    ) -> tuple[float, float, float]:
        """Fallback when no log probs or model available.

        Returns a pseudo-loss from advantages for metric tracking.
        """
        if not advantages:
            return 0.0, 0.0, 0.0
        pseudo_loss = -sum(advantages) / len(advantages)
        return pseudo_loss, 0.0, 0.0

    def _gar_gradient_step(
        self,
        rollouts: list[RolloutResult],
    ) -> tuple[float, float, float]:
        """GAR activation drift gradient step.

        Pairs rollouts into (scaffolded, naked), captures activations,
        computes drift loss, and updates model weights. Stores GAR-specific
        metrics in self._gar_last_metrics for the training loop to read.

        Falls back to pseudo-loss when model/optimizer unavailable (test mode).
        """
        from .gar_step import pair_rollouts

        pairs = pair_rollouts(rollouts)
        if not pairs:
            self._gar_last_metrics = {}
            return 0.0, 0.0, 0.0

        # Without model/optimizer (mock/test), compute pseudo-loss from scaffold fraction
        if self.model is None or self.optimizer is None:
            total_scaf = sum(s.injected_token_count for s, _ in pairs)
            total_tok = sum(len(s.completion_tokens) + len(n.completion_tokens) for s, n in pairs)
            frac = total_scaf / max(total_tok, 1)
            self._gar_last_metrics = {
                "loss": 0.0,
                "per_layer_angles": {},
                "scaffold_fraction": frac,
                "grad_norm": 0.0,
            }
            return 0.0, 0.0, 0.0

        return self._gar_gradient_step_mlx(pairs)  # pragma: no cover

    def _gar_gradient_step_mlx(  # pragma: no cover
        self,
        pairs: list[tuple[RolloutResult, RolloutResult]],
    ) -> tuple[float, float, float]:
        """MLX-backed GAR gradient step. Requires real model + optimizer."""
        from .gar_step import gar_gradient_step

        lora_layers = list(range(self.config.lora_layers or 0))

        # IS weights: extract from rollouts (naked member of each pair)
        # If missing, create uniform weights for all tokens (no masking)
        _masks: list[list[float] | None] = []
        _pw: list[list[float] | None] = []
        for _scaf, naked in pairs:
            is_wts = naked.metadata.get("_is_weights")
            if is_wts is None:
                # No IS weights: create uniform 1.0 for all generated tokens
                is_wts = [1.0] * len(naked.completion_tokens)
            _masks.append(is_wts)
            _pw.append(None)

        result = gar_gradient_step(
            model=self.model,
            optimizer=self.optimizer,
            rollout_pairs=pairs,
            lora_layers=lora_layers,
            threshold_deg=getattr(self.config, "gar_drift_threshold_deg", 5.0),
            normalize=getattr(self.config, "gar_normalize", True),
            max_grad_norm=self._max_grad_norm,
            masks=_masks,
            proximity_weights=_pw,
        )

        self._gar_last_metrics = result
        return result["loss"], 0.0, result["grad_norm"]

    # ── SFT Anchor ──

    def _sft_anchor_step(self, batch: list[Sample]) -> None:  # pragma: no cover
        """Run an SFT anchor step on the batch samples to prevent format drift."""
        lr_mult = getattr(self.config, "sft_anchor_lr_multiplier", 0.1)
        sample = batch[0]

        # Tokenize the full gold sequence (prompt + answer)
        if sample.answer is None:
            return

        system_prompt = self._resolve_system_prompt(sample)
        messages = [
            *([{"role": "system", "content": system_prompt}] if system_prompt else []),
            {"role": "user", "content": sample.prompt},
        ]
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        answer_tokens = self.tokenizer.encode(sample.answer)
        full_seq = prompt_tokens + answer_tokens

        if self._mlx_ops_available:
            self._sft_step_fn(
                self.model,
                self.optimizer,
                full_seq,
                len(prompt_tokens),
                lr_mult,
            )
        elif self._torch_ops_available:
            self._torch_sft_step_fn(
                self.model,
                self.optimizer,
                full_seq,
                len(prompt_tokens),
                lr_mult,
                device=self._torch_device,
            )

    # ── Interrupt handling ──

    def _save_interrupt_checkpoint(self) -> None:  # pragma: no cover
        """Fire on_checkpoint so hooks save weights. Uses last completed metrics."""
        metrics = self._last_metrics or StepMetrics(iteration=self._iteration)
        try:
            self._fire("on_checkpoint", metrics)
        except Exception as e:
            print(f"  Warning: checkpoint save failed: {e}")

    def _graceful_shutdown(self, sig_name: str = "INTERRUPT") -> None:  # pragma: no cover
        """Gracefully shutdown by flushing accumulated gradients and saving checkpoint.

        Called on KeyboardInterrupt (Ctrl+C) or SIGTERM to ensure no gradient
        accumulation work is lost. Flushes both iteration-level and step-level
        accumulated gradients before saving checkpoint.
        """
        print(f"\n  {sig_name}: Gracefully shutting down at iteration {self._iteration}...")

        # Flush step-level accumulation (over repeats) if active
        if self._accum_count_this_iter > 0 and self._accum_grads is not None:
            print(
                f"  Flushing {self._accum_count_this_iter} step-level accumulated "
                f"gradient(s) (over repeats)..."
            )
            _avg = self._tree_scale(self._accum_grads, 1.0 / self._accum_count_this_iter)
            self._apply_grads_fn(self.model, self.optimizer, _avg)
            self._accum_grads = None
            self._accum_count = 0
            self._accum_count_this_iter = 0

        # Flush iteration-level accumulation (over entire iterations)
        if self._accum_count > 0 and self._accum_grads is not None:
            print(f"  Flushing {self._accum_count} iteration-level accumulated gradient(s)...")
            _avg = self._tree_scale(self._accum_grads, 1.0 / self._accum_count)
            self._apply_grads_fn(self.model, self.optimizer, _avg)
            self._accum_grads = None
            self._accum_count = 0

        # Save checkpoint
        print("  Saving checkpoint...")
        self._save_interrupt_checkpoint()
        print(f"  Shutdown complete. Training can be resumed from iteration {self._iteration}.")

    # ── Reporting ──

    def _report_step(self, step: float, total: int, metrics: StepMetrics) -> None:
        """Print formatted training metrics for the current step."""
        self._recent_losses.append(metrics.loss)
        self._recent_rewards.append(metrics.mean_reward)

        avg_loss = sum(self._recent_losses) / len(self._recent_losses)
        avg_reward = sum(self._recent_rewards) / len(self._recent_rewards)

        mem_str = ""
        if metrics.peak_memory_mb > 0:
            mem_str = f"  metal={metrics.peak_memory_mb:.0f}MB"
        if metrics.rss_mb > 0:
            mem_str += f"  rss={metrics.rss_mb:.0f}MB"

        type_names = metrics.extras.get("type_names", [])
        type_stats = metrics.extras.get("type_stats", {})
        type_str = ",".join(type_names) if type_names else "?"

        type_detail = ""
        if len(type_stats) > 1:
            parts = []
            for tn, ts in type_stats.items():
                rews = ts["rewards"]
                mr = sum(rews) / len(rews) if rews else 0
                parts.append(f"{tn}:r={mr:.2f},t={ts['trained_tokens']}")
            type_detail = f"  [{' '.join(parts)}]"

        lr_str = f"  lr={metrics.learning_rate:.2e}" if metrics.learning_rate > 0 else ""
        total_views = metrics.extras.get("total_views", 0)
        views_str = f"  views={total_views}" if total_views else ""

        angles = metrics.extras.get("weight_angles", {})
        angle_str = ""
        if angles:
            angle_str = (
                "  angles[" + " ".join(f"{k}={v:.2f}°" for k, v in sorted(angles.items())) + "]"
            )

        # Dynamic gradient scaling console output
        gradient_mults = metrics.extras.get("gradient_multipliers", [])
        gradient_str = ""
        if gradient_mults:
            from mlx_grpo.training.dynamic_scaling_logger import format_multipliers_for_console

            layer_names = [f"L{20 + i}" for i in range(16)]
            gradient_str = f"  {format_multipliers_for_console(gradient_mults, layer_names)}"

        _step_disp = f"{int(step) + 1}" if step == int(step) else f"{step:.1f}"
        print(
            f"Iter {_step_disp}/{total}: "
            f"[{type_str}] "
            f"loss={metrics.loss:.4f} (avg={avg_loss:.4f}) "
            f"reward={metrics.mean_reward:.3f} (avg={avg_reward:.3f}) "
            f"KL={metrics.kl_divergence:.4f} "
            f"tok/s={metrics.tokens_per_sec:.0f} "
            f"gen={metrics.generation_sec:.1f}s "
            f"t={metrics.elapsed_sec:.1f}s"
            f"{views_str}{lr_str}{mem_str}{angle_str}{gradient_str}{type_detail}"
        )

    # ── Helpers ──

    def _get_reward(
        self,
        task_type: TaskType,
        merged_config: dict[str, Any] | None = None,
    ) -> BaseReward:
        """Resolve the reward function for a task type.

        Passes reward_kwargs from type config to the reward constructor.
        Caches instances so LLM judge stats/cache persist across samples.
        """
        cfg = merged_config if merged_config is not None else task_type.config()
        reward_name = cfg.get("reward", "accuracy")
        kwargs = cfg.get("reward_kwargs", {})
        import json as _json

        cache_key = (reward_name, _json.dumps(kwargs, sort_keys=True) if kwargs else "")
        if not hasattr(self, "_reward_cache"):
            self._reward_cache: dict[tuple, BaseReward] = {}
        if cache_key in self._reward_cache:
            return self._reward_cache[cache_key]
        if kwargs:
            try:
                reward = BaseReward.get_with_kwargs(reward_name, **kwargs)
            except TypeError:
                reward = BaseReward.get(reward_name)
        else:
            reward = BaseReward.get(reward_name)
        self._reward_cache[cache_key] = reward
        return reward

    def _build_callbacks(self, task_type: TaskType) -> CallbackRegistry:
        """Build callback registry with task-specific + builtin callbacks."""
        cb = CallbackRegistry(seed=self._iteration)
        custom = task_type.callbacks()
        if custom:
            cb.register_many(custom)
        return cb

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def progress(self) -> float:
        return self._iteration / max(self.config.iterations, 1)


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: mlx_grpo core (no additional)
# Usage: Trainer(config, dataset, backend, model=model, ref_model=ref, optimizer=opt)
#        trainer.on("after_step", hook_fn)
#        trainer.train(resume_from=0)
# Status: Complete and commit-ready
# ─────────────────────────────────────────────────────────────────────
