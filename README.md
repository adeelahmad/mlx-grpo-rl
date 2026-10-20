# MLX-GRPO

**Template-driven GRPO training framework for Apple Silicon.**

A thin, metaprogramming-first framework for training language models with Group Relative Policy Optimization. Define task types in Python or YAML, compose rollout templates with fine-grained token masking, and train on Apple Silicon via MLX — or on GPUs via PyTorch/vLLM.

**[Documentation →](https://adeelahmad.github.io/mlx-grpo-framework/)** | **[WandB Dashboard →](https://wandb.ai/adeelahmad99/qwen4b-combined-multiview-ylora)** | **[SHAMIM Whitepaper →](whitepaper.md)**

## Installation

```bash
# Core (config + rollout engine only)
pip install -e .

# Apple Silicon training
pip install -e ".[mlx]"

# GPU training (PyTorch + PEFT)
pip install -e ".[pytorch]"

# Full (all backends + dev)
pip install -e ".[all]"
```

## CLI Reference

```bash
# ── Training ──────────────────────────────────────────
python -m mlx_grpo train --config <config.yml>
python -m mlx_grpo train --config <config.yml> --dry-run
python -m mlx_grpo train --config <config.yml> --resume <dir>
python -m mlx_grpo train --config <config.yml> --model <path>
python -m mlx_grpo train --config <config.yml> --iterations 100
python -m mlx_grpo train --config <config.yml> --wandb <project>

# ── Evaluation ────────────────────────────────────────
python -m mlx_grpo eval --config <config.yml>
python -m mlx_grpo eval --config <config.yml> --max-samples 50
python -m mlx_grpo eval --config <config.yml> --checkpoint <dir>

# ── Dataset Validation ────────────────────────────────
python -m mlx_grpo validate-data --config <config.yml>
python -m mlx_grpo validate-data --config <config.yml> --verbose
python -m mlx_grpo validate-data --config <config.yml> --output-dir <dir>

# ── Discovery ─────────────────────────────────────────
python -m mlx_grpo list-types
python -m mlx_grpo list-rewards
python -m mlx_grpo list-backends
```

## Quick Start — ReasonableQwen3-4B

A full working config is provided in `examples/`. To train with the same parameters used for the original ReasonableQwen3-4B:

```bash
# 1. Validate config
python -m mlx_grpo train --config examples/reasonableqwen3.yml --dry-run

# 2. Train (fresh start)
python -m mlx_grpo train --config examples/reasonableqwen3.yml

# 3. Train (resume from existing adapter)
python -m mlx_grpo train --config examples/reasonableqwen3.yml \
    --resume adapters/my_adapter1154

# 4. Or use the auto-restart wrapper
chmod +x train.sh
./train.sh
```

The config file maps every parameter from the original `train.sh`:

```yaml
training:
  model_path: "~/.cache/lm-studio/models/.../Qwen-4B-Thinking-2507.z"
  adapter_type: lora
  loss_type: dr_grpo
  epsilon: 0.02
  epsilon_high: 0.05
  beta: 0.04
  batch_size: 4
  gradient_accumulation_steps: 4
  group_size: 4
  learning_rate: 2e-6
  iterations: 5000
  lora_rank: 16
  lora_alpha: 32.0
  lora_dropout: 0.05
  wandb_project: my-experiment-named-fresh-only
  # ... see examples/reasonableqwen3.yml for full config with mapping comments

escalation:
  enabled: true
  horizontal:
    reward_threshold: 1.0
    max_members: 3
  vertical:
    n_chunks: 3
    stop_on_close: true
  scaffold:
    target: thinking
    max_tokens: 32

importance_sampling:
  mode: token
  decay_base: 0.7
  length_exponent: 0.5
  ref_length: 512
```

## Architecture

```
YAML Config
    |
TaskType (registered via __init_subclass__)
    |
RolloutExecutor -> RolloutContext -> [generate / inject / capture]
    |
RolloutResult (segments + loss_mask + variables)
    |
Reward Functions -> GRPO Advantages -> Loss -> Gradient Update
```

**Core principle:** Every abstraction must remove code elsewhere. No file exceeds 200 lines. Zero decorators for registration — just subclass and it works.

## Defining Task Types

### Python (full control)

```python
from mlx_grpo.types.base import TaskType

class MyType(TaskType, name="my_type"):
    """Custom task with chain-of-thought."""

    @staticmethod
    def rollout(ctx):
        ctx.inject("<think>")
        ctx.generate(max_tokens=256, stop=["</think>"], tag="thinking")
        ctx.inject("</think>\n<answer>")
        ctx.generate(max_tokens=64, stop=["</answer>"], tag="answer")
        ctx.inject("</answer>")
```

### YAML (no Python needed)

```yaml
types:
  my_type:
    template:
      - inject: "<think>"
      - generate: { max_tokens: 256, stop: ["</think>"], tag: thinking }
      - inject: "</think>\n<answer>"
      - generate: { max_tokens: 64, stop: ["</answer>"], tag: answer }
      - inject: "</answer>"
```

Both register identically — `TaskType.get("my_type")` works either way.

## Built-in Types

| Type | Aliases | Description |
|------|---------|-------------|
| `mcq` | `exam` | Multiple-choice questions with thinking scaffold |
| `math` | `exam_math`, `exam_olympiad` | Mathematical reasoning with step-by-step work |
| `general_qna` | `general` | General Q&A with F1 keyword-overlap reward |
| `python_code` | `code` | Python code generation with test validation |
| `python_verified` | `code_verified` | Python code with sandboxed execution-based self-verification |
| `tool_call` | | Function calling with structured output |
| `lotto` | `tattslotto`, `lottery`, `tatts` | TattsLotto statistical prediction with MCMC prefill |
| `multi_checkpoint` | | Multi-checkpoint reasoning with per-member difficulty variation |
| `variable_prefill` | | Variable prefill across group members with curriculum decay |
| `vqa` | | Visual question answering (requires VLM model + image_path field) |
| `image_captioning` | | Image description generation with F1 reward |

Aliases resolve via `TaskType.get()` — a dataset row with `"type": "exam"` routes to `MCQType` automatically.

## Reward Functions

| Reward | Description |
|--------|-------------|
| `accuracy` | Exact/fuzzy match against ground truth |
| `format` | Structural format compliance |
| `efficiency` | Token efficiency (penalizes verbosity) |
| `reasoning` | Chain-of-thought quality scoring |
| `hierarchical` | Multi-component weighted reward |
| `multiplicative` | Product of per-checkpoint scores |
| `exam` | Composite exam grading with format/gaming detection |
| `lotto` | TattsLotto multi-component (value, positional, near miss, format) |

Custom rewards:

```python
from mlx_grpo.rewards.base import BaseReward

class MyReward(BaseReward, name="my_reward"):
    def compute(self, result, expected):
        return 1.0 if expected in result.completion_text else 0.0
```

Or inline in YAML:

```yaml
rewards:
  my_reward: "lambda result, expected: 1.0 if expected in result.completion_text else 0.0"
```

## Loss Functions

Three GRPO variants, all with per-token masking:

- **GRPO** — Standard group-relative policy optimization with KL penalty
- **BNPO** — Balanced normalized variant (separates positive/negative advantages)
- **DR-GRPO** — Reference-free variant with entropy regularization
- **SFT** — Supervised fine-tuning loss for warm-start and anchor

```yaml
training:
  loss_type: dr_grpo   # grpo | bnpo | dr_grpo
  beta: 0.04           # KL penalty coefficient
  epsilon: 0.02        # PPO clip lower bound
  epsilon_high: 0.05   # PPO clip upper bound (asymmetric clipping)
```

## ICL-Guided Escalation Engine

The escalation engine is the core training algorithm for thinking models. It replaces the fixed scaffold curriculum with an adaptive, ICL-driven multi-member rollout system that creates gradient signal even when all rollouts fail.

**How it works:** For each training sample, the engine generates a *member ladder* — multiple rollouts with ascending scaffold levels (from naked to fully-guided). Each scaffolded member receives interleaved expert-trace chunks injected into its KV cache before generating. The naked member is always the last and serves as the held-out exam.

```
Member 0 (naked):      [PROBLEM] → [MODEL GENERATES freely]
Member 1 (1 chunk):    [PROBLEM] [chunk_0] [MODEL GENERATES] [INJECT "</think>"] [ANSWER]
Member 2 (2 chunks):   [PROBLEM] [chunk_0] [MODEL] [chunk_1] [MODEL] [INJECT "</think>"] [ANSWER]
```

All injected tokens are **masked from the loss** — the model trains only on what it generated.

**Adaptive member cap** — before generating, count sentence boundaries in the expert trace. Short traces (1 boundary) cap at 3 members instead of 13, saving 80%+ compute on hard problems.

**Bidirectional reward shaping** — when all members fail (the common case for hard problems):

```
r_k_shaped = λ_f × (C_k / C_max)    # reward scaffold structure itself → nonzero gradient
```

This converts 67.6% of otherwise-zero-gradient iterations into productive training steps.

**Tri-factor IS weights** — tokens close to scaffold injections (heavily influenced by scaffolding) get lower gradient weight; tokens far from scaffolding (genuinely independent reasoning) get higher weight:

```
w(k, i, j) = α_h^(C_k/C_max) × α_v^(i/N_eff) × α_p^j
```

```yaml
escalation:
  enabled: true
  horizontal:
    reward_threshold: 1.0   # stop adding members once reward reaches this
    max_members: 3
  vertical:
    n_chunks: 3             # max scaffold chunks per member
    stop_on_close: true     # stop Phase A when model closes </think>
    close_inject: true      # inject </think> if model doesn't close it
  scaffold:
    target: thinking        # inject into thinking block
    max_tokens: 32          # per-chunk token budget

# Tri-factor IS weights apply automatically when escalation.enabled: true
```

See [`docs/config/escalation.md`](docs/docs/config/escalation.md) for the complete field reference and [`whitepaper.md`](whitepaper.md) for the theoretical foundations (SHAMIM).

## Scaffold Curriculum (Legacy)

> **Replaced by the Escalation Engine above for thinking models.** `ScaffoldConfig` still works for non-thinking types and for simple prefix injection without the full escalation apparatus.

Scaffold injects a decaying prefix of the ground-truth answer into rollouts so the model learns to continue from progressively less assistance.

**Scaffold preprocessing** — before injection, all builtins pass the answer through `strip_boxed_tail()` which removes `</think>` and `\boxed{...}` boundaries. This prevents the scaffold from closing the `<think>` tag prematurely and forces the model to independently commit its own answer.

| Field | Default | Description |
|-------|---------|-------------|
| `initial_ratio` | 1.0 | Scaffold fraction at training start |
| `end_progress` | 0.8 | Training progress where scaffold reaches 0 |
| `decay_rate` | 1.0 | Decay speed (1.0 = linear) |
| `levels` | null | Per-group multipliers on base ratio |
| `max_scaffolded_tokens` | null | Token limit the ratio operates on |
| `mode` | prefix | Slice mode: prefix, suffix, middle, by_lines |
| `target` | both | Which phase: thinking, answer, both |
| `min_ratio` | 0.05 | Skip scaffold below this ratio |

**Scaffold probe** — evaluates model accuracy at fixed scaffold ratios during `eval_every` checkpoints:

```yaml
training:
  scaffold_probe_ratios: [0.0, 0.25, 0.5, 0.75, 1.0]
  scaffold_probe_samples: 20
```

## Vision-Language Models (VLM)

Train multimodal models on visual question answering and image captioning tasks:

```yaml
training:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"

vision:
  image_token: "<image>"
  freeze_vision_encoder: true   # train only LLM layers
  projector_lr_scale: 0.1       # slow down projector updates

datasets:
  train:
    path: data/vqa_train.jsonl
    type_name: vqa
```

Dataset rows include an `image_path` field pointing to a local file or URL. The `vqa` type rewards answer accuracy; `image_captioning` uses F1 overlap on descriptions.

Built-in vision rewards: `vqa_accuracy`, `caption_f1`, `visual_format`, `visual_reasoning`. The `ModalityRouter` automatically selects text-only vs vision path based on whether input tokens contain image markers.

## Training Modes

| Mode | Config | Trainable Params | Memory | Reference Model |
|------|--------|-----------------|--------|--------------------|
| **LoRA** | `adapter_type: lora` | ~0.1-1% | Low | Same model, adapters disabled |
| **DoRA** | `adapter_type: dora` | ~0.2-2% | Low-Med | Same model, adapters disabled |
| **Full** | `adapter_type: full` | 100% | High (2x) | Frozen weight snapshot |
| **LoRA-MoE** | `adapter_type: lora_moe` | ~0.3-3% | Low-Med | Same model, expert scale set to 0 |

```yaml
training:
  adapter_type: lora          # lora | dora | full | lora_moe
  lora_rank: 16               # adapter rank (lora/dora/lora_moe)
  lora_alpha: 32.0            # adapter scaling
  lora_dropout: 0.05          # adapter dropout
  lora_targets:               # which modules to adapt
    - self_attn.q_proj
    - self_attn.v_proj
  lora_layers: null           # null = all layers, or "0-8,20-28"
```

## LoRA Mixture-of-Experts

Train N independent LoRA expert adapters per layer, routed by a learned gating network. Each expert specialises on a task domain (MCQ, math, code, etc.) via supervised routing that gradually transitions to fully learned routing.

```yaml
training:
  adapter_type: lora_moe

moe:
  num_experts: 3              # Number of LoRA experts per layer
  lora_rank: 8                # Default rank shared across experts
  lora_alpha: 16.0
  lora_targets:
    - self_attn.q_proj
    - self_attn.v_proj
  lora_layers: null           # null = all layers

  gate:
    hidden_dim: 64            # MLP hidden size
    pool_method: mean         # mean | first | last | max (input embedding pooling)
    temperature: 1.0          # Initial softmax temperature
    temperature_min: 0.1      # Minimum temperature (annealed over training)
    temperature_decay_steps: 500
    noise_std: 0.0            # Exploration noise (annealed to 0)
    noise_decay_steps: 200

  routing:
    supervised_steps: 100     # Steps using forced routing from type labels
    transition_steps: 100     # Steps blending supervised → learned routing
    supervised_loss_weight: 1.0
    balance_loss_weight: 0.01 # Entropy bonus to prevent expert collapse

  expert_types:               # Optional: map type names to expert slots
    - type_name: mcq
      expert_index: 0
      lora_rank: 4            # Per-expert rank override
    - type_name: math
      expert_index: 1
      lora_alpha: 32.0        # Per-expert alpha override
    - type_name: general_qna
      expert_index: 2
```

**How it works:** The gating network pools the input embedding sequence → MLP → softmax → expert weights. During the supervised phase the target expert for the dominant sample type is blended in (`(1-w)*learned + w*one_hot`), then decays to pure learned routing. The reference model uses the same weights with `scale=0` on all expert layers (zero-copy). See `docs/config/moe.md` for the full field reference.

## Token-Level Importance Sampling

Penalise verbosity via gradient weighting — shorter completions and earlier tokens receive proportionally higher gradient weight, discouraging the model from padding with redundant reasoning.

> **With `escalation.enabled: true`:** IS weights for escalation rollouts are replaced by **tri-factor weights** (`α_h × α_v × α_p`) that encode scaffold depth, cycle position, and causal distance from injected tokens. The `importance_sampling:` block below applies to non-escalation rollouts and to naked members.

```yaml
importance_sampling:
  mode: token                 # none | token | segment | scaffold_proximity
  decay_base: 0.7             # positional decay: w(t) = decay_base^(t/n)
  length_exponent: 0.5        # length scale: (ref_length/n)^exponent
  ref_length: 512             # tokens at which length scale = 1.0
```

**`token` mode** — every completion token `t` in a completion of length `n` gets weight:

```
w(t) = (ref_length / n)^length_exponent  ×  decay_base^(t/n)
       └── length scale ───────────────┘  └── positional decay ┘
```

A completion half as long as `ref_length` gets 1.41× the gradient weight. The last token gets ~50% the weight of the first (with `decay_base=0.5`).

**`segment` mode** — uses rollout segment tags instead of position:

```yaml
importance_sampling:
  mode: segment
  thinking_weight: 0.5        # thinking-tagged tokens get 50% weight
  answer_weight: 2.0          # answer-tagged tokens get 200% weight
```

**`scaffold_proximity` mode** — tokens close to injected scaffold boundaries get highest weight (`w = proximity_decay^distance`).

See `docs/config/importance_sampling.md` for the full field reference.

## Backends

| Backend | Hardware | Mode | Install |
|---------|----------|------|---------|
| `mlx` | Apple Silicon | Training ✓ | `pip install mlx mlx-lm` |
| `pytorch` | CUDA/MPS/CPU | Training ✓ | `pip install torch transformers peft` |
| `vllm` | Multi-GPU | Rollouts only | `pip install vllm` |
| `llamacpp` | CPU/GGUF | Rollouts only | `pip install llama-cpp-python` |

MLX and PyTorch backends support full training (forward/backward/update) with LoRA/DoRA/full adapter modes, checkpointing, and optimizer state save/restore. vLLM and llamacpp are inference-only backends, ideal for high-throughput rollout generation and evaluation.

## Per-Type Training Metrics

When training on mixed datasets, per-type breakdowns are tracked automatically:

**Console output** shows the active type(s) and per-type reward breakdown:
```
Iter 1/500: [mcq] loss=0.0029 reward=0.333 KL=0.0014 tok/s=10 t=27.9s
Iter 2/500: [mcq,math] loss=0.0031 reward=0.500 KL=0.0018 tok/s=12 t=25.1s  [mcq=0.33 math=0.67]
```

**CSV stats** (`checkpoints/stats.csv`) include per-step columns: `type`, `mean_advantage`, `group_size`, `total_generations`, `num_correct`, `num_incorrect`.

**WandB** logs per-type metrics under `type/{name}/reward`, `type/{name}/advantage`, `type/{name}/correct`, `type/{name}/incorrect`, `type/{name}/gen_tokens`, `type/{name}/trained_tokens`, `type/{name}/count`.

## Adaptive Sampling

Online curriculum sampling that adjusts sample frequency based on reward patterns — reduces exposure to mastered or impossible samples:

```yaml
training:
  adaptive_sampling: true
  adaptive_zero_threshold: 5    # Skip sample after N consecutive zero-reward runs
  adaptive_perfect_threshold: 8 # Skip sample after N consecutive perfect runs
  adaptive_retry_interval: 50   # Re-try skipped samples every N iterations
```

## Epochs and Sample Repetition

Train by epochs instead of raw iteration count:

```yaml
training:
  epochs: 3              # 3 passes through the dataset
  batch_size: 4           # iterations = epochs * ceil(dataset_size / batch_size)
```

Repeat each sampled batch multiple times with fresh rollouts for more gradient signal per prompt:

```yaml
training:
  samples_per_step: 2     # Each batch gets 2 gradient updates (fresh rollouts each)
```

## Loss Token Truncation

Generate long completions for quality but only use a subset of tokens for loss calculation, reducing memory in the gradient pass:

```yaml
generation:
  max_tokens: 2048        # Model generates up to 2048 tokens
  used_tokens: 256        # Only 256 completion tokens enter loss calculation
  truncate_from: prefix   # Keep first 256 tokens, discard the rest
```

Truncation modes:

| Mode | Behavior |
|------|----------|
| `prefix` | Keep first N completion tokens, append `...` marker |
| `postfix` | Prepend `...` marker, keep last N completion tokens |
| `middle` | `...` + middle N tokens + `...` |

The `...` marker tokens are automatically tokenized and masked (excluded from loss). When `used_tokens` is null or >= actual completion length, no truncation occurs.

## Configuration Reference

All training parameters in a single YAML file:

```yaml
training:
  # ── Model ──
  model_path: ""                        # HF model ID or local path
  adapter_path: null                    # Adapter directory (load + save checkpoints)
  adapter_type: lora                    # lora | dora | full | lora_moe

  # ── Optimization ──
  learning_rate: 1e-5
  iterations: 500                       # Training steps (ignored if epochs > 0)
  epochs: 0                             # 0 = use iterations; >0 = auto-compute iterations
  batch_size: 1
  samples_per_step: 1                   # Repeat each batch N times (fresh rollouts each)
  gradient_accumulation_steps: 1        # Effective batch = batch_size * samples_per_step * grad_accum
  seed: 0                               # Random seed (0 = random)

  # ── GRPO ──
  group_size: 4
  beta: 0.04                            # KL penalty coefficient
  epsilon: 0.1                          # PPO clip lower bound
  epsilon_high: null                    # PPO clip upper bound (null = symmetric)
  loss_type: grpo                       # grpo | bnpo | dr_grpo

  # ── Checkpoints ──
  save_every: 100
  eval_every: 50
  steps_per_report: 10
  keep_last_n_checkpoints: 5
  keep_best_n_checkpoints: 3
  save_optimizer: true                  # Save optimizer state (required for exact resume)
  wandb_log_full_model: false           # Log full model weights to WandB artifact

  # ── Adapter (LoRA/DoRA) ──
  lora_rank: 8
  lora_alpha: 16.0
  lora_dropout: 0.0
  lora_targets: [self_attn.q_proj, self_attn.v_proj]
  lora_layers: null                     # null = all, or "0-8,20-28"

  # ── Dataset ──
  shuffle_data: true
  shuffle_seed: null                    # null = random each run
  balanced_shuffle: true                # Balance by type before shuffling
  batches_per_type: false               # When true + batch_size>1: each batch is one type (round-robin)
  max_seq_length: 2048

  # ── Adaptive Sampling ──
  adaptive_sampling: false
  adaptive_zero_threshold: 5
  adaptive_perfect_threshold: 8
  adaptive_retry_interval: 50

  # ── GPU/Metal Performance ──
  grad_checkpoint: true                 # Trade compute for memory
  max_grad_seq_len: 1024                # Truncate sequences for Metal timeout safety (macOS kills >8s)
  verbose_grad: false                   # Print per-sequence truncation details
  gpu_cooldown: 1.0                     # Seconds between gradient computations
  rollout_cooldown: 0.5                 # Seconds between rollout generations
  phase_cooldown: 2.0                   # Seconds between generation and gradient phases

  # ── Scaffold Probe ──
  scaffold_probe_ratios: [0.0, 0.25, 0.5, 0.75, 1.0]
  scaffold_probe_samples: 20

  # ── Recovery ──
  auto_resume_on_crash: true
  max_crash_retries: 3
  crash_cooldown_seconds: 10

  # ── Logging ──
  wandb_project: null                   # null = disabled

generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 0.95
  top_k: 20
  min_p: 0.0                           # Min probability threshold (0.0 = disabled)
  backend: mlx                          # mlx | pytorch for training; vllm | llamacpp for rollouts only
  repetition_penalty: 1.0
  repetition_context_size: 20
  xtc_probability: 0.0                  # XTC sampling probability (0.0 = disabled)
  xtc_threshold: 0.1                    # XTC culling threshold
  min_tokens_to_keep: 1                 # Min tokens kept after filtering
  system_prompt: ""                     # Default system prompt ("" = model default)
  # KV cache quantization (MLX only)
  kv_bits: null                         # null = full precision, or 4/8
  kv_group_size: 64
  max_kv_size: null                     # null = unlimited
  # Speculative decoding
  draft_model_path: null                # Path to smaller draft model
  # Loss token truncation — generate max_tokens but only train on used_tokens
  used_tokens: null                     # null = use all; N = keep N completion tokens for loss
  truncate_from: prefix                 # prefix | postfix | middle
  truncate_marker: "...[TRUNCATED]..."  # Text inserted at truncation boundary (loss-masked)
  truncate_marker_masked: true          # Exclude truncation marker from gradient
  snap_to_boundary: false               # Snap truncation to nearest token boundary

gradient:
  train_layers: all                     # all | "0-8,20-28"
  thinking_gradient_weight: 1.0
  answer_gradient_weight: 1.0
  max_grad_norm: 1.0                    # Gradient clipping threshold (0 = disabled)
  sft_anchor_enabled: false
  sft_anchor_lr_multiplier: 0.1        # LR multiplier for SFT anchor steps
  gradient_alignment_mode: none         # none | project | interpolate
  gradient_alignment_weight: 0.5

monitor:
  enabled: true
  kl_warning: 0.04
  kl_critical: 0.08
  reward_warning: 0.3
  stop_on_critical: false

# ── Optional: ICL-guided escalation engine ──
# escalation:
#   enabled: true
#   horizontal:
#     reward_threshold: 1.0   # stop ladder once reward meets this
#     max_members: 3
#     naked_always: true      # always include a naked (no-scaffold) member
#   vertical:
#     n_chunks: 3             # max scaffold chunks per member
#     stop_on_close: true     # stop Phase A on </think>
#     close_inject: true      # inject </think> if model doesn't close
#   scaffold:
#     target: thinking        # inject into thinking block
#     max_tokens: 32          # max tokens per chunk
#   tri_factor:
#     alpha_h: 0.70           # horizontal IS decay (scaffold depth)
#     alpha_v: 0.80           # vertical IS decay (cycle position)
#     alpha_p: 0.90           # proximity IS decay (causal distance)
#   reward_shaping:
#     lambda_s_max: 0.05      # success penalty (anneals up)
#     lambda_f_max: 0.02      # failure nudge (anneals down)

# ── Optional: token-level importance sampling (verbosity penalty) ──
# importance_sampling:
#   mode: token               # none | token | segment | scaffold_proximity
#   decay_base: 0.7           # positional decay base
#   length_exponent: 0.5      # length scale exponent
#   ref_length: 512           # reference completion length in tokens

# ── Optional: spike guard (training instability protection) ──
# spike_guard:
#   kl_threshold: 0.35        # KL divergence spike threshold
#   loss_threshold: 8.0       # loss spike threshold
#   kl_delta: 0.25            # max single-step KL increase allowed
#   loss_delta: 4.0           # max single-step loss increase allowed
#   window: 20                # rolling window size for baseline stats
#   min_window: 5             # steps before spike detection activates
#   action: skip              # skip | rollback | abort
#   max_consecutive: 2        # max consecutive spikes before abort
#   log_file: null            # optional JSONL file for spike events

# ── Optional: dynamic views (named rollout variants per type) ──
# dynamic_views:
#   <type_name>:
#     views: [...]            # per-member override configs
#   See docs/docs/config/type_overrides.md for view_strategy details

# ── Optional: LoRA-MoE (requires adapter_type: lora_moe) ──
# moe:
#   num_experts: 3
#   lora_rank: 8
#   lora_targets: [self_attn.q_proj, self_attn.v_proj]
#   gate: { hidden_dim: 64, pool_method: mean }
#   routing: { supervised_steps: 100, balance_loss_weight: 0.01 }

datasets:
  train:
    path: data/train.jsonl
    type_name: mcq
    prompt_field: prompt
    answer_field: answer
```

## Dataset Validation

Validate dataset rows against their declared types before training:

```bash
# Check all datasets in config
python -m mlx_grpo validate-data --config config.yml

# Show per-row issue details
python -m mlx_grpo validate-data --config config.yml --verbose

# Export valid rows as per-type JSONL files
python -m mlx_grpo validate-data --config config.yml --output-dir clean_data/
```

## Sandboxed Code Execution

The `python_verified` type runs model-generated code in an isolated sandbox with security constraints:

- **Isolated venv** — no pip, no inherited environment variables
- **Command validation** — only `python*` commands allowed; blocks `rm`, `curl`, `ssh`, `sudo`, etc.
- **Resource limits** — 30s CPU, 256MB memory, 10MB file size, 64 file descriptors (configurable)
- **Metal-safe forking** — calls `mx.synchronize()` + `mx.clear_cache()` before subprocess fork to prevent GPU handle duplication
- **Process isolation** — `start_new_session=True` for full process group separation

```python
from mlx_grpo.sandbox import Sandbox

sb = Sandbox(timeout=30, max_memory_mb=256)
result = sb.run("python solution.py", code="print('hello')")
# result.returncode, result.stdout, result.stderr
```

## Training Hooks

```python
trainer = Trainer(config=config, dataset=dataset, backend=backend)

trainer.on("after_step", monitor.step_hook)
trainer.on("on_checkpoint", lambda t, m: checkpoint_mgr.save(...))
trainer.on("on_eval", lambda t, m: evaluator.evaluate(...))

trainer.train()
```

## Rollout Callbacks

`CallbackRegistry` provides utility functions accessible as `ctx.callbacks.*` during rollouts. Seeded per-iteration for reproducible randomness.

| Category | Functions |
|----------|-----------|
| **Validators** | `is_valid_json`, `is_valid_python`, `is_valid_xml` |
| **Extractors** | `extract_boxed`, `extract_between`, `extract_json` |
| **Text ops** | `strip_boxed_tail`, `slice_answer`, `slice_answer_tokens`, `truncate_thinking`, `has_tag`, `endswith_any`, `count_tokens` |
| **Curriculum** | `compute_curriculum_ratio`, `curriculum_filter` |
| **Randomness** | `random`, `coin_flip`, `reseed` |

Task types can register custom callbacks via their `callbacks()` method:

```python
class MCQType(TaskType, name="mcq"):
    def callbacks(self):
        return {"is_valid_mcq": lambda text: bool(re.search(r"\([A-D]\)", text))}
```

## Gradient Operations

```python
from mlx_grpo._internals import (
    compute_gradient_alignment,
    create_layer_gradient_mask,
    apply_gradient_mask,
    clip_gradient_norm,
)

aligned, info = compute_gradient_alignment(grpo_grads, sft_grads, mode="project")
mask = create_layer_gradient_mask(layer_names, freeze_patterns=["embed", "lm_head"])
grads = apply_gradient_mask(grads, mask)
```

## Project Structure

```
mlx-grpo/
├── README.md
├── whitepaper.md              # SHAMIM technical whitepaper
├── CLAUDE.md                  # AI assistant reference
├── pyproject.toml
├── train.sh                   # Auto-restart training wrapper (--no-loop, --dry-run)
├── configs/                   # Training configs
├── examples/
│   └── reasonableqwen3.yml    # Full config — maps every train.sh flag
├── docs/                      # Docusaurus site (GitHub Pages)
├── src/mlx_grpo/
│   ├── __init__.py             # Public API
│   ├── __main__.py             # python -m mlx_grpo
│   ├── cli.py                  # CLI commands (train, eval, validate-data, list-*)
│   ├── sandbox.py              # Isolated venv execution for python_verified
│   ├── scaffold.py             # Scaffold curriculum config (legacy)
│   ├── server.py               # REST API server (job submission + polling)
│   ├── server_auth.py          # API key auth middleware
│   ├── server_jobs.py          # Job queue and state management
│   ├── _internals/             # Gradient alignment, layer masking, SFT anchor
│   ├── backends/               # MLX, PyTorch, vLLM, llama.cpp
│   ├── config/
│   │   ├── loader.py           # YAML → frozen dataclasses
│   │   ├── escalation.py       # EscalationConfig + TriFactorConfig + RewardShapingConfig
│   │   └── vision.py           # VisionConfig (VLM support)
│   ├── data/                   # Dataset loader, balanced shuffle, adaptive sampler
│   ├── engine/                 # RolloutExecutor + RolloutContext + CallbackRegistry
│   ├── loss/                   # GRPO/BNPO/DR-GRPO/SFT (NumPy ref + MLX prod)
│   ├── model/
│   │   ├── lora_moe.py         # LoRA-MoE layers (LoRAExpert, GatingNetwork)
│   │   └── modality_router.py  # TextVisualRouter for VLM inputs
│   ├── rewards/
│   │   ├── exam.py, lotto.py, math.py, accuracy.py, ...
│   │   └── vision.py           # VQA + image captioning rewards
│   ├── training/
│   │   ├── __init__.py         # Trainer + GRPO loop
│   │   ├── escalation.py       # ICL-guided escalation engine (two-phase cycles)
│   │   ├── escalation_plan.py  # Scaffold planning helpers (boundary detection)
│   │   ├── token_importance.py # Token IS weights + tri-factor weights
│   │   ├── full_checkpoint.py  # Full model checkpoint save/restore
│   │   ├── vlm_loader.py       # VLM processor + vision encoder freezing
│   │   └── checkpoint.py, monitor.py, recovery.py, ...
│   └── types/
│       ├── builtins/
│       │   ├── mcq.py, math.py, general_qna.py, python_code.py, ...
│       │   ├── vqa.py           # Visual question answering
│       │   └── image_captioning.py
│       └── yaml_type.py        # YAML-defined custom types
└── tests/                      # 1,500+ tests (≥90% coverage enforced)
```

## Running Tests

```bash
python -m pytest tests/ -q                                              # All tests
python -m pytest tests/ -q --tb=short                                   # With failure details
python -m pytest tests/ --cov=mlx_grpo --cov-config=.coveragerc -q     # With coverage (≥90% required)
```

## Demonstrated Results (SHAMIM, 450 iterations)

Training Qwen-4B-Thinking-2507 on 16,872 problems (math + MCQ) with the escalation engine on an Apple M4 Max:

| Metric | Start | Iter 225 | Improvement |
|--------|-------|----------|-------------|
| Math accuracy (eval) | 8% | 31% | +23 pp |
| Exam accuracy (eval) | ~60% | ~75% | +15 pp |
| Tokens per solution | ~14,000 | ~6,695 | 2.1× compression |
| KL divergence | 0.00056 | 0.00122 | Stable |

**67.6% of training iterations** were all-wrong groups — without bidirectional reward shaping, these produce exactly zero gradient. The escalation engine sustained learning through all of them.

Peak memory: **9.2 GB**. Generation speed: **22.9 tok/s**. Full dashboard: [wandb.ai/adeelahmad99/qwen4b-combined-multiview-ylora](https://wandb.ai/adeelahmad99/qwen4b-combined-multiview-ylora)

See [whitepaper.md](whitepaper.md) for full methodology and analysis.

## Design Philosophy

- **Thin:** No file exceeds 200 lines. Every abstraction removes code elsewhere.
- **Metaprogramming-first:** `__init_subclass__` auto-registration. Zero decorators.
- **Template-driven:** Rollouts are composable sequences of generate/inject/capture.
- **Loss mask per-token:** Injected scaffolding masked, generated tokens unmasked.
- **Backend-agnostic:** Same rollout code runs on MLX, PyTorch, vLLM, or llama.cpp.
- **Signal-first:** The escalation engine ensures every training iteration produces a gradient signal, even on hard problems where all rollouts fail.

## License

MIT
# mlx-grpo-rl

<img src="https://www.google-analytics.com/g/collect?v=2&amp;tid=G-RKFB16BPXJ&amp;cid=555.mlx-grpo-rl&amp;en=page_view&amp;dl=https%3A%2F%2Fgithub.com%2Fadeelahmad%2Fmlx-grpo-rl&amp;dt=mlx-grpo-rl&amp;ep.repo=mlx-grpo-rl" width="1" height="1" alt="">
