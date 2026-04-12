"""Six new GRPO-specific analysis features.

All functions accept a collector-like object (or mock) so they're fully
testable without real MLX hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .metrics import compute_layerwise_similarity, linear_cka

# ── Collector Protocol (enables mocking in tests) ────────────────────────────


class CollectorProtocol(Protocol):
    def collect(self, prompt: str) -> Any: ...


# ── Feature 1: Checkpoint CKA Trajectory ─────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CheckpointCKAResult:
    checkpoint_labels: list[str]
    layer_keys: list[str]
    cka_matrix: npt.NDArray[np.float32]  # (n_checkpoints, n_layers)


def run_checkpoint_trajectory(
    cfg: Any,
    base_collector: CollectorProtocol,
) -> CheckpointCKAResult:
    """Feature 1: CKA drift from base across sorted checkpoint directories.

    Each checkpoint must contain adapters.safetensors or be loadable by
    mlx_lm. Returns CKA matrix of shape (n_checkpoints, n_layers).
    """
    if cfg.checkpoints_dir is None:
        raise ValueError("checkpoints_dir is required for feature=checkpoint")

    # Collect base activations
    base_result = base_collector.collect(cfg.prompt)
    base_sketches = base_result.sketches
    layer_keys = sorted(
        base_sketches.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0
    )

    ckpt_dir = Path(cfg.checkpoints_dir)
    checkpoints = sorted(ckpt_dir.glob("*/"), key=lambda p: p.name)
    if not checkpoints:
        checkpoints = sorted(ckpt_dir.glob("*.safetensors"))

    checkpoint_labels: list[str] = []
    rows: list[list[float]] = []

    for ckpt_path in checkpoints:
        label = ckpt_path.name
        try:
            from .collector import ActivationCollector, load_model_for_inspect  # noqa: PLC0415

            model, tokenizer = load_model_for_inspect(str(ckpt_path))
            ckpt_collector = ActivationCollector(model, tokenizer, cfg)
            ckpt_result = ckpt_collector.collect(cfg.prompt)
        except Exception:  # noqa: BLE001
            continue

        similarity = compute_layerwise_similarity(base_sketches, ckpt_result.sketches)
        row = [similarity.get(k, {}).get("cka", 0.0) for k in layer_keys]
        checkpoint_labels.append(label)
        rows.append(row)

    if not rows:
        cka_matrix = np.zeros((0, len(layer_keys)), dtype=np.float32)
    else:
        cka_matrix = np.array(rows, dtype=np.float32)

    return CheckpointCKAResult(
        checkpoint_labels=checkpoint_labels,
        layer_keys=layer_keys,
        cka_matrix=cka_matrix,
    )


# ── Feature 2: Reward-Stratified Activations ─────────────────────────────────


@dataclass(slots=True, frozen=True)
class RewardStratifiedResult:
    high_similarity: dict[str, dict[str, float]]
    low_similarity: dict[str, dict[str, float]]
    median_reward: float
    n_high: int
    n_low: int


def run_reward_stratified(cfg: Any, collector: CollectorProtocol) -> RewardStratifiedResult:
    """Feature 2: Compare activations for high vs low reward rollouts.

    Loads rollout_log JSONL (fields: prompt, completion, reward).
    Splits by median reward, collects activations per bin.
    """
    if cfg.rollout_log is None:
        raise ValueError("rollout_log is required for feature=reward_strat")

    records: list[dict[str, Any]] = []
    with open(cfg.rollout_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"rollout_log {cfg.rollout_log!r} is empty")

    rewards = np.array([float(r.get("reward", 0.0)) for r in records])
    median_reward = float(np.median(rewards))

    high_records = [r for r, rew in zip(records, rewards, strict=False) if rew >= median_reward]
    low_records = [r for r, rew in zip(records, rewards, strict=False) if rew < median_reward]

    def _collect_mean_sketches(recs: list[dict[str, Any]]) -> dict[str, npt.NDArray[Any]]:
        acc: dict[str, list[npt.NDArray[Any]]] = {}
        for rec in recs:
            prompt = rec.get("prompt", "") + rec.get("completion", "")
            try:
                result = collector.collect(prompt)
                for key, sketch in result.sketches.items():
                    acc.setdefault(key, []).append(sketch)
            except Exception:  # noqa: BLE001
                continue
        return {k: np.mean(v, axis=0) for k, v in acc.items() if v}

    high_sketches = _collect_mean_sketches(high_records)
    low_sketches = _collect_mean_sketches(low_records)

    # Get base sketches for comparison
    base_result = collector.collect(cfg.prompt)
    high_sim = compute_layerwise_similarity(base_result.sketches, high_sketches)
    low_sim = compute_layerwise_similarity(base_result.sketches, low_sketches)

    return RewardStratifiedResult(
        high_similarity=high_sim,
        low_similarity=low_sim,
        median_reward=median_reward,
        n_high=len(high_records),
        n_low=len(low_records),
    )


# ── Feature 3: LoRA Isolation ─────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class LoRAIsolationResult:
    base_similarity: dict[str, dict[str, float]]
    lora_similarity: dict[str, dict[str, float]]
    delta_cka: dict[str, float]  # layer_key → (lora_cka - base_cka)


def run_lora_isolation(cfg: Any, model: Any, tokenizer: Any) -> LoRAIsolationResult:
    """Feature 3: Compare activations with LoRA scale=0.0 vs scale=1.0.

    Returns similarity dicts for both states vs a reference.
    """
    from .collector import ActivationCollector  # noqa: PLC0415

    collector = ActivationCollector(model, tokenizer, cfg)

    # LoRA off (base)
    set_lora = getattr(model, "set_lora_scale", None)
    if set_lora is not None:
        set_lora(0.0)
    base_result = collector.collect(cfg.prompt)

    # LoRA on (fine-tuned)
    if set_lora is not None:
        set_lora(1.0)
    lora_result = collector.collect(cfg.prompt)

    # Compare both against base
    base_sim = compute_layerwise_similarity(base_result.sketches, base_result.sketches)
    lora_sim = compute_layerwise_similarity(base_result.sketches, lora_result.sketches)

    delta_cka: dict[str, float] = {}
    for key in lora_sim:
        b = base_sim.get(key, {}).get("cka", 1.0)
        lv = lora_sim.get(key, {}).get("cka", 1.0)
        delta_cka[key] = lv - b

    return LoRAIsolationResult(
        base_similarity=base_sim,
        lora_similarity=lora_sim,
        delta_cka=delta_cka,
    )


# ── Feature 4: Type Fingerprints ─────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class TypeFingerprintResult:
    labels: list[str]
    layer_keys: list[str]
    cross_cka: npt.NDArray[np.float32]  # (n_labels, n_labels, n_layers) pairwise CKA
    mean_sketches: dict[str, dict[str, npt.NDArray[Any]]]  # label → layer_key → sketch


def run_type_fingerprints(cfg: Any, collector: CollectorProtocol) -> TypeFingerprintResult:
    """Feature 4: Compute per-label mean activations, then CKA between all pairs.

    Loads cfg.prompts_file as JSON [{prompt: str, label: str}].
    """
    if cfg.prompts_file is None:
        raise ValueError("prompts_file is required for feature=fingerprint")

    with open(cfg.prompts_file) as f:
        records: list[dict[str, Any]] = json.load(f)

    grouped: dict[str, list[Any]] = {}
    for rec in records:
        label = str(rec.get("label", "unknown"))
        prompt = str(rec.get("prompt", ""))
        try:
            result = collector.collect(prompt)
            grouped.setdefault(label, []).append(result.sketches)
        except Exception:  # noqa: BLE001
            continue

    labels = sorted(grouped.keys())
    if not labels:
        raise ValueError("No valid records collected from prompts_file")

    # Compute mean sketch per label per layer
    mean_sketches: dict[str, dict[str, npt.NDArray[Any]]] = {}
    for label, sketch_list in grouped.items():
        layer_acc: dict[str, list[npt.NDArray[Any]]] = {}
        for sketch_dict in sketch_list:
            for key, vec in sketch_dict.items():
                layer_acc.setdefault(key, []).append(vec)
        mean_sketches[label] = {k: np.mean(v, axis=0) for k, v in layer_acc.items()}

    all_keys = sorted(
        mean_sketches[labels[0]].keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else 0
    )
    layer_keys = all_keys
    n = len(labels)
    n_layers = len(layer_keys)

    # Cross-label CKA per layer
    cross_cka = np.zeros((n, n, n_layers), dtype=np.float32)
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            for li, lk in enumerate(layer_keys):
                sa = mean_sketches[la].get(lk)
                sb = mean_sketches[lb].get(lk)
                if sa is not None and sb is not None:
                    cross_cka[i, j, li] = float(linear_cka(sa.reshape(1, -1), sb.reshape(1, -1)))

    return TypeFingerprintResult(
        labels=labels,
        layer_keys=layer_keys,
        cross_cka=cross_cka,
        mean_sketches=mean_sketches,
    )


# ── Feature 5: Scaffold vs Naked Comparison ──────────────────────────────────


@dataclass(slots=True, frozen=True)
class ScaffoldNakedResult:
    similarity: dict[str, dict[str, float]]
    mean_cka: float
    mean_cosine: float


def run_scaffold_naked_comparison(
    cfg: Any, collector: CollectorProtocol, scaffold_prefix: str
) -> ScaffoldNakedResult:
    """Feature 5: Compare activations for raw prompt vs scaffold+prompt.

    scaffold_prefix: the injected scaffold text to prepend.
    """
    raw_result = collector.collect(cfg.prompt)
    scaffold_result = collector.collect(scaffold_prefix + cfg.prompt)

    similarity = compute_layerwise_similarity(raw_result.sketches, scaffold_result.sketches)

    cka_vals = [v.get("cka", 0.0) for v in similarity.values()]
    cos_vals = [v.get("cosine", 0.0) for v in similarity.values()]

    mean_cka = float(np.mean(cka_vals)) if cka_vals else 0.0
    mean_cosine = float(np.mean(cos_vals)) if cos_vals else 0.0

    return ScaffoldNakedResult(
        similarity=similarity,
        mean_cka=mean_cka,
        mean_cosine=mean_cosine,
    )
