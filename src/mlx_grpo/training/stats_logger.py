"""CSV stats logger — per-iteration metrics saved to disk.

Writes one row per training step with all scalar metrics including
memory usage. Designed as an after_step hook for the Trainer.

    logger = StatsLogger("checkpoints/stats.csv")
    trainer.on("after_step", logger.step_hook)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_grpo.training import StepMetrics, Trainer

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler
from mlx_grpo.training.dynamic_scaling_logger import multipliers_to_csv_row

COLUMNS = [
    "iteration",
    "epoch",
    "seed",
    "type",
    "loss",
    "learning_rate",
    "mean_reward",
    "reward_std",
    "kl_divergence",
    "mean_advantage",
    "grad_norm",
    "tokens_per_sec",
    "generated_tokens",
    "trained_tokens",
    "group_size",
    "total_views",
    "samples_per_step",
    "total_generations",
    "num_correct",
    "num_incorrect",
    "generation_sec",
    "elapsed_sec",
    "peak_memory_mb",
    "rss_mb",
    "cka_drift",
    "is_weight_mean",
    "is_weight_min",
    "is_weight_max",
    "dataset_row",
    "repeat_idx",
    "repeat_total",
    "repeat_done_reason",
    "naked_phase",
    "weight_angle_avg",
    "gradient_multiplier_L20",
    "gradient_multiplier_L21",
    "gradient_multiplier_L22",
    "gradient_multiplier_L23",
    "gradient_multiplier_L24",
    "gradient_multiplier_L25",
    "gradient_multiplier_L26",
    "gradient_multiplier_L27",
    "gradient_multiplier_L28",
    "gradient_multiplier_L29",
    "gradient_multiplier_L30",
    "gradient_multiplier_L31",
    "gradient_multiplier_L32",
    "gradient_multiplier_L33",
    "gradient_multiplier_L34",
    "gradient_multiplier_L35",
    "gradient_multiplier_max",
    "gradient_multiplier_mean",
    "gar_loss",
    "gar_angle_avg",
    "gar_scaffold_fraction",
    "judge_calls",
    "judge_cache_hits",
    "judge_failures",
    "judge_correct",
    "judge_mean_confidence",
    "judge_mean_latency_ms",
]


class StatsLogger:
    """Append per-step scalar metrics to a CSV file.

    If the file already exists (e.g. resumed training), new rows are appended
    without rewriting the header.
    """

    def __init__(
        self,
        path: str | Path,
        seed: int = 0,
        epochs: int = 0,
        samples_per_step: int = 1,
        dataset_size: int = 0,
        batch_size: int = 1,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seed = seed
        self._epochs = epochs
        self._samples_per_step = samples_per_step
        self._dataset_size = dataset_size
        self._batch_size = max(batch_size, 1)
        self._file = None
        self._writer: csv.DictWriter | None = None

        exists = self._path.exists() and self._path.stat().st_size > 0
        self._file = open(self._path, "a", newline="")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS, extrasaction="ignore")
        if not exists:
            self._writer.writeheader()
            self._file.flush()

    def step_hook(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Trainer after_step hook — write one CSV row."""
        if self._writer is None:
            return

        type_names = metrics.extras.get("type_names", [])
        type_stats = metrics.extras.get("type_stats", {})
        advantages = metrics.extras.get("advantages", [])
        mean_adv = sum(advantages) / len(advantages) if advantages else 0.0
        total_gens = sum(ts["count"] for ts in type_stats.values())
        num_correct = sum(ts["correct"] for ts in type_stats.values())
        num_incorrect = sum(ts["incorrect"] for ts in type_stats.values())

        # Compute epoch from iteration + dataset geometry
        epoch = 0.0
        if self._epochs > 0 and self._dataset_size > 0:
            steps_per_epoch = self._dataset_size / self._batch_size
            epoch = (metrics.iteration + 1) / max(steps_per_epoch, 1)

        # Compute gradient multipliers from angle drifts
        gradient_mult_row = {}
        weight_angles = metrics.extras.get("weight_angles", {})
        if weight_angles:
            # Extract angle drifts per layer (convert angle dict to list for L20-L35)
            layer_names = [f"L{20 + i}" for i in range(16)]
            angle_drifts = [weight_angles.get(ln, 0.0) for ln in layer_names]
            # Compute multipliers using DynamicGradientScaler
            scaler = DynamicGradientScaler()
            multipliers = scaler.compute_multipliers(angle_drifts)
            # Convert to CSV row
            gradient_mult_row = multipliers_to_csv_row(
                multipliers,
                layer_names,
                max(multipliers) if multipliers else 1.0,
                sum(angle_drifts) / len(angle_drifts) if angle_drifts else 0.0,
            )

        # GAR mode columns (present when gar_loss in extras)
        gar_row: dict[str, str] = {"gar_loss": "", "gar_angle_avg": "", "gar_scaffold_fraction": ""}
        gar_angles = metrics.extras.get("gar_angles", {})
        if "gar_loss" in metrics.extras:
            gar_row["gar_loss"] = f"{metrics.extras['gar_loss']:.6f}"
            gar_row["gar_scaffold_fraction"] = (
                f"{metrics.extras.get('gar_scaffold_fraction', 0.0):.4f}"
            )
            if gar_angles:
                avg = sum(v["angle_mean"] for v in gar_angles.values()) / len(gar_angles)
                gar_row["gar_angle_avg"] = f"{avg:.4f}"

        # Per-layer GAR columns are dynamic — use extrasaction="ignore" already set
        per_layer_row: dict[str, str] = {}
        for layer_key in sorted(gar_angles.keys()):
            per_layer_row[f"gar_angle_per_layer_L{layer_key}"] = (
                f"{gar_angles[layer_key].get('angle_mean', 0.0):.4f}"
            )
            per_layer_row[f"gar_tokens_above_thresh_L{layer_key}"] = str(
                gar_angles[layer_key].get("tokens_above", 0)
            )

        self._writer.writerow(
            {
                "iteration": metrics.iteration,
                "epoch": f"{epoch:.3f}",
                "seed": self._seed,
                "type": ",".join(type_names) if type_names else "?",
                "loss": f"{metrics.loss:.6f}",
                "learning_rate": f"{metrics.learning_rate:.2e}",
                "mean_reward": f"{metrics.mean_reward:.4f}",
                "reward_std": f"{metrics.reward_std:.4f}",
                "kl_divergence": f"{metrics.kl_divergence:.6f}",
                "mean_advantage": f"{mean_adv:.6f}",
                "grad_norm": f"{metrics.grad_norm:.4f}",
                "tokens_per_sec": f"{metrics.tokens_per_sec:.1f}",
                "generated_tokens": metrics.generated_tokens,
                "trained_tokens": metrics.extras.get("trained_tokens", metrics.generated_tokens),
                "group_size": metrics.extras.get("group_size", 0),
                "total_views": metrics.extras.get("total_views", 0),
                "samples_per_step": self._samples_per_step,
                "total_generations": total_gens,
                "num_correct": num_correct,
                "num_incorrect": num_incorrect,
                "generation_sec": f"{metrics.generation_sec:.3f}",
                "elapsed_sec": f"{metrics.elapsed_sec:.3f}",
                "peak_memory_mb": f"{metrics.peak_memory_mb:.1f}",
                "rss_mb": f"{metrics.rss_mb:.1f}",
                "cka_drift": (
                    f"{metrics.extras['cka_drift']:.4f}" if "cka_drift" in metrics.extras else ""
                ),
                **self._is_weight_row(metrics.extras.get("is_weight_stats", [])),
                "dataset_row": metrics.extras.get("dataset_row", ""),
                "repeat_idx": metrics.extras.get("repeat_idx", ""),
                "repeat_total": metrics.extras.get("repeat_total", "") or "",
                "repeat_done_reason": metrics.extras.get("repeat_done_reason", ""),
                "naked_phase": metrics.extras.get("naked_phase", ""),
                "weight_angle_avg": (
                    f"{sum(metrics.extras['weight_angles'].values()) / len(metrics.extras['weight_angles']):.4f}"  # noqa: E501
                    if metrics.extras.get("weight_angles")
                    else ""
                ),
                **gradient_mult_row,
                **gar_row,
                **per_layer_row,
                **self._judge_row(metrics.extras.get("judge_stats", {})),
            }
        )
        self._file.flush()

    @staticmethod
    def _is_weight_row(is_stats: list[dict[str, float]]) -> dict[str, str]:
        if not is_stats:
            return {"is_weight_mean": "", "is_weight_min": "", "is_weight_max": ""}
        means = [s["mean"] for s in is_stats]
        return {
            "is_weight_mean": f"{sum(means) / len(means):.4f}",
            "is_weight_min": f"{min(s['min'] for s in is_stats):.4f}",
            "is_weight_max": f"{max(s['max'] for s in is_stats):.4f}",
        }

    @staticmethod
    def _judge_row(stats: dict[str, float]) -> dict[str, str]:
        if not stats:
            return {
                "judge_calls": "",
                "judge_cache_hits": "",
                "judge_failures": "",
                "judge_correct": "",
                "judge_mean_confidence": "",
                "judge_mean_latency_ms": "",
            }
        return {
            "judge_calls": str(int(stats.get("judge_calls", 0))),
            "judge_cache_hits": str(int(stats.get("judge_cache_hits", 0))),
            "judge_failures": str(int(stats.get("judge_failures", 0))),
            "judge_correct": str(int(stats.get("judge_correct", 0))),
            "judge_mean_confidence": f"{stats.get('judge_mean_confidence', 0):.4f}",
            "judge_mean_latency_ms": f"{stats.get('judge_mean_latency_ms', 0):.1f}",
        }

    def log_scaffold(self, probe_result: Any, iteration: int) -> None:
        """Append scaffold probe results to a separate CSV file."""
        scaffold_path = self._path.parent / "scaffold_accuracy.csv"
        exists = scaffold_path.exists() and scaffold_path.stat().st_size > 0
        ratios = sorted(probe_result.per_ratio_accuracy.keys())
        cols = ["iteration"]
        for r in ratios:
            cols += [f"mean_{r:.2f}", f"std_{r:.2f}"]
        with open(scaffold_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            row: dict[str, str] = {"iteration": str(iteration)}
            for r in ratios:
                row[f"mean_{r:.2f}"] = f"{probe_result.per_ratio_accuracy[r]:.3f}"
                row[f"std_{r:.2f}"] = f"{probe_result.per_ratio_reward_std.get(r, 0.0):.3f}"
            writer.writerow(row)

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
