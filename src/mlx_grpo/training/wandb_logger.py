"""WandB integration — optional experiment tracking.

Lazy initialization: wandb is only imported when the first log call is made.
If wandb is not installed, the logger operates as a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class WandbLogger:
    """WandB logger with eager initialization.

    WandB MUST be initialized before any Metal GPU operations to prevent
    its background sync threads from triggering macOS Metal interactivity
    timeouts. Initialize this before loading the model.
    """

    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        tags: list[str] | None = None,
    ):
        self._project = project
        self._config = config or {}
        self._run_id = run_id
        self._tags = tags
        self._run: Any = None
        self._available: bool = False

        # Initialize eagerly — must happen before GPU operations
        try:
            import wandb

            self._run = wandb.init(
                project=self._project,
                config=self._config,
                id=self._run_id,
                resume="allow" if self._run_id else None,
                tags=self._tags,
            )
            self._run_id = self._run.id
            self._available = True
        except (ImportError, Exception) as e:
            print(f"WandB not available: {e}")
            self._available = False

    def log_step(self, metrics: Any, step: int) -> None:
        """Log metrics for a training step.

        Accepts either a plain dict[str, float] (legacy) or a StepMetrics object.
        When metrics.extras contains GAR keys (populated in GAR mode), per-layer
        angle metrics are logged under the gar/ prefix.
        """
        if not self._available:
            return
        import wandb

        # Legacy dict path — unchanged
        if isinstance(metrics, dict):
            wandb.log(metrics, step=step)
            return

        # StepMetrics object path
        scalar_metrics: dict[str, float] = {
            "loss": metrics.loss,
            "mean_reward": metrics.mean_reward,
            "reward_std": metrics.reward_std,
            "kl_divergence": metrics.kl_divergence,
            "grad_norm": metrics.grad_norm,
            "tokens_per_sec": metrics.tokens_per_sec,
            "generated_tokens": float(metrics.generated_tokens),
            "generation_sec": metrics.generation_sec,
            "elapsed_sec": metrics.elapsed_sec,
            "peak_memory_mb": metrics.peak_memory_mb,
        }
        wandb.log(scalar_metrics, step=step)

        # Judge stats
        judge_stats = metrics.extras.get("judge_stats", {})
        if judge_stats:
            wandb.log(
                {f"judge/{k}": v for k, v in judge_stats.items()},
                step=step,
            )

        # GAR-mode extra metrics
        if "gar_loss" in metrics.extras:
            gar_metrics = metrics.extras.get("gar_angles", {})
            wandb.log(
                {
                    "gar/loss": metrics.extras["gar_loss"],
                    "gar/scaffold_fraction": metrics.extras.get("gar_scaffold_fraction", 0.0),
                },
                step=step,
            )
            for layer_key, angles_dict in gar_metrics.items():
                wandb.log(
                    {
                        f"gar/angle_mean/{layer_key}": angles_dict["angle_mean"],
                        f"gar/angle_std/{layer_key}": angles_dict["angle_std"],
                        f"gar/angle_max/{layer_key}": angles_dict["angle_max"],
                        f"gar/loss_per_layer/{layer_key}": angles_dict.get("loss", 0.0),
                        f"gar/tokens_above_threshold/{layer_key}": angles_dict.get(
                            "tokens_above", 0
                        ),
                        f"gar/tokens_below_threshold/{layer_key}": angles_dict.get(
                            "tokens_below", 0
                        ),
                    },
                    step=step,
                )

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def save_run_id(self, path: Path) -> None:
        """Persist run_id for resume."""
        if self._run_id:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"run_id": self._run_id}))

    @staticmethod
    def load_run_id(path: Path) -> str | None:
        """Load persisted run_id."""
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("run_id")
        return None

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        paths: list[Path],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upload files as a versioned wandb artifact."""
        if not self._available:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        for p in paths:
            if Path(p).exists():
                artifact.add_file(str(p))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        """Close the wandb run."""
        if self._run is not None:
            self._run.finish()
