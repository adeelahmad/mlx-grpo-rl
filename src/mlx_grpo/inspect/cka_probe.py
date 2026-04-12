"""CKAProbe — injectable Trainer hook for live CKA monitoring (Feature 6).

Zero changes to training/__init__.py. Register via:
    trainer.on("on_eval", probe.on_eval_hook)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .metrics import linear_cka


@dataclass(slots=True, frozen=True)
class CKAProbeResult:
    """Result of a single CKA probe evaluation."""

    iteration: int
    per_layer_cka: dict[str, float]
    mean_cka: float
    min_cka_layer: str
    min_cka_value: float


class CKAProbe:
    """Monitor representation drift during training via CKA.

    Compares current model activations against a saved base activation snapshot.
    Register as a Trainer hook — zero changes to Trainer required::

        probe = CKAProbe(model, tokenizer, cfg, base_activations_path="base_acts.npz")
        trainer.on("on_eval", probe.on_eval_hook)

    To save base activations before training::

        mlx-grpo inspect model_path --feature probe_save \\
            --prompt "..." --probe-base ./base_acts.npz
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cfg: Any,
        base_activations_path: str,
        wandb_logger: Any | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._base_acts: dict[str, npt.NDArray[Any]] | None = None
        self._base_path = base_activations_path
        self._wandb = wandb_logger
        self._iteration = 0
        self.latest_result: CKAProbeResult | None = None  # updated on each eval

    def _load_base(self, path: str) -> dict[str, npt.NDArray[Any]]:
        """Load base activations from a .npz file."""
        data = np.load(path)
        return {k: data[k] for k in data.files}

    def _ensure_base(self) -> dict[str, npt.NDArray[Any]]:
        """Lazily load base activations on first use."""
        if self._base_acts is None:
            self._base_acts = self._load_base(self._base_path)
        return self._base_acts

    def compute(self, iteration: int = 0) -> CKAProbeResult:
        """Collect current activations and compute per-layer CKA vs base.

        Returns CKAProbeResult with layer-wise CKA values.
        """
        from .collector import ActivationCollector  # noqa: PLC0415

        base = self._ensure_base()
        collector = ActivationCollector(self._model, self._tokenizer, self._cfg)
        result = collector.collect(self._cfg.prompt)

        per_layer_cka: dict[str, float] = {}
        for key in base:
            if key not in result.sketches:
                continue
            base_vec = base[key].reshape(1, -1).astype(np.float64)
            curr_vec = result.sketches[key].reshape(1, -1).astype(np.float64)
            per_layer_cka[key] = float(linear_cka(base_vec, curr_vec))

        if not per_layer_cka:
            return CKAProbeResult(
                iteration=iteration,
                per_layer_cka={},
                mean_cka=0.0,
                min_cka_layer="",
                min_cka_value=0.0,
            )

        mean_cka = float(np.mean(list(per_layer_cka.values())))
        min_key = min(per_layer_cka, key=per_layer_cka.__getitem__)
        min_val = per_layer_cka[min_key]

        return CKAProbeResult(
            iteration=iteration,
            per_layer_cka=per_layer_cka,
            mean_cka=mean_cka,
            min_cka_layer=min_key,
            min_cka_value=min_val,
        )

    def save_base_activations(self, prompt: str, out_path: str) -> None:
        """Save current model activations to a .npz file for use as base."""
        import dataclasses  # noqa: PLC0415

        from .collector import ActivationCollector  # noqa: PLC0415

        overridden_cfg = dataclasses.replace(self._cfg, prompt=prompt)
        collector2 = ActivationCollector(self._model, self._tokenizer, overridden_cfg)
        result = collector2.collect(prompt)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **result.sketches)

    def step_hook(self, trainer: Any, metrics: Any) -> None:
        """Trainer after_step callback. Injects latest CKA into metrics.extras.

        Register alongside on_eval_hook so stats_logger and monitor pick it up::

            trainer.on("after_step", probe.step_hook)
            trainer.on("on_eval", probe.on_eval_hook)
        """
        if self.latest_result is None:
            return
        extras = getattr(metrics, "extras", None)
        if extras is not None:
            extras["cka_drift"] = self.latest_result.mean_cka
            extras["cka_min_layer"] = self.latest_result.min_cka_layer

    def on_eval_hook(self, trainer: Any, metrics: Any) -> None:
        """Trainer on_eval callback. Computes CKA, stores latest_result, logs to WandB.

        Logs mean CKA plus every individual layer under
        ``{probe_wandb_key}/{layer_key}`` so per-layer drift is visible in WandB.
        """
        if isinstance(metrics, dict):
            iteration = int(metrics.get("iteration", self._iteration))
        else:
            iteration = int(getattr(metrics, "iteration", self._iteration))
        self._iteration = iteration + 1

        try:
            probe_result = self.compute(iteration=iteration)
        except Exception:  # noqa: BLE001
            return

        self.latest_result = probe_result

        if self._wandb is not None:
            with contextlib.suppress(Exception):
                log_dict: dict[str, float] = {
                    self._cfg.probe_wandb_key: probe_result.mean_cka,
                }
                for layer_key, cka_val in probe_result.per_layer_cka.items():
                    log_dict[f"{self._cfg.probe_wandb_key}/{layer_key}"] = cka_val
                self._wandb.log(log_dict, step=iteration)
