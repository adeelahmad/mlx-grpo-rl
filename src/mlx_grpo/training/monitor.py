"""Training monitor — rich live progress bar + KL guardrails.

Renders a live in-place display with:
  - Progress bar: step/total, %, elapsed, ETA
  - Stats panel: loss/reward/KL/∇norm/step-time + tokens + system
  - Trend arrows (↑↓→) from rolling window comparison
  - KL warning/critical banners
  - Falls back to tqdm → ASCII when not in a terminal or TUI mode.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from mlx_grpo.training import StepMetrics, Trainer

try:
    from rich.console import Console
    from rich.console import Group as RenderGroup
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None  # type: ignore[assignment]


@dataclass
class MonitorConfig:
    enabled: bool = True
    kl_warning: float = 0.04
    kl_critical: float = 0.08
    stop_on_critical: bool = False
    log_every: int = 1
    window_size: int = 50
    log_file: str | None = None


class MetricsWindow:
    def __init__(self, size: int = 50):
        self._values: deque[float] = deque(maxlen=size)

    def add(self, v: float) -> None:
        self._values.append(v)

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    @property
    def latest(self) -> float:
        return self._values[-1] if self._values else 0.0

    @property
    def min(self) -> float:
        return min(self._values) if self._values else 0.0

    @property
    def max(self) -> float:
        return max(self._values) if self._values else 0.0

    def trend(self, higher_is_better: bool = False) -> str:
        """Colored trend arrow comparing recent half vs older half of window."""
        if len(self._values) < 6:
            return ""
        vals = list(self._values)
        half = len(vals) // 2
        older = sum(vals[:half]) / half
        recent = sum(vals[half:]) / half
        delta = (recent - older) / (abs(older) + 1e-9)
        if abs(delta) < 0.02:
            return " [dim]→[/]"
        improving = (delta < 0) if not higher_is_better else (delta > 0)
        arrow = "↑" if delta > 0 else "↓"
        color = "green" if improving else "red"
        return f" [{color}]{arrow}[/]"

    def __len__(self) -> int:
        return len(self._values)


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


class Monitor:
    """Training monitor with rich live progress + KL guardrails."""

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self._windows: dict[str, MetricsWindow] = {}
        self._log_file: TextIO | None = None
        self._start_time = time.time()
        self._log_fn: Callable[[str], None] | None = None  # TUI bridge

        # Rich live display
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._repeat_task_id: TaskID | None = None
        self._console: Console | None = None
        self._pbar: Any = None  # tqdm fallback
        self._total: int = 0

        # Step timing
        self._last_step_t: float = time.time()
        self._step_times: deque[float] = deque(maxlen=30)

        if self.config.log_file:
            Path(self.config.log_file).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.config.log_file, "a")  # noqa: SIM115

    def track(self, name: str, value: float) -> None:
        if name not in self._windows:
            self._windows[name] = MetricsWindow(self.config.window_size)
        self._windows[name].add(value)

    def get(self, name: str) -> MetricsWindow:
        return self._windows.get(name, MetricsWindow(1))

    def step_hook(self, trainer: Trainer, metrics: StepMetrics) -> None:
        if not self.config.enabled:
            return
        now = time.time()
        self._step_times.append(now - self._last_step_t)
        self._last_step_t = now

        self.track("loss", metrics.loss)
        self.track("reward", metrics.mean_reward)
        self.track("reward_std", metrics.reward_std)
        self.track("kl", metrics.kl_divergence)
        self.track("grad_norm", metrics.grad_norm)
        self.track("tokens", float(metrics.generated_tokens))
        self.track("peak_memory_mb", metrics.peak_memory_mb)
        total_views = metrics.extras.get("total_views", 0)
        if total_views:
            self.track("total_views", float(total_views))
        self._check_kl(metrics.kl_divergence, trainer)
        cka = metrics.extras.get("cka_drift")
        if cka is not None:
            self.track("cka_drift", float(cka))
        if int(metrics.iteration) % self.config.log_every == 0:
            self._log_step(metrics, trainer)

    def _check_kl(self, kl: float, trainer: Trainer) -> None:
        if kl > self.config.kl_critical:
            self._emit(f"⚠  KL CRITICAL: {kl:.4f} > {self.config.kl_critical}", level="critical")
            if self.config.stop_on_critical:
                raise RuntimeError(f"KL divergence exceeded critical threshold: {kl:.4f}")
        elif kl > self.config.kl_warning:
            self._emit(f"⚡ KL warning: {kl:.4f} > {self.config.kl_warning}", level="warning")

    def _log_step(self, metrics: StepMetrics, trainer: Trainer) -> None:
        total = trainer.config.iterations
        sc = metrics.extras.get("scaffold_tokens", 0)
        gen = metrics.generated_tokens
        used = metrics.extras.get("trained_tokens", 0)
        views = metrics.extras.get("total_views", 0)
        _gen_sec = metrics.generation_sec if metrics.generation_sec > 0 else metrics.elapsed_sec
        tps = gen / max(_gen_sec, 0.001)
        mem = metrics.peak_memory_mb
        lr = metrics.learning_rate
        kl = metrics.kl_divergence
        cka = metrics.extras.get("cka_drift")
        loss_avg = self._windows.get("loss", MetricsWindow(1)).mean
        rew_avg = self._windows.get("reward", MetricsWindow(1)).mean
        type_names = ",".join(metrics.extras.get("type_names", []) or ["?"])
        avg_step = sum(self._step_times) / len(self._step_times) if self._step_times else 0
        remaining = max(total - metrics.iteration, 0)
        eta_s = avg_step * remaining
        accum_count = metrics.extras.get("accum_count", 0)
        accum_steps = metrics.extras.get("accum_steps", 1)
        accum_over_repeats = metrics.extras.get("accum_over_repeats", 0)
        # Show accumulated steps:
        # - If accum_over_repeats > 0: show "accN/R" (N steps accumulated over R repeats)
        # - Else if accum_count > 0: show "acc:N/M" (N steps of M-step accumulation)
        # - Else if accum_steps > 1: show "acc:✓" (boundary just applied)
        _accum_str = ""
        if accum_over_repeats > 0:
            _accum_str = f"acc{accum_over_repeats}/R"
        elif accum_steps > 1:
            _accum_str = f"acc:{accum_count}/{accum_steps}" if accum_count > 0 else "acc:✓"
        repeat_idx: int = int(metrics.extras.get("repeat_idx", 0) or 0)
        repeat_total: int = int(metrics.extras.get("repeat_total", 0) or 0)
        naked_phase: str = metrics.extras.get("naked_phase", "")
        angles: dict[str, float] = metrics.extras.get("weight_angles", {})
        angle_avg: float | None = sum(angles.values()) / len(angles) if angles else None
        angle_delta: float | None = metrics.extras.get("weight_angle_delta")
        gar_angles: dict[str, dict] = metrics.extras.get("gar_angles", {})
        gar_loss: float | None = metrics.extras.get("gar_loss")
        gar_threshold: float = metrics.extras.get("gar_drift_threshold_deg", 5.0)

        # ── TUI / log-fn mode ──
        if self._log_fn is not None:
            pct = metrics.iteration / max(total, 1)
            bar = "=" * int(pct * 20) + ">" + "." * (19 - int(pct * 20))
            line = (
                f"[{bar}] {pct * 100:.1f}% {metrics.iteration}/{total}"
                f"  loss={metrics.loss:.4f}(avg={loss_avg:.4f})"
                f"  r={metrics.mean_reward:.3f}(avg={rew_avg:.3f})"
                f"  kl={kl:.4f}  ∇={metrics.grad_norm:.3f}"
                f"  step={avg_step:.1f}s  ETA={_fmt_time(eta_s)}"
                f"  tok=s{sc}/g{gen}/u{used}"
                + (f"  v={views}" if views else "")
                + (f"  {_accum_str}" if _accum_str else "")
                + (f"  cka={cka:.4f}" if cka is not None else "")
                + (
                    f"  ∠={angle_avg:.4f}°({angle_delta:+.4f}°)"
                    if angle_avg is not None and angle_delta is not None
                    else (f"  ∠={angle_avg:.4f}°" if angle_avg is not None else "")
                )
                + (
                    f"  rep={repeat_idx}/{repeat_total or '?'}"
                    f"{('(' + naked_phase + ')') if naked_phase else ''}"
                    if repeat_idx
                    else ""
                )
                + (f"  gar={gar_loss:.4f}" if gar_loss is not None else "")
                + f"  {tps:.0f}/s  [{type_names}]"
            )
            self._log_fn(line)
            self._write_log(metrics)
            return

        # ── Rich live display ──
        if _RICH:
            self._render_rich(
                metrics,
                total,
                loss_avg,
                rew_avg,
                tps,
                mem,
                lr,
                views,
                type_names,
                sc,
                gen,
                used,
                cka,
                avg_step,
                eta_s,
                _accum_str,
                repeat_idx,
                repeat_total,
                angle_avg,
                angle_delta,
                angles or None,
                gar_angles or None,
                gar_loss,
                gar_threshold,
                naked_phase,
            )
            self._write_log(metrics)
            return

        # ── tqdm fallback ──
        if _tqdm is not None:
            if self._pbar is None:
                self._pbar = _tqdm(
                    total=total,
                    initial=max(0, metrics.iteration - 1),
                    unit="iter",
                    dynamic_ncols=True,
                    colour="cyan",
                )
            postfix: dict[str, str] = {
                "loss": f"{metrics.loss:.4f}",
                "avg_l": f"{loss_avg:.4f}",
                "r": f"{metrics.mean_reward:.3f}",
                "avg_r": f"{rew_avg:.3f}",
                "kl": f"{kl:.4f}",
                "∇": f"{metrics.grad_norm:.3f}",
                "tok/s": f"{tps:.0f}",
                "step": f"{avg_step:.1f}s",
                "ETA": _fmt_time(eta_s),
            }
            if _accum_str:
                if accum_over_repeats > 0:
                    postfix["acc"] = f"{accum_over_repeats}/R"
                else:
                    postfix["acc"] = f"{accum_count}/{accum_steps}" if accum_count > 0 else "✓"
            if cka is not None:
                postfix["cka"] = f"{cka:.4f}"
            if angle_avg is not None:
                postfix["∠"] = f"{angle_avg:.4f}°" + (
                    f"({angle_delta:+.4f}°)" if angle_delta is not None else ""
                )
            if repeat_idx:
                _np = f"({naked_phase})" if naked_phase else ""
                postfix["rep"] = f"{repeat_idx}/{repeat_total if repeat_total else '?'}{_np}"
            if gar_loss is not None:
                postfix["gar"] = f"{gar_loss:.4f}"
            self._pbar.set_postfix(**postfix, refresh=False)
            self._pbar.update(1)
            self._write_log(metrics)
            return

        # ── ASCII fallback ──
        elapsed = time.time() - self._start_time
        pct = metrics.iteration / max(total, 1)
        bar = "#" * int(pct * 30) + "." * (30 - int(pct * 30))
        print(
            f"\r[{bar}] {pct * 100:.0f}%  {metrics.iteration}/{total}"
            f"  loss={metrics.loss:.4f}(avg={loss_avg:.4f})"
            f"  r={metrics.mean_reward:.3f}  kl={kl:.4f}"
            f"  ∇={metrics.grad_norm:.3f}  {tps:.0f}tok/s"
            f"  step={avg_step:.1f}s  ETA={_fmt_time(eta_s)}  [{_fmt_time(elapsed)}]"
            + (f"  {_accum_str}" if _accum_str else "")
            + (f"  cka={cka:.4f}" if cka is not None else "")
            + (
                f"  ∠={angle_avg:.4f}°({angle_delta:+.4f}°)"
                if angle_avg is not None and angle_delta is not None
                else (f"  ∠={angle_avg:.4f}°" if angle_avg is not None else "")
            )
            + (f"  rep={repeat_idx}/{repeat_total if repeat_total else '?'}" if repeat_idx else "")
            + (f"  gar={gar_loss:.4f}" if gar_loss is not None else ""),
            end="",
            flush=True,
        )
        self._write_log(metrics)

    def _render_rich(
        self,
        metrics: StepMetrics,
        total: int,
        loss_avg: float,
        rew_avg: float,
        tps: float,
        mem: float,
        lr: float,
        views: int,
        type_names: str,
        sc: int = 0,
        gen: int = 0,
        used: int = 0,
        cka: float | None = None,
        avg_step: float = 0.0,
        eta_s: float = 0.0,
        accum_str: str = "",
        repeat_idx: int = 0,
        repeat_total: int = 0,
        angle_avg: float | None = None,
        angle_delta: float | None = None,
        angles: dict | None = None,
        gar_angles: dict | None = None,
        gar_loss: float | None = None,
        gar_threshold: float = 5.0,
        naked_phase: str = "",
    ) -> None:
        if self._progress is None:
            self._console = Console()
            self._progress = Progress(
                SpinnerColumn(style="bold cyan"),
                TextColumn("[bold cyan]MLX-GRPO[/]"),
                BarColumn(bar_width=40, style="dim cyan", complete_style="cyan"),
                TextColumn("[bold]{task.percentage:>5.1f}%[/]"),
                MofNCompleteColumn(),
                TextColumn("[dim]•[/]"),
                TimeElapsedColumn(),
                TextColumn("[dim]ETA[/]"),
                TimeRemainingColumn(),
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._task_id = self._progress.add_task("train", total=total)
            self._repeat_task_id = self._progress.add_task(
                "repeat", total=1, visible=False, start=False
            )
            self._live = Live(
                console=self._console,
                refresh_per_second=4,
                vertical_overflow="visible",
            )
            self._live.start()

        # Main bar: floor to int so MofNCompleteColumn shows clean "1434/7140"
        self._progress.update(self._task_id, completed=int(metrics.iteration) + 1)

        # Sub-bar: visible only during repeat, shows N/M or N/? progress
        if repeat_idx > 0 and self._repeat_task_id is not None:
            rt = repeat_total if repeat_total > 0 else repeat_idx + 1
            self._progress.update(
                self._repeat_task_id,
                completed=repeat_idx,
                total=rt,
                description="  [dim magenta]↳ repeat[/]",
                visible=True,
                start=True,
            )
        elif self._repeat_task_id is not None:
            self._progress.update(self._repeat_task_id, visible=False)

        kl = metrics.kl_divergence
        kl_color = (
            "red"
            if kl > self.config.kl_critical
            else ("yellow" if kl > self.config.kl_warning else "green")
        )
        loss_w = self._windows.get("loss", MetricsWindow(1))
        rew_w = self._windows.get("reward", MetricsWindow(1))
        last_step = self._step_times[-1] if self._step_times else 0.0

        # ── Stats grid (2 columns: metrics | tokens+system) ──
        t = Table.grid(padding=(0, 2))
        t.add_column(min_width=10, style="dim")  # label1
        t.add_column(min_width=36)  # value1
        t.add_column(min_width=10, style="dim")  # label2
        t.add_column(min_width=18)  # value2

        def row(l1: str, v1: str, l2: str = "", v2: str = "") -> None:
            t.add_row(l1, v1, l2, v2)

        row(
            "loss",
            f"[yellow]{metrics.loss:.4f}[/]  [dim]avg {loss_avg:.4f}[/]{loss_w.trend(False)}",
            "scaffold",
            f"[dim]{sc} tok[/]",
        )
        row(
            "reward",
            f"[green]{metrics.mean_reward:.4f}[/] [dim]±{metrics.reward_std:.3f}"
            f"  avg {rew_avg:.4f}[/]{rew_w.trend(True)}",
            "generated",
            f"[dim]{gen} tok[/]",
        )
        row(
            "KL div",
            f"[{kl_color}]{kl:.7f}[/]",
            "trained",
            f"[dim]{used} tok[/]",
        )
        row(
            "∇ norm",
            f"[cyan]{metrics.grad_norm:.4f}[/]",
            "tok/s",
            f"[cyan]{tps:.0f}[/]",
        )
        row(
            "step",
            f"[white]{last_step:.1f}s[/]  [dim]avg {avg_step:.1f}s[/]",
            "ETA",
            f"[bold white]{_fmt_time(eta_s)}[/]",
        )
        if mem > 0 or lr > 0:
            row(
                "memory" if mem > 0 else "",
                f"[magenta]{mem / 1024:.2f} GB[/]" if mem > 0 else "",
                "LR" if lr > 0 else "",
                f"[blue]{lr:.2e}[/]" if lr > 0 else "",
            )
        row(
            "types",
            f"[dim]{type_names}[/]" + (f"  [dim]views {views}[/]" if views else ""),
            "grad acc" if accum_str else "",
            f"[cyan]{accum_str}[/]" if accum_str else "",
        )
        if cka is not None:
            cka_color = "red" if cka < 0.8 else ("yellow" if cka < 0.95 else "green")
            row("cka drift", f"[{cka_color}]{cka:.4f}[/]", "", "")
        if angle_avg is not None:
            _delta_str = f"  [dim]Δ{angle_delta:+.4f}°[/]" if angle_delta is not None else ""
            row("∠ drift", f"[cyan]{angle_avg:.4f}°[/]{_delta_str}  [dim]from init[/]", "", "")
            if angles:
                for lk in sorted(angles):
                    row(f"  {lk}", f"[dim cyan]{angles[lk]:.4f}°[/]", "", "")
        if gar_loss is not None and gar_angles:
            gar_vals = list(gar_angles.values())
            avg_gar = sum(v.get("angle_mean", 0.0) for v in gar_vals) / len(gar_vals)
            row("∠ gar", f"[cyan]{avg_gar:.1f}°[/]  [dim]loss={gar_loss:.4f}[/]", "", "")
            layers_sorted = sorted(
                gar_angles.items(),
                key=lambda x: x[1].get("angle_mean", 0.0),
                reverse=True,
            )
            for lk, ad in layers_sorted[:10]:
                angle = ad.get("angle_mean", 0.0)
                marker = " [yellow]★[/]" if angle > gar_threshold else ""
                row(f"  {lk}", f"[dim cyan]{angle:.2f}°[/]{marker}", "", "")
        if repeat_idx > 0:
            rt_str = f"{repeat_total}" if repeat_total > 0 else "?"
            sub = f"{metrics.iteration:.1f}"
            _np_tag = f"  [yellow]{naked_phase}[/]" if naked_phase else ""
            row(
                "repeat",
                f"[bold magenta]{repeat_idx}/{rt_str}[/]  [dim]sub-step {sub}[/]{_np_tag}",
                "",
                "",
            )

        self._live.update(RenderGroup(self._progress, t))

    def mark_repeat_start(self, rep_idx: int, sr: int) -> None:
        """Update sub-bar at the START of a rep so it shows during the rollout."""
        if self._progress is None or self._repeat_task_id is None:
            return
        total = rep_idx + 1 if sr < 0 else sr
        self._progress.update(
            self._repeat_task_id,
            completed=rep_idx - 1,
            total=total,
            description="  [dim magenta]↳ rolling…[/]",
            visible=True,
            start=True,
        )

    def _write_log(self, metrics: StepMetrics) -> None:
        if not self._log_file:
            return
        record = {
            "iteration": metrics.iteration,
            "loss": metrics.loss,
            "mean_reward": metrics.mean_reward,
            "reward_std": metrics.reward_std,
            "kl": metrics.kl_divergence,
            "grad_norm": metrics.grad_norm,
            "tokens": metrics.generated_tokens,
            "total_views": metrics.extras.get("total_views", 0),
            "elapsed": metrics.elapsed_sec,
            "peak_memory_mb": metrics.peak_memory_mb,
            "rewards": metrics.rewards,
        }
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def _emit(self, message: str, level: str = "info") -> None:
        if self._log_fn is not None:
            self._log_fn(message)
        elif self._live is not None and self._console:
            color = "red" if level == "critical" else "yellow"
            self._console.print(f"[{color}]{message}[/]")
        elif self._pbar is not None:
            self._pbar.write(message)
        else:
            import sys

            stream = sys.stderr if level in ("warning", "critical") else sys.stdout
            print(message, file=stream, flush=True)

    def log_scaffold(self, probe_result: Any, iteration: int) -> None:
        """Called by scaffold probe hook to record probe results."""
        pass  # stats_logger handles logging; hook here for extensibility

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {"mean": w.mean, "latest": w.latest, "min": w.min, "max": w.max, "count": len(w)}
            for name, w in self._windows.items()
        }

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
        if self._pbar is not None:
            self._pbar.close()
        if self._log_file:
            self._log_file.close()
