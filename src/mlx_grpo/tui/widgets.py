"""Custom widgets — progress header, metrics table, rollout viewer, log panel."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import RichLog, Static

if TYPE_CHECKING:
    from mlx_grpo.tui.bridge import MetricsSnapshot

_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def _sparkline(hist: deque[float], width: int = 8) -> str:
    if not hist:
        return " " * width
    data = list(hist)[-width:]
    lo, hi = min(data), max(data)
    rng = hi - lo if hi > lo else 1.0
    return "".join(_SPARK[min(int((v - lo) / rng * 7), 7)] for v in data)


class ProgressHeader(Static):
    """Top bar: iteration counter, progress bar, ETA."""

    DEFAULT_CSS = """
    ProgressHeader {
        height: 1;
        background: $accent;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        bar = "\u2591" * 20
        super().__init__(Text(f" Iter 0/0  [{bar}] 0%  ETA: --"), id=id)
        self._total = 0
        self._iteration = 0
        self._elapsed_sum = 0.0
        self._step_count = 0

    def refresh_from(self, snap: MetricsSnapshot) -> None:
        self._iteration = snap.iteration + 1
        self._total = snap.total_iterations
        self._elapsed_sum += snap.elapsed_sec
        self._step_count += 1
        self._render_bar()

    def _render_bar(self) -> None:
        total = max(self._total, 1)
        pct = self._iteration / total
        bar_w = 20
        filled = int(bar_w * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
        eta = self._estimate_eta()
        text = f" Iter {self._iteration}/{total}  [{bar}] {pct:.0%}  {eta}"
        self.update(Text(text))

    def _estimate_eta(self) -> str:
        if self._step_count <= 0 or self._elapsed_sum <= 0:
            return "ETA: --"
        avg = self._elapsed_sum / self._step_count
        remaining = (self._total - self._iteration) * avg
        if remaining < 60:
            return f"ETA: {remaining:.0f}s"
        if remaining < 3600:
            return f"ETA: {remaining / 60:.1f}m"
        return f"ETA: {remaining / 3600:.1f}h"


class StepProgress(Static):
    """Second bar: group progress and phase within a single iteration."""

    DEFAULT_CSS = """
    StepProgress {
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(Text("  Waiting for rollout..."), id=id)

    def set_rollout(self, group: int, total: int, label: str) -> None:
        bar_w = 15
        pct = group / max(total, 1)
        filled = int(bar_w * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
        t = Text()
        t.append(f" Group {group}/{total} ", style="bold")
        t.append(f"[{bar}] ", style="cyan")
        t.append("Rollout ", style="bold cyan")
        if label:
            t.append(f"({label})", style="dim")
        self.update(t)

    def set_gradient(self) -> None:
        t = Text()
        t.append(" Gradient ", style="bold yellow")
        t.append("\u2588" * 15, style="yellow")
        t.append(" Computing loss + backward", style="dim")
        self.update(t)


class MetricsTable(Static):
    """Right panel: key/value display of current metrics with colored values."""

    DEFAULT_CSS = """
    MetricsTable { width: 100%; padding: 1; }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__("  Waiting for first step...", id=id)
        self._reward_hist: dict[str, deque[float]] = {}

    def refresh_from(self, snap: MetricsSnapshot) -> None:
        t = Text()
        _row(t, "Loss", f"{snap.loss:.4f}", "red")
        _row(t, "Reward", f"{snap.mean_reward:.3f}", "green")
        _row(t, "Rew Std", f"{snap.reward_std:.3f}", "green dim")
        _row(t, "KL", f"{snap.kl_divergence:.4f}", "yellow")
        _row(t, "Adv", f"{snap.mean_advantage:+.4f}", "cyan")
        _row(t, "Grad", f"{snap.grad_norm:.4f}", "magenta")
        _row(t, "LR", f"{snap.learning_rate:.2e}", "blue")
        _row(t, "Tok/s", f"{snap.tokens_per_sec:.0f}", "white bold")
        _row(t, "Trained", f"{snap.trained_tokens:,}", "white")
        for tn, reward in snap.per_type_rewards.items():
            if tn not in self._reward_hist:
                self._reward_hist[tn] = deque(maxlen=24)
            self._reward_hist[tn].append(reward)
            spark = _sparkline(self._reward_hist[tn], 6)
            correct = snap.per_type_correct.get(tn, 0)
            count = snap.per_type_count.get(tn, 0)
            _row(t, tn[:8], f"{reward:.2f} {spark} {correct}/{count}", "green")
        if snap.esc_members:
            parts = []
            for midx, chunks, rew in snap.esc_members:
                lbl = "nk" if chunks == 0 else f"m{midx}·{chunks}c"
                ri = "\u2713" if rew >= 1.0 else "~" if rew > 0 else "\u2717"
                parts.append(f"{ri}{lbl}={rew:.2f}")
            _row(t, "Members", " ".join(parts), "cyan dim")
        self.update(t)


class LogPanel(RichLog):
    DEFAULT_CSS = "LogPanel { height: 100%; border: solid $primary; }"

    def __init__(self, id: str | None = None) -> None:
        super().__init__(wrap=True, highlight=True, max_lines=200, id=id)

    def log_event(self, text: str) -> None:
        self.write(text)


def _row(t: Text, label: str, value: str, color: str) -> None:
    t.append(f"  {label:<8s} ", style="bold")
    t.append(f"{value}\n", style=color)
