"""System resource bar — colorful GPU/CPU/memory stats with sparklines.

Polls system resources on a timer and renders a single-line status bar
with colored gauges and mini sparkline history.

    bar = SystemBar(id="system-bar")
    bar.poll()  # called by timer
"""

from __future__ import annotations

import os
import time
from collections import deque

from rich.text import Text
from textual.widgets import Static

_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


class SystemBar(Static):
    """Bottom bar: colored GPU/CPU/memory gauges with sparkline history."""

    DEFAULT_CSS = """
    SystemBar {
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(Text(" System: initializing..."), id=id)
        self._gpu_hist: deque[float] = deque(maxlen=20)
        self._cpu_hist: deque[float] = deque(maxlen=20)
        self._rss_hist: deque[float] = deque(maxlen=20)
        self._prev_times = os.times()
        self._prev_wall = time.monotonic()

    def poll(self) -> None:
        """Sample system stats and re-render."""
        gpu_mb, gpu_peak = self._sample_gpu()
        rss_mb = self._sample_rss()
        cpu_pct = self._sample_cpu()

        self._gpu_hist.append(gpu_mb)
        self._cpu_hist.append(cpu_pct)
        self._rss_hist.append(rss_mb)
        self._refresh_bar(gpu_mb, gpu_peak, rss_mb, cpu_pct)

    def _sample_gpu(self) -> tuple[float, float]:
        try:
            import mlx.core as mx

            active = mx.metal.get_active_memory() / (1024**2)
            peak = mx.metal.get_peak_memory() / (1024**2)
            return active, peak
        except (ImportError, AttributeError):
            return 0.0, 0.0

    def _sample_rss(self) -> float:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        except (ImportError, AttributeError):
            return 0.0

    def _sample_cpu(self) -> float:
        now = time.monotonic()
        times = os.times()
        dt = now - self._prev_wall
        used = (times.user - self._prev_times.user) + (times.system - self._prev_times.system)
        self._prev_times, self._prev_wall = times, now
        ncpu = os.cpu_count() or 1
        return min((used / max(dt, 0.01)) / ncpu * 100, 100.0)

    def _refresh_bar(self, gpu_mb: float, gpu_peak: float, rss_mb: float, cpu_pct: float) -> None:
        t = Text()
        # GPU gauge
        gpu_gb = gpu_mb / 1024
        peak_gb = gpu_peak / 1024
        gc = "green" if gpu_gb < 4 else "yellow" if gpu_gb < 8 else "red"
        t.append(" GPU ", style="bold")
        t.append(f"{gpu_gb:.1f}", style=gc)
        t.append(f"/{peak_gb:.1f}G ", style="dim")
        t.append(_bar(gpu_mb, max(gpu_peak, 1)), style=gc)

        t.append("  \u2502 ", style="dim")

        # RSS
        rss_gb = rss_mb / 1024
        t.append("RSS ", style="bold")
        t.append(f"{rss_gb:.1f}G", style="blue")

        t.append("  \u2502 ", style="dim")

        # CPU gauge + sparkline
        cc = "cyan" if cpu_pct < 50 else "yellow" if cpu_pct < 80 else "red"
        t.append("CPU ", style="bold")
        t.append(f"{cpu_pct:.0f}% ", style=cc)
        t.append(_sparkline(self._cpu_hist), style=cc)

        t.append("  \u2502 ", style="dim")

        # GPU sparkline
        t.append("GPU ", style="bold dim")
        t.append(_sparkline(self._gpu_hist), style=gc)

        self.update(t)


def _bar(value: float, max_val: float, width: int = 8) -> str:
    if max_val <= 0:
        return "\u2591" * width
    filled = int(width * min(value / max_val, 1.0))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _sparkline(history: deque[float], width: int = 10) -> str:
    if not history:
        return " " * width
    data = list(history)[-width:]
    lo, hi = min(data), max(data)
    rng = hi - lo if hi > lo else 1
    return "".join(_SPARK[min(int((v - lo) / rng * 7), 7)] for v in data)
