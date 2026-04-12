"""Plotext chart widgets — renders metric curves in-terminal.

Generic MetricChart renders named series via plotext.build().
Supports line, bar, and box chart types for visual variety.
Uses Rich Text.from_ansi() to convert plotext ANSI output for Textual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.text import Text
from textual.widgets import Static

# Distinctive colors for multi-series charts
COLORS = [
    "red+",
    "green+",
    "cyan+",
    "yellow+",
    "magenta+",
    "blue+",
    "orange",
    "white",
]


@dataclass
class Series:
    """A single data series within a chart."""

    name: str
    color: str = "green+"
    values: list[float] = field(default_factory=list)


class MetricChart(Static):
    """Generic terminal chart — plots one or more named series.

    chart_type controls rendering:
      "line"  — time-series line plot (default)
      "bar"   — horizontal bar chart (latest values per series)
      "box"   — box plot (distribution from raw values per series)
    """

    DEFAULT_CSS = """
    MetricChart {
        width: 1fr;
        height: 1fr;
        padding: 0;
    }
    """

    def __init__(
        self,
        title: str = "",
        series_names: list[str] | None = None,
        id: str | None = None,
        colors: list[str] | None = None,
        chart_type: str = "line",
    ) -> None:
        super().__init__(f"  [{title}] waiting for data...", id=id)
        self._title = title
        self._steps: list[int] = []
        self._series: dict[str, Series] = {}
        self._chart_type = chart_type
        self._raw_groups: dict[str, list[float]] = {}  # for box plots
        for i, name in enumerate(series_names or []):
            c = colors[i] if colors and i < len(colors) else COLORS[i % len(COLORS)]
            self._series[name] = Series(name=name, color=c)

    def push_scalar(self, step: int, name: str, value: float) -> None:
        """Push a single scalar value to a named series."""
        if name not in self._series:
            idx = len(self._series)
            self._series[name] = Series(name=name, color=COLORS[idx % len(COLORS)])
        self._series[name].values.append(value)
        if not self._steps or self._steps[-1] != step:
            self._steps.append(step)

    def push_dict(self, step: int, values: dict[str, float]) -> None:
        """Push multiple named values at once."""
        for name, val in values.items():
            self.push_scalar(step, name, val)

    def push_box_data(self, name: str, values: list[float]) -> None:
        """Store raw values for box plot rendering."""
        self._raw_groups[name] = values

    def render_chart(self) -> None:
        """Re-render the chart from current data."""
        try:
            import plotext as plt
        except ImportError:
            self.update("plotext not installed")
            return

        w, h = self._get_size()
        if w < 10 or h < 3:
            self._render_fallback()
            return

        plt.clf()
        plt.plotsize(w, h)
        plt.theme("dark")

        if self._chart_type == "bar":
            self._render_bar(plt)
        elif self._chart_type == "box":
            self._render_box(plt)
        else:
            self._render_line(plt)

    def _render_line(self, plt: object) -> None:
        """Standard time-series line plot."""
        has_data = False
        for series in self._series.values():
            if len(series.values) < 2:
                continue
            steps = self._steps[: len(series.values)]
            plt.plot(steps, series.values, label=series.name, color=series.color)
            has_data = True

        if not has_data:
            for series in self._series.values():
                if series.values:
                    steps = self._steps[: len(series.values)]
                    plt.scatter(steps, series.values, label=series.name, color=series.color)
                    has_data = True

        if not has_data:
            self.update(f"  [{self._title}] waiting for data...")
            return
        plt.title(self._title)
        self.update(Text.from_ansi(plt.build()))

    def _render_bar(self, plt: object) -> None:
        """Horizontal bar chart — latest values per series."""
        names, vals, colors = [], [], []
        for s in self._series.values():
            if s.values:
                names.append(s.name[:12])
                vals.append(s.values[-1])
                colors.append(s.color)
        if not names:
            self.update(f"  [{self._title}] waiting for data...")
            return
        plt.bar(names, vals, orientation="h", color=colors, width=3 / 5)
        plt.title(self._title)
        self.update(Text.from_ansi(plt.build()))

    def _render_box(self, plt: object) -> None:
        """Box plot — distribution from raw values per group."""
        if not self._raw_groups:
            self.update(f"  [{self._title}] waiting for data...")
            return
        labels = list(self._raw_groups.keys())
        data = [self._raw_groups[k] for k in labels]
        labels = [lb[:10] for lb in labels]
        plt.box(labels, data, width=0.3)
        plt.title(self._title)
        self.update(Text.from_ansi(plt.build()))

    def _render_fallback(self) -> None:
        vals = {s.name: s.values[-1] for s in self._series.values() if s.values}
        if vals:
            txt = " | ".join(f"{k}={v:.4g}" for k, v in vals.items())
            self.update(f"  {self._title}: {txt}")

    def _get_size(self) -> tuple[int, int]:
        try:
            r = self.content_region
            return max(r.width, 15), max(r.height, 4)
        except Exception:
            return 40, 8
