"""Textual TUI for live training visualization.

Provides a terminal dashboard with live charts (loss, reward, KL,
advantages, LR, grad norm), metrics table, streaming rollout viewer,
system resource bar, progress bars, and scrollable log panel.

    from mlx_grpo.tui import run_tui
    run_tui(train_fn, total_iterations=500)
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.worker import Worker, WorkerState

from mlx_grpo.tui.bridge import (
    LogEvent,
    MetricsBridge,
    MetricsSnapshot,
    MetricsUpdated,
    PauseStatusEvent,
    PhaseEvent,
    RolloutEvent,
    StreamSegmentEvent,
    StreamStartEvent,
    StreamTokenEvent,
)
from mlx_grpo.tui.charts import MetricChart
from mlx_grpo.tui.pause_panel import PausePanel
from mlx_grpo.tui.rollout_panel import RolloutPanel
from mlx_grpo.tui.system_bar import SystemBar
from mlx_grpo.tui.widgets import (
    LogPanel,
    MetricsTable,
    ProgressHeader,
    StepProgress,
)


class TrainingApp(App[None]):
    """Textual application for live training monitoring."""

    CSS = """
    Screen { layout: vertical; }
    #top-bar { height: 1; }
    #step-bar { height: 1; }
    #main-area { height: 1fr; }
    #left-col { width: 3fr; layout: vertical; }
    #rollout-pane { height: 1fr; }
    #pause-pane { height: 12; }
    #right-col { width: 2fr; layout: vertical; }
    #metrics-pane { height: 13; }
    .chart-row { height: 1fr; layout: horizontal; }
    #log-pane { height: 6; }
    #system-bar { height: 1; }
    """

    BINDINGS = [
        ("q", "stop_training", "Quit"),
        ("ctrl+q", "stop_training", "Quit"),
        ("ctrl+c", "stop_training", "Force Quit"),
        ("left", "rollout_prev", "Prev iter"),
        ("right", "rollout_next", "Next iter"),
        ("end", "rollout_current", "Current iter"),
        ("p", "toggle_pause", "Pause after rollout"),
        ("r", "resume_pause", "Resume"),
    ]

    def __init__(
        self,
        train_fn: Callable[[], None] | None,
        total_iterations: int = 0,
    ) -> None:
        super().__init__()
        self._train_fn = train_fn
        self._total = total_iterations
        self.bridge = MetricsBridge(self, total_iterations=total_iterations)

    def compose(self) -> ComposeResult:
        yield ProgressHeader(id="top-bar")
        yield StepProgress(id="step-bar")
        with Horizontal(id="main-area"):
            with Vertical(id="left-col"):
                yield RolloutPanel(id="rollout-pane")
                yield PausePanel(id="pause-pane")
            with Vertical(id="right-col"):
                yield MetricsTable(id="metrics-pane")
                with Horizontal(classes="chart-row"):
                    yield MetricChart("Loss", ["loss"], id="loss-chart", colors=["red+"])
                    yield MetricChart("Reward/Type", id="reward-chart", chart_type="bar")
                with Horizontal(classes="chart-row"):
                    yield MetricChart("KL", ["KL"], id="kl-chart", colors=["yellow+"])
                    yield MetricChart("Advantages", id="adv-chart", chart_type="box")
                with Horizontal(classes="chart-row"):
                    yield MetricChart("LR", ["lr"], id="lr-chart", colors=["blue+"])
                    yield MetricChart("Grad Norm", ["grad"], id="grad-chart", colors=["magenta+"])
        yield LogPanel(id="log-pane")
        yield SystemBar(id="system-bar")

    def on_mount(self) -> None:
        if self._train_fn:
            self.run_worker(self._train_fn, thread=True, name="training")
        self.query_one(LogPanel).log_event("[tui] Training started")
        self.set_interval(2.0, self._poll_system)

    def _poll_system(self) -> None:
        self.query_one(SystemBar).poll()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "training":
            return
        panel = self.query_one(LogPanel)
        if event.state == WorkerState.SUCCESS:
            panel.log_event("[tui] Training complete")
        elif event.state == WorkerState.ERROR:
            panel.log_event(f"[tui] Training error: {event.worker.error}")
        elif event.state == WorkerState.CANCELLED:
            panel.log_event("[tui] Training cancelled")

    def on_metrics_updated(self, message: MetricsUpdated) -> None:
        snap = message.snapshot
        step = snap.iteration + 1
        self.query_one(ProgressHeader).refresh_from(snap)
        self.query_one(MetricsTable).refresh_from(snap)
        self._update_charts(step, snap)

    def on_phase_event(self, message: PhaseEvent) -> None:
        sp = self.query_one(StepProgress)
        if message.phase == "gradient":
            sp.set_gradient()
        else:
            sp.set_rollout(message.group, message.total, message.label)

    def on_rollout_event(self, message: RolloutEvent) -> None:
        self.query_one(RolloutPanel).show_rollouts(message.entries, message.iteration)

    def on_stream_start_event(self, message: StreamStartEvent) -> None:
        self.query_one(RolloutPanel).stream_start(
            message.prompt,
            message.type_name,
            message.group,
            message.total,
            member_idx=message.member_idx,
            chunks=message.chunks,
            is_naked=message.is_naked,
        )

    def action_rollout_prev(self) -> None:
        self.query_one(RolloutPanel).nav_prev()

    def action_rollout_next(self) -> None:
        self.query_one(RolloutPanel).nav_next()

    def action_rollout_current(self) -> None:
        self.query_one(RolloutPanel).nav_current()

    def on_stream_token_event(self, message: StreamTokenEvent) -> None:
        self.query_one(RolloutPanel).stream_token(message.text)

    def on_stream_segment_event(self, message: StreamSegmentEvent) -> None:
        """Scaffold injection → RED block before following generation."""
        self.query_one(RolloutPanel).stream_segment(
            message.member_idx, message.source, message.text, message.tag
        )

    def _update_charts(self, step: int, snap: MetricsSnapshot) -> None:
        loss_c: MetricChart = self.query_one("#loss-chart", MetricChart)
        loss_c.push_scalar(step, "loss", snap.loss)
        loss_c.render_chart()

        # Bar chart: per-type mean rewards
        reward_c: MetricChart = self.query_one("#reward-chart", MetricChart)
        if snap.per_type_rewards:
            reward_c.push_dict(step, snap.per_type_rewards)
        else:
            reward_c.push_scalar(step, "reward", snap.mean_reward)
        reward_c.render_chart()

        kl_c: MetricChart = self.query_one("#kl-chart", MetricChart)
        kl_c.push_scalar(step, "KL", snap.kl_divergence)
        kl_c.render_chart()

        # Box chart: advantage distribution per type
        adv_c: MetricChart = self.query_one("#adv-chart", MetricChart)
        if snap.per_type_reward_lists:
            # Show advantage distribution grouped by type
            for tn, rews in snap.per_type_reward_lists.items():
                adv_c.push_box_data(tn, rews)
        elif snap.raw_advantages:
            adv_c.push_box_data("all", snap.raw_advantages)
        adv_c.render_chart()

        lr_c: MetricChart = self.query_one("#lr-chart", MetricChart)
        lr_c.push_scalar(step, "lr", snap.learning_rate)
        lr_c.render_chart()

        grad_c: MetricChart = self.query_one("#grad-chart", MetricChart)
        grad_c.push_scalar(step, "grad", snap.grad_norm)
        grad_c.render_chart()

    def on_log_event(self, message: LogEvent) -> None:
        self.query_one(LogPanel).log_event(message.text)

    def on_pause_status_event(self, msg: PauseStatusEvent) -> None:
        raw_segs = self.query_one(RolloutPanel).get_all_raw_segs()
        self.query_one(PausePanel).set_paused(msg.is_paused, raw_segs)

    def action_toggle_pause(self) -> None:
        self.bridge.pause_after_rollout = not self.bridge.pause_after_rollout
        s = "ON" if self.bridge.pause_after_rollout else "OFF"
        self.query_one(LogPanel).log_event(f"[tui] Pause-after-rollout: {s}")

    def action_resume_pause(self) -> None:
        self.bridge._resume_event.set()

    def on_pause_panel_resume_action(self, _msg: PausePanel.ResumeAction) -> None:
        self.bridge._resume_event.set()

    def on_pause_panel_rerun_action(self, msg: PausePanel.RerunAction) -> None:
        from mlx_grpo.tui.bridge import EditRequest  # noqa: PLC0415

        self.bridge.edit_queue.put(EditRequest(msg.member_idx, msg.scaffold_segs))
        self.bridge._resume_event.set()

    def action_stop_training(self) -> None:
        """q / ctrl+q / ctrl+c — signal trainer to stop, then exit TUI."""
        self.bridge.stop_event.set()
        with contextlib.suppress(Exception):
            self.query_one(LogPanel).log_event("[tui] Stopping training...")
        self.exit()

    def action_quit(self) -> None:
        """Override Textual built-in quit so stop_event is always set."""
        self.action_stop_training()


def run_tui(
    train_fn: Callable[[], None],
    total_iterations: int = 0,
) -> None:
    """Public entry point — launch TUI with training in a worker thread."""
    app = TrainingApp(train_fn, total_iterations=total_iterations)
    try:
        app.run()
    except KeyboardInterrupt:
        app.bridge.stop_event.set()
