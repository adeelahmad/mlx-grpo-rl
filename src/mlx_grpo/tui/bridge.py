"""Metrics bridge — adapter between Trainer hooks and TUI messages."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from textual.message import Message

if TYPE_CHECKING:
    from textual.app import App

    from mlx_grpo.training import StepMetrics, Trainer


@dataclass(slots=True)
class MetricsSnapshot:
    """Flat subset of StepMetrics for TUI consumption."""

    iteration: int = 0
    total_iterations: int = 0
    loss: float = 0.0
    mean_reward: float = 0.0
    reward_std: float = 0.0
    kl_divergence: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    tokens_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    rss_mb: float = 0.0
    elapsed_sec: float = 0.0
    generated_tokens: int = 0
    trained_tokens: int = 0
    mean_advantage: float = 0.0
    raw_advantages: list[float] = field(default_factory=list)
    per_type_reward_lists: dict[str, list[float]] = field(default_factory=dict)
    type_names: list[str] = field(default_factory=list)
    per_type_rewards: dict[str, float] = field(default_factory=dict)
    per_type_correct: dict[str, int] = field(default_factory=dict)
    per_type_count: dict[str, int] = field(default_factory=dict)
    # Escalation member breakdown: (member_idx, n_chunks, reward)
    esc_members: list[tuple[int, int, float]] = field(default_factory=list)


class MetricsUpdated(Message):
    """Posted when a training step completes."""

    def __init__(self, snapshot: MetricsSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot


class PhaseEvent(Message):
    def __init__(self, phase: str, group: int = 0, total: int = 0, label: str = "") -> None:
        super().__init__()
        self.phase, self.group, self.total, self.label = phase, group, total, label


class RolloutEvent(Message):
    def __init__(self, entries: list[dict[str, str]], iteration: int = 0) -> None:
        super().__init__()
        self.entries = entries
        self.iteration = iteration


class StreamStartEvent(Message):
    def __init__(
        self,
        prompt: str,
        type_name: str,
        group: int,
        total: int,
        member_idx: int = 0,
        chunks: int = 0,
        is_naked: bool = False,
    ) -> None:
        super().__init__()
        self.prompt, self.type_name = prompt, type_name
        self.group, self.total = group, total
        self.member_idx, self.chunks, self.is_naked = member_idx, chunks, is_naked


class StreamTokenEvent(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StreamSegmentEvent(Message):
    """Scaffold chunk injected into context — display RED before the following generation."""

    def __init__(self, member_idx: int, source: str, text: str, tag: str = "") -> None:
        super().__init__()
        self.member_idx = member_idx
        self.source = source  # "scaffold" | "answer_prefix"
        self.text = text
        self.tag = tag


class LogEvent(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PauseStatusEvent(Message):
    """Posted when training enters or leaves the paused state."""

    def __init__(self, is_paused: bool) -> None:
        super().__init__()
        self.is_paused = is_paused


@dataclass
class EditRequest:
    """Scaffold edit from the TUI — re-run a specific member with custom scaffold."""

    member_idx: int
    scaffold_segs: list[str]


class MetricsBridge:
    """Hook adapter — captures trainer metrics and posts TUI messages."""

    def __init__(self, app: App[Any], total_iterations: int = 0) -> None:
        self._app = app
        self._total = total_iterations
        self.history: list[MetricsSnapshot] = []
        self.cumulative_tokens: int = 0
        self.stop_event = threading.Event()
        self.pause_after_rollout: bool = False
        self._resume_event: threading.Event = threading.Event()
        self._resume_event.set()  # initially unblocked
        self.edit_queue: queue.SimpleQueue[EditRequest] = queue.SimpleQueue()

    def step_hook(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Trainer after_step hook — build snapshot and post to TUI."""
        type_stats = metrics.extras.get("type_stats", {})
        advantages = metrics.extras.get("advantages", [])
        trained = metrics.extras.get("trained_tokens", metrics.generated_tokens)
        self.cumulative_tokens += trained

        per_type_rewards: dict[str, float] = {}
        per_type_correct: dict[str, int] = {}
        per_type_count: dict[str, int] = {}
        per_type_reward_lists: dict[str, list[float]] = {}
        for tn, ts in type_stats.items():
            rews = ts["rewards"]
            per_type_rewards[tn] = sum(rews) / len(rews) if rews else 0.0
            per_type_correct[tn] = ts["correct"]
            per_type_count[tn] = ts["count"]
            per_type_reward_lists[tn] = list(rews)

        mean_adv = sum(advantages) / len(advantages) if advantages else 0.0
        rollouts_raw = metrics.extras.get("rollouts", [])
        rew_list = metrics.rewards or []
        esc_members: list[tuple[int, int, float]] = []
        for i, r in enumerate(rollouts_raw):
            midx = r.metadata.get("_esc_member_idx", -1)
            if midx >= 0:
                chunks = int(r.metadata.get("_esc_allocated_chunks", 0))
                rew = float(rew_list[i]) if i < len(rew_list) else 0.0
                esc_members.append((midx, chunks, rew))
        snap = MetricsSnapshot(
            iteration=metrics.iteration,
            total_iterations=self._total or trainer.config.iterations,
            loss=metrics.loss,
            mean_reward=metrics.mean_reward,
            reward_std=metrics.reward_std,
            kl_divergence=metrics.kl_divergence,
            grad_norm=metrics.grad_norm,
            learning_rate=metrics.learning_rate,
            tokens_per_sec=metrics.tokens_per_sec,
            peak_memory_mb=metrics.peak_memory_mb,
            rss_mb=metrics.rss_mb,
            elapsed_sec=metrics.elapsed_sec,
            generated_tokens=metrics.generated_tokens,
            trained_tokens=self.cumulative_tokens,
            mean_advantage=mean_adv,
            raw_advantages=list(advantages),
            per_type_reward_lists=per_type_reward_lists,
            type_names=metrics.extras.get("type_names", []),
            per_type_rewards=per_type_rewards,
            per_type_correct=per_type_correct,
            per_type_count=per_type_count,
            esc_members=esc_members,
        )
        self.history.append(snap)
        self._app.post_message(MetricsUpdated(snap))
        self._post_rollouts(metrics)

    def group_hook(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Trainer on_group hook — intra-step rollout progress."""
        extras = metrics.extras
        self._app.post_message(
            PhaseEvent(
                phase=extras.get("phase", "rollout"),
                group=extras.get("group_index", 0),
                total=extras.get("group_size", 0),
                label=extras.get("type_name", ""),
            )
        )

    def gradient_hook(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Trainer on_gradient hook — gradient phase start."""
        self._app.post_message(PhaseEvent(phase="gradient"))

    def _post_rollouts(self, metrics: StepMetrics) -> None:
        rollouts = metrics.extras.get("rollouts", [])
        prompts = metrics.extras.get("prompts", [])
        rewards = metrics.rewards or []
        if not rollouts:
            return
        from mlx_grpo.engine.result import SegmentSource  # noqa: PLC0415

        entries: list[dict[str, Any]] = []
        for i, r in enumerate(rollouts):
            prompt = prompts[i] if i < len(prompts) else ""
            reward = rewards[i] if i < len(rewards) else 0.0
            segs: list[dict[str, Any]] = []
            for seg in getattr(r, "segments", []):
                if seg.source == SegmentSource.PROMPT:
                    continue
                segs.append(
                    {
                        "source": seg.source.name,
                        "tag": seg.tag or "",
                        "text": seg.text[:200],
                        "tokens": len(seg.tokens),
                        "masked": bool(seg.masked),
                    }
                )
            entries.append(
                {
                    "type": r.type_name or "?",
                    "prompt": prompt[:200],
                    "generation": r.completion_text if hasattr(r, "completion_text") else "",
                    "reward": f"{reward:.3f}",
                    "member_idx": r.metadata.get("_esc_member_idx", i),
                    "chunks": r.metadata.get("_esc_allocated_chunks", 0),
                    "is_naked": bool(r.metadata.get("_view_is_naked", False)),
                    "view_name": r.metadata.get("_view_name", f"view_{i}"),
                    "segments": segs,
                    "_member_role": r.metadata.get("_member_role", ""),
                    "_esc_forced_answer": bool(r.metadata.get("_esc_forced_answer", False)),
                }
            )
        self._app.post_message(RolloutEvent(entries, iteration=metrics.iteration))

    def rollout_start_hook(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Trainer on_rollout_start hook — new generation starting."""
        e = metrics.extras
        self._app.post_message(
            StreamStartEvent(
                e.get("prompt", ""),
                e.get("type_name", ""),
                e.get("group_index", 0),
                e.get("group_size", 0),
                member_idx=e.get("member_idx", 0),
                chunks=e.get("allocated_chunks", 0),
                is_naked=bool(e.get("is_naked", False)),
            )
        )

    def make_stream_cb(self) -> Any:
        """Create a streaming callback for token-by-token TUI updates."""
        app = self._app

        def _cb(delta: str) -> None:
            app.post_message(StreamTokenEvent(delta))

        return _cb

    def make_segment_cb(self, member_idx: int) -> Any:
        """Create a scaffold segment callback for _vertical_rollout.

        Called by run_escalation() per member so scaffold injections appear
        as RED blocks in the live stream before the following green generation.
        """
        app = self._app

        def _cb(source: str, text: str, tag: str = "") -> None:
            if text:
                app.post_message(StreamSegmentEvent(member_idx, source, text, tag))

        return _cb

    def pause_hook(self, trainer: Any, metrics: Any) -> None:
        """after_rollout hook — cooperative pause before gradient step."""
        if not self.pause_after_rollout:
            return
        self._resume_event.clear()
        self._app.call_from_thread(self._app.post_message, PauseStatusEvent(True))
        self._resume_event.wait()
        self._app.call_from_thread(self._app.post_message, PauseStatusEvent(False))

    def post_log(self, text: str) -> None:
        import contextlib

        with contextlib.suppress(Exception):  # App already exited — drop the message silently
            self._app.post_message(LogEvent(text))
