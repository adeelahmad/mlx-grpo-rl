"""Per-type scaffold configuration — token budget, position, and decay control.

Decouples scaffold settings from TrainingConfig so each task type can
override scaffold behaviour independently via YAML or Python config.

    sc = ScaffoldConfig(max_tokens=80, mode="prefix", target="thinking")
    ratio = sc.effective_ratio(progress=0.3)
    budget = sc.effective_tokens(progress=0.3, answer_token_count=200)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ScaffoldConfig:
    """Frozen scaffold settings — can live on BatchContext or per-type config."""

    max_tokens: int | None = None  # Legacy token budget (None = use ratio)
    mode: str = "prefix"  # prefix | suffix | middle | by_lines
    target: str = "both"  # thinking | answer | both
    initial_ratio: float = 1.0
    decay_rate: float = 1.0
    end_progress: float = 0.8
    min_ratio: float = 0.05
    levels: tuple[float, ...] | None = None  # Per-group scaffold multipliers
    max_scaffolded_tokens: int | None = None  # Token limit ratio applies to
    dynamic_max_tokens: bool = False  # Equalize total tokens across scaffold groups
    per_group_config: PerGroupConfig | None = None  # Multi-view contrastive scaffold

    def effective_ratio(self, progress: float, group_index: int = 0) -> float:
        """Current scaffold ratio given training progress (0→1).

        When *levels* is set, the base ratio is scaled by the level for
        the rollout's position within the group, creating a curriculum
        gradient across group members.
        """
        if self.end_progress <= 0 or progress >= self.end_progress:
            return 0.0
        base = self.initial_ratio * max(
            0.0,
            1.0 - (progress / self.end_progress) * self.decay_rate,
        )
        if self.levels is not None:
            return base * self.levels[group_index % len(self.levels)]
        return base

    def effective_tokens(
        self,
        progress: float,
        answer_token_count: int,
        group_index: int = 0,
    ) -> int:
        """Token budget at current progress.

        Priority: *max_scaffolded_tokens* → *max_tokens* → *answer_token_count*.
        When *max_scaffolded_tokens* is set the ratio operates on that limit,
        keeping scaffold cost bounded regardless of target text length.
        """
        ratio = self.effective_ratio(progress, group_index)
        if self.max_scaffolded_tokens is not None:
            return max(0, int(self.max_scaffolded_tokens * ratio))
        if self.max_tokens is not None:
            return max(0, int(self.max_tokens * ratio))
        return max(0, int(answer_token_count * ratio))

    def dynamic_budget(
        self, progress: float, base_max: int, answer_token_count: int, group_index: int
    ) -> int:
        """Adjusted max_completion_tokens — extra generation room for less-scaffolded members."""
        if not self.dynamic_max_tokens or self.max_scaffolded_tokens is None:
            return base_max
        ceiling = self.max_scaffolded_tokens
        actual = self.effective_tokens(progress, answer_token_count, group_index)
        return base_max + (ceiling - actual)

    def dynamic_used(
        self, progress: float, base_used: int, answer_token_count: int, group_index: int
    ) -> int:
        """Adjusted used_tokens — proportional to dynamic_budget."""
        if not self.dynamic_max_tokens or self.max_scaffolded_tokens is None:
            return base_used
        ceiling = self.max_scaffolded_tokens
        actual = self.effective_tokens(progress, answer_token_count, group_index)
        return base_used + (ceiling - actual)

    @property
    def thinking_ratio_factor(self) -> float:
        """Multiplier: 0.0 when target is answer-only, 1.0 otherwise."""
        return 0.0 if self.target == "answer" else 1.0

    @property
    def answer_ratio_factor(self) -> float:
        """Multiplier: 0.0 when target is thinking-only, 1.0 otherwise."""
        return 0.0 if self.target == "thinking" else 1.0


@dataclass(slots=True, frozen=True)
class ViewConfig:
    """Per-member view for multi-view contrastive scaffolding."""

    name: str
    scaffold_position: str  # "prefix", "middle", "postfix", "none"
    truncate_from: str  # "prefix", "middle", "postfix"
    used_tokens: int
    masked: bool = True
    decay_rate: float = 1.0
    end_progress: float = 0.3
    prefix_lines: int | None = None  # Inject first N lines of thinking before main scaffold


@dataclass(slots=True, frozen=True)
class PerGroupConfig:
    """Multi-view scaffold: each group member gets a different view."""

    target: str  # "thinking", "answer", "both"
    views: tuple[ViewConfig, ...]


def get_view_scaffold_ratio(view: ViewConfig, progress: float) -> float:
    """Ratio for a single view at training progress (0→1).

    Decays from 1.0 → 0.0 using power law over end_progress.
    Returns 0.0 immediately for scaffold_position="none".
    """
    if view.scaffold_position == "none" or view.end_progress <= 0:
        return 0.0
    if progress >= view.end_progress:
        return 0.0
    return max(0.0, (1.0 - progress / view.end_progress) ** view.decay_rate)


def build_scaffold_config(
    data: dict[str, Any] | None,
    defaults: ScaffoldConfig | None = None,
) -> ScaffoldConfig:
    """Build a ScaffoldConfig from a dict, merging with *defaults*.

    Used for per-type overrides: the global config is *defaults*, and
    the type's ``scaffold:`` sub-dict is *data*.
    """
    if data is None and defaults is not None:
        return defaults
    base = defaults or ScaffoldConfig()
    if not data:
        return base
    levels_raw = data.get("levels")
    levels = tuple(float(x) for x in levels_raw) if levels_raw else base.levels
    mst = data.get("max_scaffolded_tokens")
    max_scaffolded = int(mst) if mst is not None else base.max_scaffolded_tokens
    dmt = data.get("dynamic_max_tokens")
    dynamic = bool(dmt) if dmt is not None else base.dynamic_max_tokens
    pgc_raw = data.get("per_group_config") if data else None
    pgc: PerGroupConfig | None = None
    if pgc_raw and isinstance(pgc_raw, dict):
        views = tuple(
            ViewConfig(
                name=str(v["name"]),
                scaffold_position=str(v.get("scaffold_position", "prefix")),
                truncate_from=str(v.get("truncate_from", "prefix")),
                used_tokens=int(v.get("used_tokens", 64)),
                masked=bool(v.get("masked", True)),
                decay_rate=float(v.get("decay_rate", 1.0)),
                end_progress=float(v.get("end_progress", 0.3)),
                prefix_lines=int(v["prefix_lines"]) if v.get("prefix_lines") else None,
            )
            for v in pgc_raw.get("views", [])
        )
        pgc = PerGroupConfig(target=str(pgc_raw.get("target", "thinking")), views=views)
    return ScaffoldConfig(
        max_tokens=data.get("max_tokens", base.max_tokens),
        mode=data.get("mode", base.mode),
        target=data.get("target", base.target),
        initial_ratio=float(data.get("initial_ratio", base.initial_ratio)),
        decay_rate=float(data.get("decay_rate", base.decay_rate)),
        end_progress=float(data.get("end_progress", base.end_progress)),
        min_ratio=float(data.get("min_ratio", base.min_ratio)),
        levels=levels,
        max_scaffolded_tokens=max_scaffolded,
        dynamic_max_tokens=dynamic,
        per_group_config=pgc,
    )
