"""RolloutPanel — tabbed rollout viewer.

Live stream: scaffold injections → RED, generated text → GREEN (type color).
Each group member gets its own scrollable tab; previous steps browsable via ◀/▶.
Completed view reuses the same red/green scheme with reward header.
"""

from __future__ import annotations

import contextlib

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, RichLog, Tab, Tabs

_TYPE_COL: dict[str, str] = {
    "math": "green",
    "exam": "cyan",
    "mcq": "blue",
    "general_qna": "magenta",
    "python_code": "yellow",
    "tool_call": "white",
}


def _esc(t: str) -> str:
    return t.replace("[", "\\[").replace("]", "\\]")


def _seg_markup(source: str, text: str, tag: str, col: str) -> str:
    """One segment → Rich markup. scaffold=red, answer_prefix=bright_green, gen=col."""
    t = _esc(text.strip())
    if not t:
        return ""
    if source == "scaffold":
        return f"[bold red]◀ {tag} ▶[/bold red]\n[red]{t}[/red]"
    if source == "answer_prefix":
        return (
            f"[bold bright_green]── answer ──[/bold bright_green]\n[bright_green]{t}[/bright_green]"
        )
    return f"[{col}]{t}[/{col}]"


def _render_entry(entry: dict) -> str:  # noqa: C901
    """Completed member → full color markup. scaffold=red, gen=col, answer=bright_green."""
    tn = entry.get("type", "?")
    col = _TYPE_COL.get(tn, "white")
    naked = entry.get("is_naked", False)
    label = "naked" if naked else f"m{entry.get('member_idx', 0)}·{entry.get('chunks', 0)}c"
    # Member role indicator: forced_answer, escalation, naked_first, naked_last, etc.
    member_role = entry.get("_member_role", "")
    role_tag = ""
    if member_role == "forced_answer":
        role_tag = " [bold yellow]⚡FORCED[/bold yellow]"
    elif member_role == "naked_followup":
        role_tag = " [dim]🔄follow[/dim]"
    try:
        rv = float(entry.get("reward", 0) or 0)
        ri = "✓" if rv >= 1.0 else "~" if rv > 0 else "✗"
        rc = "green" if rv >= 1.0 else "yellow" if rv > 0 else "red"
    except (ValueError, TypeError):
        ri, rc = "?", "dim"
    reward_str = entry.get("reward", "?")
    header = (
        f"[bold {col}]▸ [{tn}] {label}{role_tag}[/bold {col}]  "
        f"[{rc}]reward={reward_str} {ri}[/{rc}]"
    )
    parts = [
        header,
        f"[{col}]{'─' * 40}[/{col}]",
    ]
    for seg in entry.get("segments", []):
        src = seg.get("source", "")
        tag_s = seg.get("tag", "")
        tok = seg.get("tokens", 0)
        if src == "INJECTED":
            source = "scaffold"
        elif tag_s == "answer":
            source = "answer_prefix"
        else:
            source = "gen"
        tag_disp = f"{tag_s} [{tok}t]" if tok else tag_s
        m = _seg_markup(source, seg.get("text", ""), tag_disp, col)
        if m:
            parts.append(m)
    return "\n".join(parts)


class RolloutPanel(Vertical):
    """Left panel: member Tabs + RichLog live stream + ◀/▶ history navigation."""

    DEFAULT_CSS = """
    RolloutPanel { height: 100%; border: solid $success; }
    RolloutPanel Tabs { height: 2; }
    #nav-bar { height: 1; layout: horizontal; }
    #nav-label { width: 1fr; height: 1; text-align: center; color: $text-muted; }
    #nav-prev, #nav-next, #nav-cur { min-width: 6; height: 1; border: none; padding: 0 1; }
    #stream-scroll { height: 1fr; }
    RichLog { height: 1fr; padding: 0 1; }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._history: list[list[dict]] = []
        self._cursor: int = 0
        self._active_tab: int = 0
        self._tab_indices: list[int] = []
        self._tab_headers: dict[int, str] = {}  # idx → header markup string
        self._display_segs: dict[int, list[list]] = {}  # idx → [[source, text, tag]]
        self._live_sealed: dict[int, bool] = {}  # idx → seal next token as new block
        self._live_type: dict[int, str] = {}  # idx → type_name
        self._completed: dict[int, str] = {}  # idx → rendered completed markup
        self._current_iter: int = 0

    def compose(self) -> ComposeResult:
        yield Tabs(id="member-tabs")
        with VerticalScroll(id="stream-scroll"):
            yield RichLog(highlight=True, markup=True, id="stream-log")
        with Horizontal(id="nav-bar"):
            yield Button("◀", id="nav-prev")
            yield Label("live  [dim]◀/▶ browse  ↓cur[/dim]", id="nav-label")
            yield Button("▶", id="nav-next")
            yield Button("↓ cur", id="nav-cur")

    def stream_start(
        self,
        prompt: str,
        type_name: str,
        group: int,
        total: int,
        member_idx: int = 0,
        chunks: int = 0,
        is_naked: bool = False,
    ) -> None:
        col = _TYPE_COL.get(type_name, "white")
        label = "naked" if is_naked else f"m{member_idx}·{chunks}c"
        self._live_type[member_idx] = type_name
        self._display_segs[member_idx] = []
        self._live_sealed[member_idx] = False
        hdr = (
            f"[bold {col}]▸ [{type_name}] {label} — g{group}/{total}[/bold {col}]\n"
            f"[dim]{_esc(prompt[:120])}…[/dim]\n[{col}]{'─' * 40}[/{col}]"
        )
        self._tab_headers[member_idx] = hdr
        self._ensure_tab(member_idx, label)
        if self._cursor == 0:
            self._activate(member_idx)

    def stream_segment(self, member_idx: int, source: str, text: str, tag: str = "") -> None:
        """Scaffold inject (RED) or answer marker — seal current gen block, add new."""
        if self._cursor != 0:
            return
        segs = self._display_segs.setdefault(member_idx, [])
        segs.append([source, text, tag])
        self._live_sealed[member_idx] = True  # next stream_token starts a fresh green block
        if self._active_tab == member_idx:
            self._redraw_live(member_idx)

    def stream_token(self, text: str) -> None:
        """Append generated token delta to current green block (or start one if sealed)."""
        if self._cursor != 0:
            return
        idx = self._active_tab
        segs = self._display_segs.setdefault(idx, [])
        sealed = self._live_sealed.get(idx, False)
        if not segs or segs[-1][0] not in ("gen", "generated") or sealed:
            segs.append(["gen", text, "thinking"])
            self._live_sealed[idx] = False
        else:
            segs[-1][1] += text
        if self._active_tab == idx:
            self._redraw_live(idx)

    def _redraw_live(self, member_idx: int) -> None:
        col = _TYPE_COL.get(self._live_type.get(member_idx, "?"), "white")
        log = self.query_one("#stream-log", RichLog)
        log.clear()
        log.write(self._tab_headers.get(member_idx, ""))
        for seg in self._display_segs.get(member_idx, []):
            src, txt, tag = seg
            m = _seg_markup(src, txt, tag, col)
            if m:
                log.write(m)

    def show_rollouts(self, entries: list[dict], iteration: int = 0) -> None:
        self._current_iter = iteration
        for e in entries:
            midx = e.get("member_idx", 0)
            lbl = "naked" if e.get("is_naked") else f"m{midx}·{e.get('chunks', 0)}c"
            self._ensure_tab(midx, lbl)
        if entries:
            self._history.insert(0, entries)
            if len(self._history) > 500:
                self._history.pop()
        if self._cursor == 0:
            self._completed = {e.get("member_idx", 0): _render_entry(e) for e in entries}
            self._redraw_completed()
        self._update_nav()

    def nav_prev(self) -> None:
        if self._cursor < len(self._history) - 1:
            self._cursor += 1
            self._load(self._cursor)

    def nav_next(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1
            self._load(self._cursor)

    def nav_current(self) -> None:
        self._cursor = 0
        if self._history:
            self._load(0)
        self._update_nav()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        tab_id = (event.tab.id or "") if event.tab else ""
        if tab_id.startswith("tab-"):
            self._active_tab = int(tab_id[4:])
            if self._cursor == 0 and self._active_tab in self._completed:
                self._redraw_completed()
            elif self._cursor == 0:
                self._redraw_live(self._active_tab)
            else:
                self._redraw_completed()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        _m = {"nav-prev": self.nav_prev, "nav-next": self.nav_next, "nav-cur": self.nav_current}
        if fn := _m.get(event.button.id or ""):
            fn()

    def _ensure_tab(self, idx: int, label: str) -> None:
        tabs = self.query_one(Tabs)
        if idx not in self._tab_indices:
            self._tab_indices.append(idx)
            self._tab_indices.sort()
            tabs.add_tab(Tab(label, id=f"tab-{idx}"))
        else:
            # Update label on existing tab (e.g. new step starts with same member_idx)
            with contextlib.suppress(Exception):
                tabs.query_one(f"#tab-{idx}", Tab).label = label  # type: ignore[assignment]

    def _activate(self, idx: int) -> None:
        self._active_tab = idx
        with contextlib.suppress(Exception):
            self.query_one(Tabs).active = f"tab-{idx}"

    def _redraw_completed(self) -> None:
        log = self.query_one("#stream-log", RichLog)
        log.clear()
        content = self._completed.get(self._active_tab, "No content yet.")
        log.write(content)

    def _load(self, cursor: int) -> None:
        if not self._history or cursor >= len(self._history):
            return
        self._completed = {e.get("member_idx", 0): _render_entry(e) for e in self._history[cursor]}
        target = self._active_tab if self._active_tab in self._completed else min(self._completed)
        self._activate(target)
        self._redraw_completed()
        self._update_nav()

    def get_all_raw_segs(self) -> dict[int, list]:
        """Return {member_idx: [[source, text, tag], ...]} for the current live step."""
        return dict(self._display_segs)

    def _update_nav(self) -> None:
        n = len(self._history)
        if self._cursor == 0:
            txt = (
                f"iter {self._current_iter} ▸ live  [dim]{n} steps[/dim]"
                if n
                else "live [dim]waiting…[/dim]"
            )
        else:
            pos = int(self._cursor / max(n - 1, 1) * 16)
            bar = "◼" * pos + "◻" * (16 - pos)
            txt = (
                f"[dim]◀[/dim] {bar} {self._cursor}/{n - 1}"
                f"  [dim]iter {self._current_iter - self._cursor}[/dim]"
            )
        with contextlib.suppress(Exception):
            self.query_one("#nav-label", Label).update(txt)
