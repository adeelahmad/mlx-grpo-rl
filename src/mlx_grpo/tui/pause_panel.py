"""PausePanel — interactive scaffold edit panel shown when training is paused."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Label, Select, Static, TextArea


class PausePanel(Vertical):
    """Panel that appears when training is paused, allowing scaffold editing and rerun."""

    DEFAULT_CSS = """
    PausePanel { display: none; border: solid $warning; height: 12; padding: 0 1; }
    PausePanel Static { color: $warning; height: 1; }
    PausePanel #pause-controls { height: 1; layout: horizontal; margin-top: 0; }
    PausePanel #pause-lbl { width: 10; height: 1; }
    PausePanel Select { width: 24; height: 1; }
    PausePanel #scaffold-edit { height: 6; margin-top: 0; }
    PausePanel #pause-btns { height: 1; layout: horizontal; margin-top: 0; }
    PausePanel #btn-resume { margin-right: 2; }
    """

    class ResumeAction(Message):
        """User clicked Resume — unblock training thread."""

    class RerunAction(Message):
        """User clicked Rerun — re-execute member with edited scaffold."""

        def __init__(self, member_idx: int, scaffold_segs: list[str]) -> None:
            super().__init__()
            self.member_idx = member_idx
            self.scaffold_segs = scaffold_segs

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._raw_segs: dict[int, list] = {}

    def compose(self) -> ComposeResult:
        yield Static("⏸ PAUSED — Edit scaffold then Rerun, or Resume to continue")
        with Horizontal(id="pause-controls"):
            yield Label("Member: ", id="pause-lbl")
            yield Select([], id="member-select", allow_blank=True)
        yield TextArea(id="scaffold-edit")
        with Horizontal(id="pause-btns"):
            yield Button("▶ Resume", id="btn-resume", variant="success")
            yield Button("⟳ Rerun Member", id="btn-rerun", variant="primary")

    def set_paused(self, is_paused: bool, raw_segs: dict[int, list]) -> None:
        """Show/hide panel and populate TextArea with scaffold segments."""
        self.display = is_paused
        if not is_paused:
            return
        self._raw_segs = raw_segs
        sel = self.query_one("#member-select", Select)
        options = [(f"Member {k}", k) for k in sorted(raw_segs)]
        sel.set_options(options)  # type: ignore[arg-type]
        if raw_segs:
            first_key = min(raw_segs)
            sel.value = first_key  # type: ignore[assignment]
            self._fill_textarea(first_key)

    def _fill_textarea(self, member_idx: int) -> None:
        segs = self._raw_segs.get(member_idx, [])
        scaffold_texts = [s[1] for s in segs if s[0] == "scaffold"]
        ta = self.query_one("#scaffold-edit", TextArea)
        ta.load_text("\n\n".join(scaffold_texts))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "member-select":
            val = event.value
            if val is not None and val is not Select.BLANK:
                self._fill_textarea(int(val))  # type: ignore[arg-type]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-resume":
            self.post_message(self.ResumeAction())
        elif event.button.id == "btn-rerun":
            sel = self.query_one("#member-select", Select)
            ta = self.query_one("#scaffold-edit", TextArea)
            val = sel.value
            member_idx = int(val) if (val is not None and val is not Select.BLANK) else 0  # type: ignore[arg-type]
            scaffold_segs = [s for s in ta.text.split("\n\n") if s.strip()]
            self.post_message(self.RerunAction(member_idx, scaffold_segs))
