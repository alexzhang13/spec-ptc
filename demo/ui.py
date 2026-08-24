"""Shared TUI components for the spec-ptc demos — a Codex-style minimal look.

Widgets: ChatBox (a user query typing itself), StreamView (the main model's
output, code-styled as it streams, with turn dividers and REPL output blocks),
CallRow (one tool call with a `╰─▸` line streaming its partial output), and
Column (head + stream + calls + answer panel for one agent).

`apply_side_event(col, ev)` routes the standard bus events into a Column so
each demo app only handles its own extras (chat, verdict, side_done).

Careful when adding state to App subclasses: private-looking names may shadow
Textual internals — `self._running = False` silently gates App._display and
freezes the screen while everything else keeps running.
"""

from __future__ import annotations

import re
import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

PALETTE = {
    "chrome": "#7d8590",
    "dim": "#6e7681",
    "text": "#e6edf3",
    "prose": "#9198a1",
    "comment": "#6e7681",
    "string": "#a5d6ff",
    "keyword": "#ff7b72",
    "number": "#79c0ff",
    "tool": "#d2a8ff",
    "spec": "#bc8cff",
    "run": "#58a6ff",
    "ok": "#3fb950",
    "warn": "#d29922",
    "err": "#f85149",
}

# CSS every demo app should include (Screen/topbar/verdict stay app-specific)
UI_CSS = """
.chat {
    height: auto; max-height: 5; margin: 0 1; padding: 0 1;
    border: round #30363d; border-title-color: #7d8590;
}
.col { width: 1fr; border: round #30363d; margin: 0 1; padding: 0 1; }
.colhead { height: 1; }
.colname { width: auto; }
.colphase { width: 1fr; content-align: right middle; }
.streamwrap { height: 3fr; margin: 1 0 0 0; }
.callslabel { height: 1; margin: 1 0 0 0; }
.callswrap { height: 2fr; }
.execwrap {
    display: none; height: 3fr; margin: 1 0 0 0;
    padding: 0 1; border: round #58a6ff 50%; border-title-color: #58a6ff;
}
.answerwrap {
    display: none; height: auto; max-height: 5; margin: 1 0 0 0;
    padding: 0 1; border: round #238636; border-title-color: #3fb950;
}
CallRow { height: 2; }
"""

_RE_STRING = re.compile(r"'[^']*'?")
_RE_TOOL = re.compile(r"\b(llm_query_batched|llm_query|solve_dp)\b")
_RE_KEYWORD = re.compile(
    r"\b(def|for|if|elif|else|while|return|in|not|and|or|try|except|finally|"
    r"break|continue|lambda|max|len|str|range|enumerate|zip|next)\b"
)
_RE_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")
_RE_VERDICT_TAG = re.compile(r"\[(OK|WATCH|CONCERN)\]")

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spin(t: float) -> str:
    return _SPINNER[int(t * 10) % len(_SPINNER)]


def style_code_line(line: str) -> Text:
    out = Text()
    body, comment = line, ""
    hash_pos = -1
    in_str = False
    for i, ch in enumerate(line):
        if ch == "'":
            in_str = not in_str
        elif ch == "#" and not in_str:
            hash_pos = i
            break
    if hash_pos >= 0:
        body, comment = line[:hash_pos], line[hash_pos:]

    spans: list[tuple[int, int, str]] = []

    def claim(regex: re.Pattern, style: str) -> None:
        for m in regex.finditer(body):
            if any(not (m.end() <= s or m.start() >= e) for s, e, _ in spans):
                continue
            spans.append((m.start(), m.end(), style))

    claim(_RE_STRING, PALETTE["string"])
    claim(_RE_TOOL, f"bold {PALETTE['tool']}")
    claim(_RE_KEYWORD, PALETTE["keyword"])
    claim(_RE_NUMBER, PALETTE["number"])
    spans.sort()
    pos = 0
    for s, e, style in spans:
        if s > pos:
            out.append(body[pos:s], style=PALETTE["text"])
        out.append(body[s:e], style=style)
        pos = e
    if pos < len(body):
        out.append(body[pos:], style=PALETTE["text"])
    if comment:
        out.append(comment, style=PALETTE["comment"])
    return out


def style_answer(final: str) -> Text:
    """A final report, with [OK]/[WATCH]/[CONCERN] tags colorized."""
    colors = {"OK": PALETTE["ok"], "WATCH": PALETTE["warn"], "CONCERN": PALETTE["err"]}
    t = Text()
    for i, line in enumerate(final.splitlines()):
        if i:
            t.append("\n")
        if i == 0:
            t.append(line, style=f"bold {PALETTE['text']}")
            continue
        pos = 0
        for m in _RE_VERDICT_TAG.finditer(line):
            t.append(line[pos : m.start()], style=PALETTE["prose"])
            t.append(m.group(0), style=f"bold {colors[m.group(1)]}")
            pos = m.end()
        t.append(line[pos:], style=PALETTE["prose"])
    return t


class ChatBox(Static):
    """A user query, typed out as if being entered live, then 'sent'.

    Two modes: auto (types itself at `cps`, for recordings) and manual (starts
    empty; the presenter's keystrokes advance the scripted text a few chars at
    a time, enter completes it — the demo starts when they start typing)."""

    def __init__(self, query: str, on_sent, cps: float = 45.0, manual: bool = False) -> None:
        super().__init__("", classes="chat")
        self.qtext = query
        self.on_sent = on_sent
        self.cps = cps
        self.manual = manual
        self.n = 0.0
        self.sent = False
        self._timer = None

    def on_mount(self) -> None:
        self.border_title = "you"
        if self.manual:
            self._paint(done=False)
        else:
            self._timer = self.set_interval(0.03, self._advance)

    def restart(self) -> None:
        self.n = 0.0
        self.sent = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._paint(done=False)
        if not self.manual:
            self._timer = self.set_interval(0.03, self._advance)

    def keypress(self, key: str) -> None:
        """Manual mode: any keystroke writes the next few scripted chars."""
        if not self.manual or self.sent:
            return
        if key == "enter":
            self.n = len(self.qtext)
        else:
            self.n = min(self.n + 3, len(self.qtext))
        if self.n >= len(self.qtext):
            self._send()
        else:
            self._paint(done=False)

    def _send(self) -> None:
        self.sent = True
        if self._timer is not None:
            self._timer.stop()
        self._paint(done=True)
        self.set_timer(0.5, self.on_sent)

    def _advance(self) -> None:
        if self.sent:
            return
        self.n += self.cps * 0.03
        if self.n >= len(self.qtext):
            self._send()
        else:
            self._paint(done=False)

    def _paint(self, done: bool) -> None:
        t = Text()
        t.append(self.qtext if done else self.qtext[: int(self.n)], style=PALETTE["text"])
        if done:
            t.append("  ⏎ sent", style=PALETTE["dim"])
        else:
            t.append("▌", style=PALETTE["run"])
        self.update(t)


class StreamView(Static):
    """The main model's output, styled incrementally as it streams.
    Supports multiple turns: dividers and REPL output blocks in between."""

    def __init__(self) -> None:
        super().__init__("")
        self._lines: list[Text] = []
        self._cur = ""
        self._pending = ""
        self._in_code = False
        self.streaming = False

    def on_mount(self) -> None:
        # repaint at ~10Hz, not per token: sustained per-event repaints can
        # wedge Textual's render pipeline in a real terminal
        self.set_interval(0.1, self._flush)

    def feed(self, delta: str) -> None:
        self.streaming = True
        self._pending += delta

    def _flush(self) -> None:
        if not self._pending:
            return
        self._cur += self._pending
        self._pending = ""
        while "\n" in self._cur:
            line, self._cur = self._cur.split("\n", 1)
            self._lines.append(self._freeze(line))
        self._repaint()
        parent = self.parent
        if parent is not None and hasattr(parent, "scroll_end"):
            parent.scroll_end(animate=False)

    def finish(self) -> None:
        self._cur += self._pending
        self._pending = ""
        while "\n" in self._cur:
            line, self._cur = self._cur.split("\n", 1)
            self._lines.append(self._freeze(line))
        if self._cur:
            self._lines.append(self._freeze(self._cur))
            self._cur = ""
        self.streaming = False
        self._in_code = False
        self._repaint()

    def add_marker(self, label: str) -> None:
        self._lines.append(
            Text(f"── {label} {'─' * max(0, 30 - len(label))}", style=PALETTE["dim"])
        )
        self._repaint()

    def add_output_block(self, text: str, style: str | None = None) -> None:
        """A REPL stdout block (or any literal block) between turns."""
        for line in text.rstrip().splitlines() or [""]:
            self._lines.append(Text("  " + line, style=style or PALETTE["chrome"]))
        self._repaint()

    def add_line(self, t: Text) -> int:
        self._lines.append(t)
        self._repaint()
        return len(self._lines) - 1

    def update_line(self, idx: int, t: Text) -> None:
        if 0 <= idx < len(self._lines):
            self._lines[idx] = t
            self._repaint()

    def _freeze(self, line: str) -> Text:
        if line.strip().startswith("```"):
            opening = len(line.strip()) > 3
            self._in_code = opening
            label = " python " if opening else "────────"
            return Text(f"──{label}{'─' * 28}", style=PALETTE["dim"])
        if self._in_code:
            return style_code_line(line)
        return Text(line, style=PALETTE["prose"])

    def _repaint(self) -> None:
        t = Text()
        for ln in self._lines:
            t.append_text(ln)
            t.append("\n")
        if self._cur:
            t.append_text(
                style_code_line(self._cur)
                if self._in_code
                else Text(self._cur, style=PALETTE["prose"])
            )
        if self.streaming:
            t.append("▌", style=PALETTE["run"])
        self.update(t)


class CallRow(Static):
    """One tool call: a status line, plus a `╰─▸` line streaming its output."""

    OUT_W = 66  # visible tail of the sub-model's stream

    def __init__(self, side: str, tool: str, preview: str, source: str, n: int = 1):
        super().__init__("")
        self.side, self.tool, self.n = side, tool, n
        self.preview = preview.strip("'\" ")
        self.source = source  # shadow | peek | inline
        self.state = "running" if side == "serial" else "spec"
        self.t_born = time.perf_counter()
        self.ms: float | None = None
        self.saved_s: float | None = None
        self.out = ""
        self.out_dirty = False
        self.rerender()

    def feed_out(self, text: str) -> None:
        self.out += text
        self.out_dirty = True

    def _out_line(self, t: Text) -> None:
        t.append("\n   ╰─▸ ", style=PALETTE["dim"])
        if self.out:
            tail = self.out.replace("\n", " ")[-self.OUT_W :]
            live = self.state in ("spec", "running")
            t.append(
                ("…" if len(self.out) > self.OUT_W else "") + tail,
                style=PALETTE["run"] if live else PALETTE["dim"],
            )
        elif self.tool == "solve_dp":
            t.append("(local compute)", style=PALETTE["dim"])
        else:
            t.append("…", style=PALETTE["dim"])

    def rerender(self) -> None:
        self.out_dirty = False
        now = time.perf_counter()
        t = Text()
        label = self.tool + (f" ×{self.n}" if self.n > 1 else "")
        if self.side == "serial":
            if self.state == "running":
                t.append(f" {spin(now)} ", style=PALETTE["run"])
                t.append(f"{label[:20]:<20} ", style=PALETTE["text"])
                t.append(f"{self.preview[:32]:<34}", style=PALETTE["dim"])
                t.append(f"{now - self.t_born:5.1f}s", style=PALETTE["run"])
            else:
                t.append(" ✓ ", style=PALETTE["ok"])
                t.append(f"{label[:20]:<20} ", style=PALETTE["chrome"])
                t.append(f"{self.preview[:32]:<34}", style=PALETTE["dim"])
                t.append(f"{(self.ms or 0) / 1000:5.1f}s", style=PALETTE["ok"])
        else:
            glyph, style = {
                "spec": (f" {spin(now)} ", PALETTE["spec"]),
                "ready": (" ✓ ", PALETTE["ok"]),
                "claimed": (" ✚ ", f"bold {PALETTE['ok']}"),
                "evicted": (" ✕ ", PALETTE["err"]),
            }[self.state]
            t.append(glyph, style=style)
            t.append(f"{label:<16}", style=PALETTE["text"])
            t.append(f"{self.preview[:28]:<30}", style=PALETTE["dim"])
            if self.state == "spec":
                src = "peek" if self.source == "peek" else "speculating"
                t.append(f"{src} · {now - self.t_born:4.1f}s", style=PALETTE["spec"])
            elif self.state == "ready":
                t.append(f"cached · took {(self.ms or 0) / 1000:.1f}s", style=PALETTE["ok"])
            elif self.state == "claimed":
                extra = f" · +{self.saved_s:.1f}s" if self.saved_s else ""
                t.append(f"cache hit{extra}", style=f"bold {PALETTE['ok']}")
            elif self.state == "evicted":
                t.append("evicted", style=PALETTE["err"])
        self._out_line(t)
        self.update(t)


class ExecPanel(Static):
    """The interpreter's view of the current REPL block: one row per
    statement, a ▶ arrow on the one running now, ✓ + elapsed when done."""

    def __init__(self) -> None:
        super().__init__("")
        self.rows: list[dict] = []
        self.extra: list[str] = []  # REPL stdout, shown under the statements
        self.cur: int | None = None

    def set_block(self, stmts: list[str]) -> None:
        self.rows = []
        for src in stmts:
            lines = src.splitlines() or [""]
            label = lines[0] + (" …" if len(lines) > 1 else "")
            self.rows.append({"label": label, "state": "pending", "ms": 0.0, "t0": 0.0})
        self.extra = []
        self.cur = None
        self.repaint()

    def add_output(self, text: str, style_key: str = "chrome") -> None:
        for line in text.rstrip().splitlines() or [""]:
            self.extra.append((line, style_key))
        self.repaint()

    def begin(self, i: int) -> None:
        if 0 <= i < len(self.rows):
            self.cur = i
            self.rows[i]["state"] = "running"
            self.rows[i]["t0"] = time.perf_counter()
            self.repaint()

    def end(self, i: int, ms: float) -> None:
        if 0 <= i < len(self.rows):
            self.rows[i]["state"] = "done"
            self.rows[i]["ms"] = ms
            if self.cur == i:
                self.cur = None
            self.repaint()

    def tick(self) -> None:
        if self.cur is not None:
            self.repaint()

    def repaint(self) -> None:
        # size rows to the panel so long statements truncate, never wrap
        w = self.size.width or 100
        label_w = max(30, w - 11)
        t = Text()
        now = time.perf_counter()
        for i, row in enumerate(self.rows):
            if i:
                t.append("\n")
            label = row["label"]
            label = label[: label_w - 2] + " …" if len(label) > label_w else label
            if row["state"] == "running":
                t.append(" ▶ ", style=f"bold {PALETTE['run']}")
                t.append(f"{label:<{label_w}}", style=PALETTE["text"])
                t.append(f"{now - row['t0']:5.1f}s", style=PALETTE["run"])
            elif row["state"] == "done":
                t.append(" ✓ ", style=PALETTE["ok"])
                t.append(f"{label:<{label_w}}", style=PALETTE["dim"])
                ms = row["ms"]
                t.append(
                    f"{ms / 1000:5.1f}s" if ms >= 100 else "     ·",
                    style=PALETTE["ok"] if ms >= 100 else PALETTE["dim"],
                )
            else:
                t.append("   ", style=PALETTE["dim"])
                t.append(label, style=PALETTE["dim"])
        for line, style_key in self.extra:
            t.append("\n   ")
            t.append(line[: label_w + 4], style=PALETTE[style_key])
        self.update(t)


class Column(Vertical):
    """One agent's view: head (name + phase/timer), stream, calls, answer."""

    def __init__(
        self,
        side: str,
        title: Text | None = None,
        hint: str | None = None,
        show_turns: bool = False,
    ):
        super().__init__(classes="col")
        self.side = side
        self.title_text = title or (
            Text("● speculative", style=f"bold {PALETTE['spec']}")
            if side == "spec"
            else Text("○ serial", style=f"bold {PALETTE['run']}")
        )
        self.hint = hint or (
            "sub-calls run ahead of the interpreter; results land in the cache"
            if side == "spec"
            else "sub-calls can only start after generation finishes"
        )
        self.show_turns = show_turns
        self.phase = "waiting"
        self.turn = 0
        self.t_start: float | None = None
        self.wall: float | None = None
        self.rows: dict = {}
        self.stream = StreamView()
        self.exec_panel = ExecPanel()
        self._placeholder: Static | None = None
        self._inline_row: CallRow | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="colhead"):
            yield Static(self.title_text, classes="colname")
            yield Static("", classes="colphase")
        with VerticalScroll(classes="streamwrap"):
            yield self.stream
        with VerticalScroll(classes="execwrap"):
            yield self.exec_panel
        label = "─ speculation cache " if self.side == "spec" else "─ tool calls · run inline "
        yield Static(Text(label, style=PALETTE["dim"]), classes="callslabel")
        yield VerticalScroll(classes="callswrap")
        with VerticalScroll(classes="answerwrap"):
            yield Static("", classes="answerbox")

    def on_mount(self) -> None:
        self._placeholder = Static(Text(f" · {self.hint}", style=PALETTE["dim"]))
        self.query_one(".callswrap").mount(self._placeholder)

    def add_row(self, key, row: CallRow) -> None:
        if self._placeholder is not None:
            self._placeholder.remove()
            self._placeholder = None
        self.rows[key] = row
        wrap = self.query_one(".callswrap")
        wrap.mount(row)
        wrap.scroll_end(animate=False)

    def show_answer(self, final: str, wall: float, extra: str = "") -> None:
        wrap = self.query_one(".answerwrap")
        wrap.border_title = f"answer · after {wall:.1f}s{extra}"
        self.query_one(".answerbox", Static).update(style_answer(final))
        wrap.display = True

    def reset(self) -> None:
        for r in list(self.rows.values()):
            r.remove()
        self.rows.clear()
        self._inline_row = None
        self.stream.remove()
        self.stream = StreamView()
        self.query_one(".streamwrap").mount(self.stream)
        wrap = self.query_one(".answerwrap")
        wrap.display = False
        self.query_one(".answerbox", Static).update("")
        self.exec_panel.set_block([])
        self.query_one(".execwrap").display = False
        self.query_one(".streamwrap").display = True
        self.phase, self.t_start, self.wall, self.turn = "waiting", None, None, 0
        self._placeholder = None
        self.on_mount()

    def tick(self) -> None:
        if not self.is_mounted:
            return
        now = time.perf_counter()
        elapsed = (self.wall if self.wall is not None else now - (self.t_start or now)) or 0
        t = Text()
        if self.wall is not None:
            t.append("done ", style=f"bold {PALETTE['ok']}")
            t.append(f"{elapsed:.1f}s", style=f"bold {PALETTE['ok']}")
        elif self.phase == "waiting":
            t.append("…", style=PALETTE["dim"])
        else:
            color = PALETTE["run"] if self.phase == "generating" else PALETTE["warn"]
            turn = f"turn {self.turn} · " if self.show_turns and self.turn else ""
            t.append(f"{spin(now)} {turn}{self.phase} ", style=color)
            t.append(f"{elapsed:.1f}s", style=PALETTE["chrome"])
        self.query_one(".colphase", Static).update(t)
        self.exec_panel.tick()
        for r in self.rows.values():
            if r.out_dirty or r.state in ("running", "spec"):
                r.rerender()


def apply_side_event(col: Column, ev, solve_dp_preview: str = "risk lattice") -> str:
    """Route one bus event into a Column. Returns the event kind (so callers
    can layer extra behavior, e.g. on 'side_done' or 'real_block_end')."""
    d, k = ev.data, ev.kind
    if k == "stream_begin":
        col.phase = "generating"
        col.turn += 1
        col.t_start = col.t_start or time.perf_counter()
        # generation resumes: bring the assistant pane back, tuck the repl away
        col.query_one(".streamwrap").display = True
        col.query_one(".execwrap").display = False
        if col.show_turns and col.turn > 1:
            col.stream.add_marker(f"turn {col.turn}")
    elif k == "token":
        col.stream.feed(d["text"])
    elif k == "stream_end":
        # the assistant pane ends with its reasoning — nothing else goes in it
        col.stream.finish()
        col.phase = "executing"
    elif k == "dispatch":
        preview = solve_dp_preview if d["tool"] == "solve_dp" else d.get("preview", "")
        col.add_row(d["seq"], CallRow("spec", d["tool"], preview, d.get("source", "shadow")))
    elif k == "subtoken" and d.get("seq") in col.rows:
        col.rows[d["seq"]].feed_out(d.get("text", ""))
    elif k == "ready" and d.get("seq") in col.rows:
        row = col.rows[d["seq"]]
        row.state, row.ms = "ready", d.get("ms")
        row.rerender()
    elif k == "claim_hit" and d.get("seq") in col.rows:
        row = col.rows[d["seq"]]
        row.state = "claimed"
        row.rerender()
    elif k == "claim_miss":
        col.exec_panel.add_output(
            f"✗ cache miss: {d.get('tool', '?')} — computing inline now", "err"
        )
    elif k == "claim_done" and d.get("seq") in col.rows:
        row = col.rows[d["seq"]]
        row.saved_s = d.get("saved_ms", 0) / 1000
        row.rerender()
    elif k == "evict" and d.get("seq") in col.rows:
        row = col.rows[d["seq"]]
        row.state = "evicted"
        row.rerender()
    elif k == "call_begin":
        row = CallRow("serial", d["tool"], d.get("preview", ""), "inline", d.get("n", 1))
        col.add_row(("c", d["cid"]), row)
        col._inline_row = row
    elif k == "inline_token":
        if col._inline_row is not None:
            col._inline_row.feed_out(d.get("text", ""))
    elif k == "call_end" and ("c", d.get("cid")) in col.rows:
        row = col.rows[("c", d["cid"])]
        row.state, row.ms = "done", d.get("ms")
        row.rerender()
        if col._inline_row is row:
            col._inline_row = None
    elif k == "real_block_begin":
        if d.get("stmts"):
            # the visual cue: the assistant pane collapses and a big repl
            # panel takes its place — generation is over, execution begins
            col.exec_panel.set_block(d["stmts"])
            col.query_one(".streamwrap").display = False
            wrap = col.query_one(".execwrap")
            wrap.border_title = "⚡ generation done — executing repl"
            wrap.display = True
    elif k == "real_stmt_begin":
        col.exec_panel.begin(d.get("index", -1))
        wrap = col.query_one(".execwrap")
        wrap.scroll_to(y=max(0, d.get("index", 0) - 6), animate=False)
    elif k == "real_stmt_end":
        col.exec_panel.end(d.get("index", -1), d.get("ms", 0.0))
    elif k == "real_block_end":
        wrap = col.query_one(".execwrap")
        wrap.border_title = f"repl · block ran in {d.get('ms', 0) / 1000:.1f}s"
        if d.get("stdout"):
            col.exec_panel.add_output(d["stdout"][:600])
    return k
