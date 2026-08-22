"""spec-ptc TUI — four tmux-style panels, hard split between what is."""

from __future__ import annotations

import argparse
import threading
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, RichLog, Static

from demo.scenarios import CATALOG, get_scenario
from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM
from spec_ptc.runtime.harness import Harness

PANELS = ("main", "shadow", "real", "specs")


class SpecCard(Static):
    """One speculated call: header + its live sub-LM tokens."""

    def __init__(self, seq: int, tool: str, preview: str, source: str = "shadow"):
        super().__init__("")
        self.seq, self.tool, self.preview, self.source = seq, tool, preview, source
        self.state = "spec"
        self.tokens = ""
        self.ms: float | None = None
        self.refresh_text()

    def refresh_text(self) -> None:
        icon, color = {
            "spec": ("◆" if self.source == "peek" else "●", "yellow"),
            "ready": ("✔", "green"),
            "claimed": ("✚", "bright_green"),
            "evicted": ("✖", "red"),
        }[self.state]
        t = Text()
        t.append(f"{icon} #{self.seq} {self.tool} ", style=f"bold {color}")
        if self.source == "peek":
            t.append("peek ", style="bold magenta")
        if self.ms is not None:
            t.append(f"{self.ms:.0f}ms ", style="dim")
        t.append(f"({self.preview})\n", style="dim italic")
        tail = self.tokens[-160:]
        t.append(
            ("…" if len(self.tokens) > 160 else "") + tail,
            style="cyan" if self.state == "spec" else "dim cyan",
        )
        self.update(t)


class SpecPTCApp(App):
    CSS = """
    #row-top, #row-bottom { height: 1fr; }
    .panel { border: round $primary-darken-2; padding: 0 1; width: 1fr; height: 1fr; }
    .panel.focused { border: heavy $warning; }
    .panel.zoomed { width: 1fr; height: 1fr; }
    #stats { height: 4; border: round $accent; padding: 0 1; }
    SpecCard { margin: 0 0 1 0; }
    RichLog { background: transparent; }
    """
    BINDINGS = [
        ("q", "quit", "quit"),
        ("r", "restart", "restart"),
        ("1", "focus_panel('main')", "main"),
        ("2", "focus_panel('shadow')", "speculating"),
        ("3", "focus_panel('real')", "real exec"),
        ("4", "focus_panel('specs')", "sub-streams"),
        ("o", "cycle_focus", "next panel"),
        ("z", "zoom", "zoom"),
    ]

    def __init__(self, scenario_name: str, live_engine=None):
        super().__init__()
        self.scenario_name = scenario_name
        self.live_engine = live_engine
        self.cards: dict[int, SpecCard] = {}
        self.main_text = Text()
        self._focus_name = "main"
        self._zoomed = False
        self._t_start = 0.0
        self._serial_est = 0.0
        self._saved_ms = 0.0
        self._done = False
        self._final_wall: float | None = None
        self._stream_s: float | None = None
        self._runner: threading.Thread | None = None

    # ------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal(id="row-top"):
                with VerticalScroll(id="main", classes="panel focused"):
                    yield Static(Text(), id="mainstream")
                yield RichLog(
                    id="shadow", classes="panel", wrap=True, highlight=False, markup=True
                )
            with Horizontal(id="row-bottom"):
                yield RichLog(
                    id="real", classes="panel", wrap=True, highlight=False, markup=True
                )
                with VerticalScroll(id="specs", classes="panel"):
                    pass
            yield Static("", id="stats")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"spec-ptc — {self.scenario_name}"
        self.query_one("#main").border_title = "[1] MAIN STREAM (generation)"
        self.query_one("#shadow").border_title = "[2] SPECULATING — shadow side"
        self.query_one("#real").border_title = "[3] ACTUALLY RUNNING — real REPL"
        self.query_one("#specs").border_title = "[4] SUB-STREAMS (per call)"
        self.action_restart()
        self.set_interval(0.25, self.refresh_stats)

    # ------------------------------------------------------------- tmux keys
    def action_focus_panel(self, name: str) -> None:
        self._focus_name = name
        for p in PANELS:
            self.query_one(f"#{p}").set_class(p == name, "focused")
        self.query_one(f"#{name}").focus()

    def action_cycle_focus(self) -> None:
        i = PANELS.index(self._focus_name)
        self.action_focus_panel(PANELS[(i + 1) % len(PANELS)])

    def action_zoom(self) -> None:
        self._zoomed = not self._zoomed
        for p in PANELS:
            w = self.query_one(f"#{p}")
            w.display = (not self._zoomed) or p == self._focus_name
        self.query_one("#row-top").display = (not self._zoomed) or self._focus_name in (
            "main",
            "shadow",
        )
        self.query_one("#row-bottom").display = (not self._zoomed) or self._focus_name in (
            "real",
            "specs",
        )

    def action_restart(self) -> None:
        if self._runner and self._runner.is_alive():
            return
        for c in list(self.cards.values()):
            c.remove()
        self.cards.clear()
        self.main_text = Text()
        self.query_one("#shadow", RichLog).clear()
        self.query_one("#real", RichLog).clear()
        self._serial_est = self._saved_ms = 0.0
        self._done = False
        self._final_wall = None
        self._runner = threading.Thread(target=self._play, daemon=True)
        self._runner.start()

    # ------------------------------------------------------------- events
    def _on_event(self, ev) -> None:
        self.call_from_thread(self._apply_event, ev)

    def _apply_event(self, ev) -> None:
        d, k = ev.data, ev.kind
        t = ev.t - self._t_start
        shadow = self.query_one("#shadow", RichLog)
        real = self.query_one("#real", RichLog)
        if k == "token":
            self.main_text.append(d["text"], style="white")
            self.query_one("#mainstream", Static).update(self.main_text)
            self.query_one("#main", VerticalScroll).scroll_end(animate=False)
        elif k == "stmt_closed":
            src = d["src"].splitlines()[0][:46]
            shadow.write(f"[dim]{t:6.2f}s ▸ stmt closed: {src}[/dim]")
        elif k == "shadow_exec":
            shadow.write(f"[blue]{t:6.2f}s ▶ shadow exec: {d['src'][:46]}[/blue]")
        elif k == "dispatch":
            icon = "◆ peek" if d.get("source") == "peek" else "● shadow"
            shadow.write(
                f"[yellow]{t:6.2f}s {icon} dispatch #{d['seq']} "
                f"{d['tool']}({d.get('preview', '')[:36]})[/yellow]"
            )
            card = SpecCard(
                d["seq"], d["tool"], d.get("preview", ""), d.get("source", "shadow")
            )
            self.cards[d["seq"]] = card
            self.query_one("#specs", VerticalScroll).mount(card)
            self.query_one("#specs", VerticalScroll).scroll_end(animate=False)
        elif k == "adopt":
            shadow.write(f"[magenta]{t:6.2f}s ◇ adopt #{d['seq']} (peek owned)[/magenta]")
        elif k == "subtoken":
            c = self.cards.get(d["seq"])
            if c:
                c.tokens += d["text"]
                c.refresh_text()
        elif k == "ready":
            c = self.cards.get(d["seq"])
            if c:
                c.state, c.ms = "ready", d.get("ms")
                self._serial_est += d.get("ms", 0.0) / 1000
                c.refresh_text()
        elif k == "claim_hit":
            real.write(
                f"[bright_green]{t:6.2f}s ✚ CLAIM #{d.get('seq')} {d['tool']} "
                f"(ready={d.get('already_ready')})[/bright_green]"
            )
            c = self.cards.get(d.get("seq"))
            if c:
                c.state = "claimed"
                c.refresh_text()
        elif k == "claim_done":
            self._saved_ms += d.get("saved_ms", 0.0)
        elif k == "claim_miss":
            real.write(f"[red]{t:6.2f}s ○ miss {d['tool']} — running inline[/red]")
        elif k == "evict":
            shadow.write(f"[red]{t:6.2f}s ✖ evict #{d.get('seq')} ({d.get('reason')})[/red]")
            c = self.cards.get(d.get("seq"))
            if c:
                c.state = "evicted"
                c.refresh_text()
        elif k == "shadow_stop":
            shadow.write(f"[bold red]{t:6.2f}s ▲ shadow stopped: {d.get('reason')}[/bold red]")
        elif k == "real_block_begin":
            real.write(f"[bold]{t:6.2f}s ══ executing block {d['block']} ══[/bold]")
        elif k == "real_block_end":
            if d.get("stdout"):
                real.write(f"[white]{d['stdout'].rstrip()[:400]}[/white]")
            if d.get("stderr", "").strip():
                real.write(f"[red]{d['stderr'][:200]}[/red]")
            real.write(f"[dim]block {d['block']} done in {d['ms']:.0f}ms[/dim]")
            if d.get("answer"):
                real.write(f"[bold green]ANSWER: {d['answer'][:200]}[/bold green]")
        elif k in ("stream_begin", "stream_end", "exec_begin", "exec_end"):
            label = {
                "stream_begin": "generation begins",
                "stream_end": "generation done",
                "exec_begin": "real execution begins",
                "exec_end": "turn complete",
            }[k]
            self.main_text.append(f"\n── {label} ──\n", style="dim blue")
            self.query_one("#mainstream", Static).update(self.main_text)
            (real if k.startswith("exec") else shadow).write(
                f"[blue]{t:6.2f}s ── {label} ──[/blue]"
            )

    def refresh_stats(self) -> None:
        n = len(self.cards)
        ready = sum(1 for c in self.cards.values() if c.state in ("ready", "claimed"))
        claimed = sum(1 for c in self.cards.values() if c.state == "claimed")
        evicted = sum(1 for c in self.cards.values() if c.state == "evicted")
        wall = (
            self._final_wall
            if self._done and self._final_wall
            else (time.perf_counter() - self._t_start if self._t_start else 0.0)
        )
        t = Text()
        t.append(
            f" ◆● speculated {n}   ✔ ready {ready}   ✚ claimed {claimed}   "
            f"✖ evicted {evicted}   banked {self._saved_ms / 1000:.1f}s\n",
            style="bold",
        )
        baseline_est = self._serial_est + (self._stream_s or 0.0)
        t.append(f" wall {wall:6.1f}s   serial-call-time {self._serial_est:6.1f}s", style="")
        if wall > 0 and baseline_est > 0:
            t.append(f"   est. baseline {baseline_est:5.1f}s   ", style="dim")
            t.append(
                f"speedup ≈ {baseline_est / max(wall, 0.01):4.2f}x",
                style="bold green" if baseline_est > wall else "bold red",
            )
        if self._done:
            t.append("   [DONE — r replays]", style="bold yellow")
        self.query_one("#stats", Static).update(t)

    # ------------------------------------------------------------- the run
    def _play(self) -> None:
        sc = get_scenario(self.scenario_name)
        bus = EventBus()
        bus.subscribe(self._on_event)
        eng = self.live_engine or MockLM(sc.timing)
        h = Harness(eng, "spec", bus=bus, context=sc.context)
        self._t_start = time.perf_counter()
        for turn in sc.turns:
            h.run_turn(eng.stream_main(turn))
        self._final_wall = time.perf_counter() - self._t_start
        ev = [e for e in bus.history if e.kind in ("stream_begin", "stream_end")]
        self._stream_s = sum(b.t - a.t for a, b in zip(ev[::2], ev[1::2], strict=False))
        h.launcher.shutdown()
        self._done = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="oolong-mood-agg")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--vllm", action="store_true")
    args = ap.parse_args()
    if args.list:
        for s in CATALOG:
            print(f"{s.name:26s} [{s.category}] {s.description}")
        return
    engine = None
    if args.vllm:
        from demo.live import HybridEngine, engine_from_env
        from demo.scenarios import get_scenario as _gs

        engine = HybridEngine(_gs(args.scenario).timing, engine_from_env())
    SpecPTCApp(args.scenario, live_engine=engine).run()


if __name__ == "__main__":
    main()
