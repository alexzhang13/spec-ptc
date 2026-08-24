"""SOLO speculative run (video/demo asset — intentionally not committed).

No comparison column: one agent, full width, and the mechanism laid bare.
While the model streams its code, sub-calls are speculated ahead of the
interpreter and their results land in the SPECULATION CACHE (left lane, each
with a `╰─▸` line streaming its output). When generation finishes, the
ACTUALLY RUNNING panel (right) shows the interpreter walking the block —
every call is a cache hit, served instantly from work done during streaming.

Starts empty: your keystrokes type the question (enter completes it).

Run:  python -m demo.solo [--speed 1.5] [--auto-type] [--auto-exit N]
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque

from demo.race import CONTEXT, QUERY, SCRIPT, RaceEngine, RaceTiming, SideResult
from spec_ptc.contracts.events import EventBus, SpecEvent
from spec_ptc.runtime.engines import MockTiming
from spec_ptc.runtime.harness import Harness


def start_solo(speed: float, on_event):
    """One speculative RLM turn; a 'side_done' event closes the run."""
    rt = RaceTiming().scaled(speed)
    result = SideResult()
    bus = EventBus()
    bus.subscribe(on_event)
    eng = RaceEngine(
        MockTiming(
            main_tok_per_s=rt.tok_s,
            sub_base_s=rt.sub_base_s,
            sub_jitter_s=rt.sub_jitter_s,
            sub_tokens=12,
        ),
        dp_s=rt.dp_s,
    )
    # fast-but-not-instant: the interpreter arrow visibly walks the block
    h = Harness(eng, "spec", bus=bus, context=CONTEXT, stmt_pace=0.15 / speed)

    def run():
        t_start = time.perf_counter()
        out = h.run_turn(eng.stream_main(SCRIPT))
        result.wall = time.perf_counter() - t_start
        result.final = out.final_answer
        if out.metrics:
            result.hits, result.misses = out.metrics.hits, out.metrics.misses
            result.dispatched = out.metrics.dispatched
        bus.emit(
            "side_done",
            wall=result.wall,
            hits=result.hits,
            misses=result.misses,
            final=result.final or "",
        )
        h.launcher.shutdown()

    threading.Thread(target=run, daemon=True).start()
    return result


def build_tui(speed: float, auto_exit: float = 0.0, manual: bool = False):
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Static

    from demo.ui import (
        PALETTE,
        CallRow,
        ChatBox,
        ExecPanel,
        StreamView,
        spin,
        style_answer,
    )

    class SoloApp(App):
        CSS = """
        Screen { background: $background; }
        #topbar { height: 1; padding: 0 2; }
        #topleft { width: 1fr; }
        #topright { width: auto; }
        .chat {
            height: auto; max-height: 5; margin: 0 1; padding: 0 1;
            border: round #30363d; border-title-color: #7d8590;
        }
        #streamwrap {
            height: 2fr; border: round #30363d; margin: 0 1; padding: 0 1;
            border-title-color: #7d8590;
        }
        #lanes { height: 1fr; }
        .lane { width: 1fr; border: round #30363d; margin: 0 1; padding: 0 1; }
        #cachelane { border: round #bc8cff 50%; border-title-color: #bc8cff; }
        #runlane-box { border: round #58a6ff 50%; border-title-color: #58a6ff; }
        .lanescroll { height: 1fr; }
        CallRow { height: 2; }
        #answerwrap {
            display: none; height: auto; max-height: 4; margin: 0 1;
            padding: 0 1; border: round #238636; border-title-color: #3fb950;
        }
        #verdict { height: 1; padding: 0 2; }
        """
        BINDINGS = [("q", "quit", "quit"), ("r", "restart", "restart")]

        def __init__(self) -> None:
            super().__init__()
            self.title = "spec-ptc · solo"
            self.chat = ChatBox(QUERY, self._start, cps=45.0 * speed, manual=manual)
            self.stream = StreamView()
            self.exec_panel = ExecPanel()
            self.rows: dict = {}
            self.stats = {"disp": 0, "cached": 0, "cached_ms": 0.0, "hits": 0, "misses": 0}
            self.wall: float | None = None
            self.final: str | None = None
            self.phase = "waiting"
            self.t_start: float | None = None
            self._stream_t0: float | None = None
            self.stream_s = 0.0
            self._evq: deque = deque()
            self._race_active = False

        def compose(self) -> ComposeResult:
            with Horizontal(id="topbar"):
                left = Text()
                left.append("spec-ptc ", style=f"bold {PALETTE['text']}")
                left.append(
                    "· speculative tool calls — one agent, cache in plain sight",
                    style=PALETTE["chrome"],
                )
                yield Static(left, id="topleft")
                yield Static(
                    Text(f"{speed:g}× · q quit · r rerun", style=PALETTE["dim"]),
                    id="topright",
                )
            yield self.chat
            with VerticalScroll(id="streamwrap") as sw:
                sw.border_title = "assistant"
                yield self.stream
            with Horizontal(id="lanes"):
                with Vertical(classes="lane", id="cachelane") as cache:
                    cache.border_title = "speculation cache — fills while the model streams"
                    yield VerticalScroll(classes="lanescroll", id="cachescroll")
                with Vertical(classes="lane", id="runlane-box") as run:
                    run.border_title = "actually running — the interpreter"
                    with VerticalScroll(classes="lanescroll", id="runscroll"):
                        yield self.exec_panel
            with VerticalScroll(id="answerwrap"):
                yield Static("", id="answerbox")
            yield Static("", id="verdict")

        def on_mount(self) -> None:
            self.set_interval(0.05, self._pump)
            self.set_interval(0.1, self._tick)
            hint = Static(
                Text(" · nothing speculated yet", style=PALETTE["dim"]), id="cachehint"
            )
            self.query_one("#cachescroll").mount(hint)
            run_hint = Static(
                Text(" · waiting for generation to finish", style=PALETTE["dim"]),
                id="runhint",
            )
            self.query_one("#runscroll").mount(run_hint, before=self.exec_panel)

        def on_key(self, event) -> None:
            if manual and not self.chat.sent:
                self.chat.keypress(event.key)
                event.stop()

        def check_action(self, action: str, parameters) -> bool | None:
            if manual and not self.chat.sent and action in ("quit", "restart"):
                return False
            return True

        def _start(self) -> None:
            self._race_active = True
            self.t_start = time.perf_counter()
            start_solo(speed, self._on_event)

        def action_restart(self) -> None:
            if self._race_active:
                return
            for r in list(self.rows.values()):
                r.remove()
            self.rows.clear()
            self.stream.remove()
            self.stream = StreamView()
            self.query_one("#streamwrap").mount(self.stream)
            self.exec_panel.set_block([])
            self.stats = {"disp": 0, "cached": 0, "cached_ms": 0.0, "hits": 0, "misses": 0}
            self.wall = self.final = None
            self.phase, self.t_start, self.stream_s = "waiting", None, 0.0
            wrap = self.query_one("#answerwrap")
            wrap.display = False
            self.query_one("#answerbox", Static).update("")
            self.query_one("#streamwrap").display = True
            self.query_one("#runlane-box").border_title = "actually running — the interpreter"
            self.chat.restart()

        # ------------------------------------------------------------ events
        def _on_event(self, ev: SpecEvent) -> None:
            self._evq.append(ev)

        def _pump(self) -> None:
            while self._evq:
                try:
                    self._apply(self._evq.popleft())
                except Exception as e:
                    self._ui_error = f"{type(e).__name__}: {e}"

        def _apply(self, ev: SpecEvent) -> None:
            d, k = ev.data, ev.kind
            if k == "stream_begin":
                self.phase = "generating"
                self._stream_t0 = ev.t
            elif k == "token":
                self.stream.feed(d["text"])
            elif k == "stream_end":
                self.stream.finish()
                if self._stream_t0 is not None:
                    self.stream_s += ev.t - self._stream_t0
                self.phase = "executing"
            elif k == "dispatch":
                hint = self.query_one("#cachehint", Static)
                if hint.display:
                    hint.display = False
                preview = (
                    "risk lattice 6×6, horizon=12"
                    if d["tool"] == "solve_dp"
                    else d.get("preview", "")
                )
                row = CallRow("spec", d["tool"], preview, d.get("source", "shadow"))
                self.rows[d["seq"]] = row
                self.stats["disp"] += 1
                sc = self.query_one("#cachescroll")
                sc.mount(row)
                sc.scroll_end(animate=False)
            elif k == "subtoken" and d.get("seq") in self.rows:
                self.rows[d["seq"]].feed_out(d.get("text", ""))
            elif k == "ready" and d.get("seq") in self.rows:
                row = self.rows[d["seq"]]
                row.state, row.ms = "ready", d.get("ms")
                row.rerender()
                self.stats["cached"] += 1
                self.stats["cached_ms"] += d.get("ms", 0.0)
            elif k == "claim_hit" and d.get("seq") in self.rows:
                row = self.rows[d["seq"]]
                row.state = "claimed"
                row.rerender()
                self.stats["hits"] += 1
            elif k == "claim_done" and d.get("seq") in self.rows:
                row = self.rows[d["seq"]]
                row.saved_s = d.get("saved_ms", 0) / 1000
                row.rerender()
            elif k == "claim_miss":
                self.stats["misses"] += 1
                self.exec_panel.add_output(
                    f"✗ cache miss: {d.get('tool', '?')} — computing inline now", "err"
                )
            elif k == "evict" and d.get("seq") in self.rows:
                row = self.rows[d["seq"]]
                row.state = "evicted"
                row.rerender()
            elif k == "real_block_begin":
                # the cue: the assistant pane collapses; the interpreter pane
                # (already on screen) becomes the main event
                if d.get("stmts"):
                    self.query_one("#runhint", Static).display = False
                    self.exec_panel.set_block(d["stmts"])
                self.query_one("#streamwrap").display = False
                self.query_one(
                    "#runlane-box"
                ).border_title = "⚡ generation done — executing repl"
            elif k == "real_stmt_begin":
                self.exec_panel.begin(d.get("index", -1))
                self.query_one("#runscroll").scroll_to(
                    y=max(0, d.get("index", 0) - 6), animate=False
                )
            elif k == "real_stmt_end":
                self.exec_panel.end(d.get("index", -1), d.get("ms", 0.0))
            elif k == "real_block_end":
                self.query_one(
                    "#runlane-box"
                ).border_title = f"repl · block ran in {d.get('ms', 0) / 1000:.1f}s"
                if d.get("stdout"):
                    self.exec_panel.add_output(d["stdout"][:600])
            elif k == "side_done":
                self.wall = d["wall"]
                self.final = d.get("final") or ""
                self.phase = "done"
                self._race_active = False
                wrap = self.query_one("#answerwrap")
                wrap.border_title = f"answer · after {self.wall:.1f}s"
                self.query_one("#answerbox", Static).update(style_answer(self.final))
                wrap.display = True

        # ------------------------------------------------------------ ticker
        def _tick(self) -> None:
            try:
                self._tick_inner()
            except Exception as e:
                self._ui_error = f"{type(e).__name__}: {e}"

        def _tick_inner(self) -> None:
            now = time.perf_counter()
            self.exec_panel.tick()
            for r in self.rows.values():
                if r.out_dirty or r.state in ("running", "spec"):
                    r.rerender()
            s = self.stats
            t = Text()
            if self.wall is not None:
                baseline = self.stream_s + s["cached_ms"] / 1000
                t.append(f"✓ answered in {self.wall:.1f}s", style=f"bold {PALETTE['ok']}")
                t.append(
                    f" · {s['hits']}/{s['disp']} calls served from the cache"
                    f" · {s['cached_ms'] / 1000:.1f}s of call time pre-computed",
                    style=PALETTE["chrome"],
                )
                t.append(
                    f" · without speculation ≈ {baseline:.1f}s "
                    f"({baseline / max(self.wall, 0.01):.2f}× slower)",
                    style=f"bold {PALETTE['ok']}",
                )
                t.append(" · r to rerun", style=PALETTE["dim"])
            elif self.phase == "waiting":
                if manual and self.chat.n == 0:
                    t.append(
                        "start typing — your keystrokes write the question "
                        "(enter completes it)",
                        style=PALETTE["dim"],
                    )
                else:
                    t.append("…", style=PALETTE["dim"])
            else:
                el = now - (self.t_start or now)
                color = PALETTE["run"] if self.phase == "generating" else PALETTE["warn"]
                t.append(f"{spin(now)} {self.phase} · {el:5.1f}s   ", style=color)
                inflight = s["disp"] - s["cached"]
                t.append(
                    f"cache: {inflight} speculating · {s['cached']} cached "
                    f"({s['cached_ms'] / 1000:.1f}s of call time) · "
                    f"{s['hits']} hits",
                    style=PALETTE["chrome"],
                )
            if getattr(self, "_ui_error", None):
                t.append(f"  ⚠ {self._ui_error}", style=PALETTE["err"])
            self.query_one("#verdict", Static).update(t)
            if auto_exit and self.wall is not None:
                self._done_at = getattr(self, "_done_at", time.perf_counter())
                if time.perf_counter() - self._done_at > auto_exit:
                    self.exit()

    return SoloApp()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speed", type=float, default=1.0, help="time-scale (2.0 = twice as fast)")
    ap.add_argument(
        "--auto-exit",
        type=float,
        default=0.0,
        help="exit N seconds after the run finishes (for recordings)",
    )
    ap.add_argument(
        "--auto-type",
        action="store_true",
        help="the query types itself (for unattended recordings)",
    )
    args = ap.parse_args()
    manual = not (args.auto_type or args.auto_exit)
    build_tui(args.speed, auto_exit=args.auto_exit, manual=manual).run()


if __name__ == "__main__":
    main()
