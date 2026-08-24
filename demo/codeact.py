"""Interactive CodeAct TUI: a live RLM with speculative tool calls.

Type a question about the loaded context (an OOLONG trec-coarse dataset, 787
general-knowledge questions in ~32k tokens); the main model writes REPL code
turn by turn against seeded, temperature-0 vLLM endpoints. Three views:
the session (main context: your questions, the streaming code, REPL output),
the SPECULATING lane (sub-calls launched while code is still streaming, each
with a `╰─▸` line showing its output arrive), and the ACTUALLY RUNNING lane
(claims, misses, executed blocks).

Run:  python -m demo.codeact          (needs live endpoints: `just serve`)
Keys: enter to send · ctrl+q to quit. The input is prefilled with the task's
standard question — press enter to watch it, or ask your own.
"""

from __future__ import annotations

import argparse
import threading
import time

from demo.live import RLM_SYSTEM, rlm_turns
from demo.oolong_race import DEFS, N_CHUNKS, load_task
from spec_ptc.contracts.events import EventBus, SpecEvent
from spec_ptc.runtime.engines import engine_from_env
from spec_ptc.runtime.harness import Harness

MAX_TURNS = 5

CONTEXT_NOTE = (
    "The loaded context holds 787 general-knowledge questions, one per line "
    "(lines contain ' || Instance: '). Each question's answer falls into one "
    f"of 6 categories: location, numeric value, description and abstract "
    f"concept, abbreviation, human being, entity ({DEFS}).\n"
    f"For aggregate questions: split the question lines into ~{N_CHUNKS} "
    "chunks, fan out ONE llm_query per chunk with llm_query_batched asking "
    "for machine-parseable per-line labels or extractions (repeat the label "
    "definitions inside each sub-prompt), then aggregate in code over the "
    "returned list. Print intermediate results, then answer in a later turn."
)


def build_tui(prefill: bool = True):
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Input, Static

    from demo.ui import PALETTE, CallRow, StreamView, spin

    task = load_task()

    class CodeActApp(App):
        CSS = """
        Screen { background: $background; }
        #topbar { height: 1; padding: 0 2; }
        #topleft { width: 1fr; }
        #topright { width: auto; }
        #streamwrap {
            height: 2fr; border: round #30363d; margin: 0 1; padding: 0 1;
            border-title-color: #7d8590;
        }
        #lanes { height: 1fr; }
        .lane { width: 1fr; border: round #30363d; margin: 0 1; padding: 0 1; }
        .lanescroll { height: 1fr; }
        CallRow { height: 2; }
        #status { height: 1; padding: 0 2; }
        #ask { margin: 0 1; border: round #30363d; }
        #ask:focus { border: round #58a6ff; }
        """
        BINDINGS = [("ctrl+q", "quit", "quit")]

        def __init__(self) -> None:
            super().__init__()
            self.title = "spec-ptc · codeact"
            self.stream = StreamView()
            self.runlog = StreamView()
            self.rows: dict = {}
            self._busy = False
            self._t_query = 0.0
            self.turn = 0
            self.bus = EventBus()
            self.bus.subscribe(self._on_event)
            self.engine = engine_from_env(
                bus=self.bus,
                main_temperature=0.0,
                sub_temperature=0.0,
                seed=0,
                sub_max_tokens=512,
            )
            self.harness = Harness(self.engine, "spec", bus=self.bus, context=task["context"])
            self.messages = [
                {"role": "system", "content": RLM_SYSTEM + "\n" + CONTEXT_NOTE},
            ]

        def compose(self) -> ComposeResult:
            with Horizontal(id="topbar"):
                left = Text()
                left.append("spec-ptc ", style=f"bold {PALETTE['text']}")
                left.append(
                    "· codeact — live RLM over an OOLONG 32k context · "
                    "Qwen3.5-27B ×2, temp 0, seed 0",
                    style=PALETTE["chrome"],
                )
                yield Static(left, id="topleft")
                yield Static(
                    Text("enter send · ctrl+q quit", style=PALETTE["dim"]), id="topright"
                )
            with VerticalScroll(id="streamwrap"):
                yield self.stream
            with Horizontal(id="lanes"):
                with Vertical(classes="lane") as spec_lane:
                    spec_lane.border_title = "speculation cache"
                    yield VerticalScroll(classes="lanescroll", id="speclane")
                with Vertical(classes="lane") as run_lane:
                    run_lane.border_title = "actually running"
                    with VerticalScroll(classes="lanescroll", id="runlane"):
                        yield self.runlog
            yield Static("", id="status")
            yield Input(
                value=task["question"] if prefill else "",
                placeholder="ask about the loaded context…",
                id="ask",
            )

        def on_mount(self) -> None:
            self.set_interval(0.1, self._tick)
            self.stream.add_line(
                Text(
                    f"context loaded: {task['task_id']} — 787 questions, "
                    f"{len(task['context'])} chars",
                    style=PALETTE["dim"],
                )
            )
            self.query_one("#ask", Input).focus()

        # ---------------------------------------------------------- input
        def on_input_submitted(self, event) -> None:
            q = event.value.strip()
            if not q or self._busy:
                return
            self._busy = True
            self.turn = 0
            self._t_query = time.perf_counter()
            box = self.query_one("#ask", Input)
            box.value = ""
            box.placeholder = "…running — wait for the answer"
            u = Text()
            u.append("\n❯ you  ", style=f"bold {PALETTE['run']}")
            u.append(q, style=PALETTE["text"])
            self.stream.add_line(u)
            self.messages.append({"role": "user", "content": q})
            threading.Thread(target=self._worker, daemon=True).start()

        def _worker(self) -> None:
            # fresh answer scaffold per question, same persistent REPL
            self.harness.repl.locals["answer"] = {"content": "", "ready": False}
            final = rlm_turns(self.harness, self.engine, self.messages, max_turns=MAX_TURNS)
            self.bus.emit("query_done", final=final or "(no answer set)")

        # ---------------------------------------------------------- events
        def _on_event(self, ev: SpecEvent) -> None:
            if not self.is_running:
                return
            try:
                self.call_from_thread(self._apply, ev)
            except Exception:
                pass

        def _log(self, text: str, style: str) -> None:
            self.runlog.add_line(Text(text, style=style))
            self.query_one("#runlane").scroll_end(animate=False)

        def _apply(self, ev: SpecEvent) -> None:
            try:
                self._apply_inner(ev)
            except Exception as e:
                self._ui_error = f"{type(e).__name__}: {e}"

        def _apply_inner(self, ev: SpecEvent) -> None:
            d, k = ev.data, ev.kind
            if k == "stream_begin":
                self.turn += 1
                self.stream.add_marker(f"turn {self.turn}")
            elif k == "token":
                self.stream.feed(d["text"])
                self.query_one("#streamwrap").scroll_end(animate=False)
            elif k == "stream_end":
                self.stream.finish()
                self.stream.add_marker("generation done — executing repl")
            elif k == "dispatch":
                row = CallRow(
                    "spec", d["tool"], d.get("preview", ""), d.get("source", "shadow")
                )
                self.rows[d["seq"]] = row
                lane = self.query_one("#speclane")
                lane.mount(row)
                lane.scroll_end(animate=False)
            elif k == "subtoken" and d.get("seq") in self.rows:
                self.rows[d["seq"]].feed_out(d.get("text", ""))
            elif k == "ready" and d.get("seq") in self.rows:
                row = self.rows[d["seq"]]
                row.state, row.ms = "ready", d.get("ms")
                row.rerender()
            elif k == "evict" and d.get("seq") in self.rows:
                row = self.rows[d["seq"]]
                row.state = "evicted"
                row.rerender()
            elif k == "claim_hit":
                if d.get("seq") in self.rows:
                    row = self.rows[d["seq"]]
                    row.state = "claimed"
                    row.rerender()
                self._log(f"✚ cache hit #{d.get('seq', '?')} {d['tool']}", PALETTE["ok"])
            elif k == "claim_miss":
                self._log(f"○ cache miss {d['tool']} — running inline", PALETTE["err"])
            elif k == "real_block_begin":
                self._log(f"══ block {d['block']} ══", PALETTE["chrome"])
            elif k == "real_block_end":
                if d.get("stdout"):
                    for line in d["stdout"].rstrip().splitlines()[:10]:
                        self._log("  " + line[:90], PALETTE["prose"])
                self._log(f"block {d['block']} done in {d.get('ms', 0):.0f}ms", PALETTE["dim"])
            elif k == "query_done":
                a = Text()
                a.append("\n✔ answer  ", style=f"bold {PALETTE['ok']}")
                a.append(str(d["final"]), style=PALETTE["text"])
                self.stream.add_line(a)
                self._log(f"ANSWER: {str(d['final'])[:80]}", f"bold {PALETTE['ok']}")
                self._busy = False
                box = self.query_one("#ask", Input)
                box.placeholder = "ask a follow-up…"
                box.focus()

        # ---------------------------------------------------------- ticker
        def _tick(self) -> None:
            try:
                self._tick_inner()
            except Exception as e:
                self._ui_error = f"{type(e).__name__}: {e}"

        def _tick_inner(self) -> None:
            for r in self.rows.values():
                if r.state == "spec":
                    r.rerender()
            t = Text()
            now = time.perf_counter()
            n = len(self.rows)
            claimed = sum(1 for r in self.rows.values() if r.state == "claimed")
            if self._busy:
                t.append(
                    f"{spin(now)} turn {self.turn} · {now - self._t_query:5.1f}s   ",
                    style=PALETTE["run"],
                )
            else:
                t.append("ready   ", style=PALETTE["ok"])
            t.append(
                f"◆ speculated {n} · ✓ claimed {claimed}",
                style=PALETTE["chrome"],
            )
            if getattr(self, "_ui_error", None):
                t.append(f"  ⚠ {self._ui_error}", style=PALETTE["err"])
            self.query_one("#status", Static).update(t)

    return CodeActApp()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-prefill", action="store_true", help="start with an empty input")
    args = ap.parse_args()
    build_tui(prefill=not args.no_prefill).run()


if __name__ == "__main__":
    main()
