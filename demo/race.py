"""SIMULATED side-by-side race (video/demo asset — intentionally not committed).

The SAME assistant turn, generated token-by-token twice: left speculates while
the code streams, right waits for generation to finish. A user query types
itself into a chat box; each column shows its final report when it finishes.
Timing is simulated (MockLM) and the sub-model's answers are scripted, so the
run is deterministic and camera-ready. The honest, live-model versions are
demo/codeact.py (interactive) and demo/oolong_race.py (real RLM on OOLONG).

Run:  python -m demo.race            (TUI)
      python -m demo.race --headless (event timeline + verdict)
      --speed 2.0 halves every latency and doubles the token rate.
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

from demo.live import instrument_inline_calls
from demo.oolong import make_log_context
from spec_ptc.contracts.events import EventBus, SpecEvent
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

# --------------------------------------------------------------------- script
QUERY = (
    "I just inherited this team and their activity log — 240 entries over "
    "several months. Give me a quick health read: what are they working on, "
    "how has morale trended, and which stretch should I dig into first?"
)

PRE = (
    "Rather than reading all 240 entries in order, I'll window the log, pull "
    "one overview read, price which stretch matters most, and collect a short "
    "morale report per window — then stitch those into a health read. Code "
    "first, commentary after."
)

CODE = """\
# stage 1 — window the log and get an overview read
W = max(1, len(context) // 6)
windows = [context[i:i+W] for i in range(0, len(context), W)][:6]
overview = llm_query(
    'You are reading a team activity log. In two sentences: what is this '
    'team working on, and what is the overall tone? Log excerpt: '
    + windows[0][:600])

# stage 2 — prioritization plan: exact DP over the inter-window risk lattice
def risk_lattice(ws):
    lat = []
    for i, w in enumerate(ws):
        lat.append([(len(w) * (j + 3) + i * 17) % 97 for j in range(len(ws))])
    return lat

M = risk_lattice(windows)
plan_cost = solve_dp(M, horizon=12, discount=0.94)

# stage 3 — a morale report per window, batched over the sub-model
prompts = [
    'Assess this stretch of a team activity log. Two sentences on the '
    'dominant morale and the main friction point, then a final line '
    'VERDICT: ok, watch, or concern. Entries: ' + w[:800]
    for w in windows[:5]
]
reports = llm_query_batched(prompts)

# stage 4 — free-text labels must never crash the fold: parse defensively
verdicts = []
for raw in reports:
    text = str(raw).lower()
    v = next((c for c in ('concern', 'watch', 'ok') if 'verdict: ' + c in text), None)
    if v is None:
        v = next((c for c in ('concern', 'watch', 'ok') if c in text), 'watch')
    verdicts.append(v)

focus = verdicts.index('concern') if 'concern' in verdicts else 0
lines = ['TEAM HEALTH READ']
lines.append('Focus first: window ' + str(focus) + ' — ' + verdicts[focus]
             + ' (plan cost ' + str(plan_cost) + ' over a 12-step horizon)')
lines.append('Overview: ' + str(overview))
lines.append('')
for i, (v, r) in enumerate(zip(verdicts, reports)):
    lines.append('window ' + str(i) + ' [' + v.upper() + '] ' + str(r))
answer['content'] = '\\n'.join(lines)
answer['ready'] = True"""

POST = (
    "While that runs, the reasoning. The log is too long to read in one pass, "
    "so stage one windows it and pulls a two-sentence overview to anchor the "
    "report. Stage two prices which stretch deserves attention first with an "
    "exact dynamic program over a small inter-window risk lattice — pure local "
    "compute, so its cost is fixed however we schedule it. Stage three asks "
    "the sub-model for a short morale report per window; those prompts depend "
    "only on the windows from stage one, not on the solver, so their inputs "
    "were ready long before the plan came back. Stage four assumes the worst "
    "about free-text answers: verdict labels are parsed defensively, unknowns "
    "clamp to watch instead of raising, and the fold can never crash on a "
    "malformed reply. The final answer leads with the focus recommendation, "
    "then the overview, then the per-window verdicts."
)

SCRIPT = f"{PRE}\n```repl\n{CODE}\n```\n{POST}"
CONTEXT = make_log_context()

# Scripted sub-model answers: the demo's trajectory is deterministic, so the
# sub-LM's content is too — latency still comes from the timing machinery and
# is identical on both sides (claims key on args, not results).
OVERVIEW_ANSWER = (
    "An eight-person research team iterating on training runs, a gpu cluster, "
    "benchmarks, a dataset and a paper draft. The tone swings between genuine "
    "excitement over results and recurring frustration with infrastructure."
)

WINDOW_ANSWERS = [
    "Morale is broadly positive: several clear wins land and the benchmark "
    "work is generating momentum. Friction centers on intermittent gpu "
    "cluster hiccups, but nothing systemic yet. VERDICT: ok",
    "The mood turns uneven as deploy problems and flaky infrastructure "
    "interrupt otherwise steady progress. Several entries show patience "
    "wearing thin around the same recurring blockers. VERDICT: watch",
    "This is the roughest stretch: repeated regressions, visible anger at "
    "the tooling, and motivation running low across multiple people. The "
    "same failure keeps resurfacing without a durable fix. VERDICT: concern",
    "A clear recovery: calm, unhurried progress on the paper draft and "
    "dataset work with few surprises. Energy is steady rather than high, "
    "which reads as healthy consolidation. VERDICT: ok",
    "An anxious undertone dominates: people second-guess results and hover "
    "over dashboards even as the work advances. Watch for burnout signals "
    "if the uncertainty persists. VERDICT: watch",
]

FALLBACK_ANSWER = (
    "Steady progress with intermittent infrastructure friction; morale is "
    "serviceable but uneven. VERDICT: watch"
)

_RE_ENTRY = re.compile(r"\(entry (\d+)\)")


# --------------------------------------------------------------------- timing
@dataclass
class RaceTiming:
    tok_s: float = 16.0  # main-model stream rate (words/s)
    sub_base_s: float = 3.2  # sub-LM latency: base + hash jitter
    sub_jitter_s: float = 1.6
    dp_s: float = 10.0  # solve_dp stall

    def scaled(self, speed: float) -> RaceTiming:
        return RaceTiming(
            tok_s=self.tok_s * speed,
            sub_base_s=self.sub_base_s / speed,
            sub_jitter_s=self.sub_jitter_s / speed,
            dp_s=self.dp_s / speed,
        )


class RaceEngine(MockLM):
    """MockLM with scripted sub-answers plus solve_dp: looks like a solver, is
    a sleep — pure, so the shadow may run it early; the serial side pays for
    it inline."""

    def __init__(self, timing: MockTiming, dp_s: float):
        super().__init__(timing)
        self.dp_s = dp_s
        self._inline_lock = threading.Lock()
        self._inline_owner: int | None = None

    def _canned(self, prompt: str) -> str:
        if prompt.startswith("You are reading a team activity log"):
            return OVERVIEW_ANSWER
        if prompt.startswith("Assess this stretch"):
            m = _RE_ENTRY.search(prompt)
            if m:  # windows are contiguous sixths of a 240-entry log (~40
                # entries each); slices start mid-entry, so round, not floor
                idx = min(round(int(m.group(1)) / 40), len(WINDOW_ANSWERS) - 1)
                return WINDOW_ANSWERS[idx]
        return FALLBACK_ANSWER

    def sub_call(self, prompt: str, _spec=None) -> str:
        me = threading.get_ident()
        if _spec is None:  # inline: one coherent stream at a time for the UI
            with self._inline_lock:
                if self._inline_owner is None:
                    self._inline_owner = me
        text = self._canned(str(prompt))
        words = text.split(" ")
        step = self._latency(str(prompt)) / max(len(words), 1)
        try:
            for k, w in enumerate(words):
                if _spec is not None and _spec.state == "evicted":
                    raise RuntimeError("evicted")
                time.sleep(step)
                tok = w + (" " if k < len(words) - 1 else "")
                if _spec is not None:
                    self.bus.emit("subtoken", seq=_spec.seq, text=tok)
                elif self._inline_owner == me:
                    self.bus.emit("inline_token", text=tok)
        finally:
            if _spec is None and self._inline_owner == me:
                with self._inline_lock:
                    self._inline_owner = None
        return text

    def make_tools(self, reg, bus) -> None:
        super().make_tools(reg, bus)

        def solve_dp(matrix, horizon=8, discount=0.95, _spec=None):
            t_end = time.perf_counter() + self.dp_s
            while time.perf_counter() < t_end:
                if _spec is not None and _spec.state == "evicted":
                    raise RuntimeError("evicted")
                time.sleep(0.05)
            best = sum(min(row) for row in matrix)
            return round(best * (discount**horizon), 2)

        solve_dp.wants_spec = True  # type: ignore[attr-defined]
        reg.register(
            "solve_dp",
            solve_dp,
            speculatable=True,
            pure=True,
            latency_hint_ms=self.dp_s * 1000,
        )


# --------------------------------------------------------------------- runner
@dataclass
class SideResult:
    wall: float = 0.0
    final: str | None = None
    hits: int = 0
    misses: int = 0
    dispatched: int = 0


def start_race(speed: float = 1.0, on_event=None):
    """Launch both sides concurrently. on_event(side, SpecEvent) is called for
    every bus event; a synthetic 'side_done' event closes each side."""
    rt = RaceTiming().scaled(speed)
    results: dict[str, SideResult] = {"spec": SideResult(), "serial": SideResult()}
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()

    for mode, side in (("spec", "spec"), ("baseline", "serial")):
        bus = EventBus()
        if on_event is not None:
            bus.subscribe(lambda ev, s=side: on_event(s, ev))
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
        h = Harness(eng, mode, bus=bus, context=CONTEXT, stmt_pace=0.15 / speed)
        if mode == "baseline":
            instrument_inline_calls(h, bus, solve_dp_preview="risk lattice 6×6, horizon=12")

        def run(h=h, eng=eng, bus=bus, side=side):
            t_start = time.perf_counter()
            out = h.run_turn(eng.stream_main(SCRIPT))
            r = results[side]
            r.wall = time.perf_counter() - t_start
            r.final = out.final_answer
            if out.metrics:
                r.hits, r.misses = out.metrics.hits, out.metrics.misses
                r.dispatched = out.metrics.dispatched
            bus.emit(
                "side_done", wall=r.wall, hits=r.hits, misses=r.misses, final=r.final or ""
            )
            h.launcher.shutdown()

        threads.append(threading.Thread(target=run, daemon=True))

    for th in threads:
        th.start()
    return threads, results, t0


# ------------------------------------------------------------- headless twin
def run_headless(speed: float) -> None:
    t0_holder = {}
    lock = threading.Lock()

    def on_event(side: str, ev: SpecEvent) -> None:
        dt = ev.t - t0_holder.get("t0", ev.t)
        d = ev.data
        tag = "spec  " if side == "spec" else "serial"
        line = None
        if ev.kind == "dispatch":
            line = f"◆ dispatch #{d['seq']:<2} {d['tool']}({d.get('preview', '')[:38]})"
        elif ev.kind == "ready":
            line = f"✓ ready    #{d['seq']:<2} ({d.get('ms', 0) / 1000:.1f}s)"
        elif ev.kind == "claim_hit":
            line = f"✚ claim    #{d.get('seq', '?'):<2} {d['tool']}"
        elif ev.kind == "claim_miss":
            line = f"○ miss     {d['tool']}"
        elif ev.kind == "call_begin":
            n = f" ×{d['n']}" if d.get("n", 1) > 1 else ""
            line = f"▶ run      {d['tool']}{n} ({d.get('preview', '')[:38]})"
        elif ev.kind == "call_end":
            line = f"✓ done     {d['tool']} ({d.get('ms', 0) / 1000:.1f}s)"
        elif ev.kind in ("stream_begin", "stream_end", "exec_begin", "exec_end"):
            line = f"── {ev.kind} ──"
        elif ev.kind == "side_done":
            line = f"■ finished in {d['wall']:.1f}s"
        if line:
            with lock:
                # events can fire ON the exec thread, whose sys.stdout is the
                # REPL's capture buffer — write to the real stream instead
                import sys

                print(f"{dt:6.1f}s {tag} {line}", file=sys.__stdout__, flush=True)

    print(f"you: {QUERY}\n", flush=True)
    threads, results, t0 = start_race(speed, on_event)
    t0_holder["t0"] = t0
    for th in threads:
        th.join()
    spec, serial = results["spec"], results["serial"]
    print(flush=True)
    print(
        f"same code, same latencies · spec {spec.wall:.1f}s vs serial "
        f"{serial.wall:.1f}s · speculation saved {serial.wall - spec.wall:.1f}s "
        f"({serial.wall / max(spec.wall, 0.01):.2f}x) · "
        f"{spec.hits}/{spec.dispatched} claims, {spec.misses} misses",
        flush=True,
    )
    assert spec.final == serial.final, (spec.final, serial.final)
    print("\n--- final answer (identical on both sides) ---", flush=True)
    print(str(spec.final), flush=True)


# ------------------------------------------------------------------ the app
def build_tui(speed: float, auto_exit: float = 0.0, manual: bool = False):
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Static

    from demo.ui import PALETTE, UI_CSS, ChatBox, Column, apply_side_event

    class RaceApp(App):
        CSS = (
            """
        Screen { background: $background; }
        #topbar { height: 1; padding: 0 2; }
        #topleft { width: 1fr; }
        #topright { width: auto; }
        #cols { height: 1fr; }
        #verdict { height: 1; padding: 0 2; }
        """
            + UI_CSS
        )
        BINDINGS = [("q", "quit", "quit"), ("r", "restart", "restart")]

        def __init__(self) -> None:
            super().__init__()
            self.title = "spec-ptc · race"
            self.cols = {"spec": Column("spec"), "serial": Column("serial")}
            self.chat = ChatBox(QUERY, self._start, cps=45.0 * speed, manual=manual)
            self.results = None
            self._race_active = False
            self._evq: deque = deque()

        def compose(self) -> ComposeResult:
            with Horizontal(id="topbar"):
                left = Text()
                left.append("spec-ptc ", style=f"bold {PALETTE['text']}")
                left.append("· speculative tool calls, side by side", style=PALETTE["chrome"])
                yield Static(left, id="topleft")
                yield Static(
                    Text(f"{speed:g}× · q quit · r rerun", style=PALETTE["dim"]),
                    id="topright",
                )
            yield self.chat
            with Horizontal(id="cols"):
                yield self.cols["spec"]
                yield self.cols["serial"]
            yield Static("", id="verdict")

        def on_mount(self) -> None:
            self.set_interval(0.05, self._pump)
            self.set_interval(0.1, self._tick)

        def on_key(self, event) -> None:
            # manual mode: the presenter's keystrokes write the query
            if manual and not self.chat.sent:
                self.chat.keypress(event.key)
                event.stop()

        def check_action(self, action: str, parameters) -> bool | None:
            # while the presenter is typing, q/r are just keystrokes
            if manual and not self.chat.sent and action in ("quit", "restart"):
                return False
            return True

        def _start(self) -> None:
            self._race_active = True
            _, self.results, _ = start_race(speed, self._on_event)

        def action_restart(self) -> None:
            if self._race_active:
                return
            for col in self.cols.values():
                col.reset()
            self.chat.restart()

        def _on_event(self, side: str, ev: SpecEvent) -> None:
            # worker threads only enqueue; the UI drains on a timer. (Calling
            # call_from_thread per event hammered the app pump hard enough to
            # kill it mid-run — a real-terminal-only Textual race.)
            self._evq.append((side, ev))

        def _pump(self) -> None:
            while self._evq:
                side, ev = self._evq.popleft()
                self._apply(side, ev)

        def _apply(self, side: str, ev: SpecEvent) -> None:
            # a single bad event must never take down the TUI (Textual kills
            # the app on any handler exception) — surface it instead
            try:
                self._apply_inner(side, ev)
            except Exception as e:
                self._ui_error = f"{type(e).__name__}: {e}"

        def _apply_inner(self, side: str, ev: SpecEvent) -> None:
            col = self.cols[side]
            kind = apply_side_event(col, ev, solve_dp_preview="risk lattice 6×6, horizon=12")
            if kind == "side_done":
                d = ev.data
                col.wall = d["wall"]
                col.phase = "done"
                if d.get("final"):
                    col.show_answer(d["final"], d["wall"])
                # NB: never name this _running — that shadows textual.App._running,
                # which gates App._display (screen freezes at race end)
                self._race_active = not all(c.wall is not None for c in self.cols.values())

        def _tick(self) -> None:
            try:
                self._tick_inner()
            except Exception as e:
                self._ui_error = f"{type(e).__name__}: {e}"

        def _tick_inner(self) -> None:
            for col in self.cols.values():
                col.tick()
            spec, serial = self.cols["spec"], self.cols["serial"]
            if auto_exit and spec.wall is not None and serial.wall is not None:
                self._done_at = getattr(self, "_done_at", time.perf_counter())
                if time.perf_counter() - self._done_at > auto_exit:
                    self.exit()
            t = Text()
            if spec.wall is not None and serial.wall is not None:
                saved = serial.wall - spec.wall
                r = self.results or {}
                hits = r["spec"].hits if r else 0
                disp = r["spec"].dispatched if r else 0
                t.append("⚡ same code, same latencies — ", style=PALETTE["chrome"])
                t.append(
                    f"speculation answered {saved:.1f}s sooner "
                    f"({serial.wall / max(spec.wall, 0.01):.2f}× faster)",
                    style=f"bold {PALETTE['ok']}",
                )
                t.append(f" · {hits}/{disp} claims", style=PALETTE["chrome"])
                t.append(" · r to rerun", style=PALETTE["dim"])
            elif spec.wall is not None:
                t.append("● speculative side has answered — ", style=PALETTE["spec"])
                t.append("serial is still paying for its calls…", style=PALETTE["chrome"])
            elif not self.chat.sent:
                if manual and self.chat.n == 0:
                    t.append(
                        "start typing — your keystrokes write the question "
                        "(enter completes it, ctrl+c quits)",
                        style=PALETTE["dim"],
                    )
                else:
                    t.append(
                        "a user asks for a health read over a long log…", style=PALETTE["dim"]
                    )
            else:
                t.append(
                    "both sides stream the same tokens at the same rate; "
                    "watch when each one can start its tool calls",
                    style=PALETTE["dim"],
                )
            if getattr(self, "_ui_error", None):
                t.append(f"  ⚠ {self._ui_error}", style=PALETTE["err"])
            self.query_one("#verdict", Static).update(t)

    return RaceApp()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true", help="print the merged event timeline")
    ap.add_argument("--speed", type=float, default=1.0, help="time-scale (2.0 = twice as fast)")
    ap.add_argument(
        "--auto-exit",
        type=float,
        default=0.0,
        help="exit N seconds after both sides finish (for recordings)",
    )
    ap.add_argument(
        "--auto-type",
        action="store_true",
        help="the query types itself (for unattended recordings); by default "
        "the app starts empty and your keystrokes write the question",
    )
    args = ap.parse_args()
    if args.headless:
        run_headless(args.speed)
    else:
        manual = not (args.auto_type or args.auto_exit)
        build_tui(args.speed, auto_exit=args.auto_exit, manual=manual).run()


if __name__ == "__main__":
    main()
