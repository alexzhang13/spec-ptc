"""LIVE side-by-side race: a real RLM solving one OOLONG trec-coarse 32k task,
twice — speculative vs serial — against seeded, temperature-0 vLLM endpoints
(Qwen3.5-27B for both the main model and the sub-calls; `just serve`).

The task's standard question types itself into the chat box; both columns then
run the same deterministic turn-by-turn RLM loop: turn 1 chunks the 787
questions and fans out one llm_query per chunk (label every question), turn 2
counts the labels in code and answers. The prompt prescribes this method, so
at temperature 0 both sides write identical code — the only difference is
WHEN each side's sub-calls run.

Run:  python -m demo.oolong_race            (TUI)
      python -m demo.oolong_race --headless (event timeline + verdict)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from demo.live import RLM_SYSTEM, instrument_inline_calls, rlm_turns
from spec_ptc.contracts.events import EventBus, SpecEvent
from spec_ptc.runtime.engines import engine_from_env
from spec_ptc.runtime.harness import Harness

TASK_PATH = Path(__file__).parent / "data" / "oolong_trec32k.json"
N_CHUNKS = 16
MAX_TURNS = 5


def load_task() -> dict:
    """One trec-coarse 32k task from oolong-synth, cached locally."""
    if TASK_PATH.exists():
        return json.load(open(TASK_PATH))
    from datasets import load_dataset

    val = load_dataset("oolongbench/oolong-synth", split="validation")
    r = next(r for r in val if r["dataset"] == "trec_coarse" and int(r["context_len"]) == 32768)
    task = {
        "task_id": f"trec32k-{r['id']}",
        "dataset": "trec_coarse_32768",
        "context": r["context_window_text"],
        "question": r["question"],
        "answer": str(r["answer"]),
        "answer_type": str(r["answer_type"]),
    }
    TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(task, open(TASK_PATH, "w"))
    return task


def gold_label(task: dict) -> str:
    import ast

    try:
        return str(ast.literal_eval(task["answer"])[0])
    except Exception:
        return task["answer"]


DEFS = (
    "abbreviation = asks what an abbreviation or acronym stands for, or how "
    "something is abbreviated; description and abstract concept = asks for a "
    "definition, an explanation, or a why/how; entity = a thing (animal, "
    "product, substance, color, food, work, ...); human being = a person or "
    "group of people; location = a place; numeric value = a count, date, "
    "price, distance, age, or other number"
)


def user_message(task: dict) -> str:
    return (
        f"Question: {task['question']}\n\n"
        f"(context is loaded; len(context) = {len(task['context'])} chars — "
        "one general-knowledge question per line)\n\n"
        "Solve it in exactly two REPL turns.\n"
        "Turn 1, one code block, statements in exactly this order:\n"
        "- qs = [l for l in context.split('\\n') if ' || Instance: ' in l]\n"
        f"- split qs into {N_CHUNKS} roughly equal chunks, each joined with newlines\n"
        "- for each chunk build this prompt: 'Label each question with exactly one "
        "of: location, numeric value, description and abstract concept, "
        f"abbreviation, human being, entity. Definitions: {DEFS}. Output ONLY the "
        "labels, one per line, in order. Questions:\\n' + chunk\n"
        "- labels_raw = llm_query_batched(prompts)\n"
        "- THEN, still in the same block, define the helper for next turn: "
        "def tally(outputs): lowercase and strip each output line, match it with "
        "startswith against the six label strings, and return a dict of totals\n"
        "- print(len(qs), 'questions in', len(prompts), 'chunks')\n"
        "After closing the block, explain the approach in 3-4 sentences while "
        "the calls run. Your first response must contain exactly ONE code block "
        "and must END after that explanation — do not tally or answer until you "
        "have seen the REPL output.\n"
        "Turn 2 (after you see the REPL output), keep it short:\n"
        "- totals = tally(labels_raw); print(totals)\n"
        "- answer['content'] = 'Label: ' + the most common label; "
        "answer['ready'] = True"
    )


# --------------------------------------------------------------------- runner
@dataclass
class SideResult:
    wall: float = 0.0
    final: str | None = None
    hits: int = 0
    misses: int = 0
    dispatched: int = 0
    turns: int = 0


def start_race(task: dict, on_event=None):
    """Two real RLM loops, spec vs baseline, same seeded temp-0 endpoints."""
    results: dict[str, SideResult] = {"spec": SideResult(), "serial": SideResult()}
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()
    gold = gold_label(task)

    for mode, side in (("spec", "spec"), ("baseline", "serial")):
        bus = EventBus()
        if on_event is not None:
            bus.subscribe(lambda ev, s=side: on_event(s, ev))
        eng = engine_from_env(
            bus=bus,
            main_temperature=0.0,
            sub_temperature=0.0,
            seed=0,
            sub_max_tokens=512,
            main_max_tokens=1200,
        )
        h = Harness(eng, mode, bus=bus, context=task["context"])
        if mode == "baseline":
            instrument_inline_calls(h, bus)
        messages = [
            {"role": "system", "content": RLM_SYSTEM},
            {"role": "user", "content": user_message(task)},
        ]

        def run(h=h, eng=eng, bus=bus, side=side, messages=messages):
            t_start = time.perf_counter()
            final = rlm_turns(h, eng, messages, max_turns=MAX_TURNS)
            r = results[side]
            r.wall = time.perf_counter() - t_start
            r.final = final
            m = [e for e in bus.history if e.kind == "turn_begin"]
            r.turns = len(m)
            r.hits = sum(1 for e in bus.history if e.kind == "claim_hit")
            r.misses = sum(1 for e in bus.history if e.kind == "claim_miss")
            r.dispatched = sum(1 for e in bus.history if e.kind == "dispatch")
            correct = bool(final) and gold.lower() in str(final).lower()
            bus.emit(
                "side_done",
                wall=r.wall,
                hits=r.hits,
                misses=r.misses,
                final=final or "(no answer)",
                correct=correct,
                gold=gold,
                turns=r.turns,
            )
            h.launcher.shutdown()

        threads.append(threading.Thread(target=run, daemon=True))

    for th in threads:
        th.start()
    return threads, results, t0


# ------------------------------------------------------------- headless twin
def run_headless() -> None:
    import sys

    task = load_task()
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
            line = f"▶ run      {d['tool']}{n}"
        elif ev.kind == "call_end":
            line = f"✓ done     {d['tool']} ({d.get('ms', 0) / 1000:.1f}s)"
        elif ev.kind in ("stream_begin", "stream_end", "exec_begin", "exec_end"):
            line = f"── {ev.kind} ──"
        elif ev.kind == "real_block_end" and d.get("stdout"):
            line = "stdout: " + d["stdout"][:100].replace("\n", " | ")
        elif ev.kind == "side_done":
            line = (
                f"■ finished in {d['wall']:.1f}s · {d['turns']} turns · "
                f"{'✓ CORRECT' if d['correct'] else '✗ wrong'} (gold: {d['gold']}) · "
                f"answer: {d['final'][:60]}"
            )
        if line:
            with lock:
                print(f"{dt:6.1f}s {tag} {line}", file=sys.__stdout__, flush=True)

    print(f"task: {task['task_id']}\nyou: {task['question']}\n", flush=True)
    threads, results, t0 = start_race(task, on_event)
    t0_holder["t0"] = t0
    for th in threads:
        th.join()
    spec, serial = results["spec"], results["serial"]
    print(flush=True)
    print(
        f"live RLM, deterministic endpoints · spec {spec.wall:.1f}s vs serial "
        f"{serial.wall:.1f}s · saved {serial.wall - spec.wall:.1f}s "
        f"({serial.wall / max(spec.wall, 0.01):.2f}x) · "
        f"{spec.hits}/{spec.dispatched} claims, {spec.misses} misses",
        flush=True,
    )
    print(f"spec:   {str(spec.final)[:120]}", flush=True)
    print(f"serial: {str(serial.final)[:120]}", flush=True)


# ------------------------------------------------------------------ the app
def build_tui():
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Static

    from demo.ui import PALETTE, UI_CSS, ChatBox, Column, apply_side_event

    task = load_task()

    class OolongRaceApp(App):
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
        BINDINGS = [("q", "quit", "quit")]

        def __init__(self) -> None:
            super().__init__()
            self.title = "spec-ptc · oolong"
            self.cols = {
                "spec": Column("spec", show_turns=True),
                "serial": Column("serial", show_turns=True),
            }
            self.chat = ChatBox(task["question"], self._start)
            self.done: dict[str, dict] = {}

        def compose(self) -> ComposeResult:
            with Horizontal(id="topbar"):
                left = Text()
                left.append("spec-ptc ", style=f"bold {PALETTE['text']}")
                left.append(
                    "· live RLM on OOLONG trec-coarse 32k · Qwen3.5-27B ×2, temp 0, seed 0",
                    style=PALETTE["chrome"],
                )
                yield Static(left, id="topleft")
                yield Static(Text("q quit", style=PALETTE["dim"]), id="topright")
            yield self.chat
            with Horizontal(id="cols"):
                yield self.cols["spec"]
                yield self.cols["serial"]
            yield Static("", id="verdict")

        def on_mount(self) -> None:
            self.set_interval(0.1, self._tick)

        def _start(self) -> None:
            start_race(task, self._on_event)

        def _on_event(self, side: str, ev: SpecEvent) -> None:
            if not self.is_running:
                return
            try:
                self.call_from_thread(self._apply, side, ev)
            except Exception:
                pass

        def _apply(self, side: str, ev: SpecEvent) -> None:
            try:
                self._apply_inner(side, ev)
            except Exception as e:
                self._ui_error = f"{type(e).__name__}: {e}"

        def _apply_inner(self, side: str, ev: SpecEvent) -> None:
            col = self.cols[side]
            kind = apply_side_event(col, ev)
            if kind == "side_done":
                d = ev.data
                col.wall = d["wall"]
                col.phase = "done"
                self.done[side] = d
                mark = " · ✓ correct" if d["correct"] else f" · ✗ gold: {d['gold']}"
                col.show_answer(d["final"], d["wall"], extra=mark)

        def _tick(self) -> None:
            try:
                self._tick_inner()
            except Exception as e:
                self._ui_error = f"{type(e).__name__}: {e}"

        def _tick_inner(self) -> None:
            for col in self.cols.values():
                col.tick()
            spec, serial = self.cols["spec"], self.cols["serial"]
            t = Text()
            if spec.wall is not None and serial.wall is not None:
                saved = serial.wall - spec.wall
                d = self.done.get("spec", {})
                t.append("⚡ real model, real task, same seed — ", style=PALETTE["chrome"])
                t.append(
                    f"speculation answered {saved:.1f}s sooner "
                    f"({serial.wall / max(spec.wall, 0.01):.2f}× faster)",
                    style=f"bold {PALETTE['ok']}",
                )
                t.append(
                    f" · {d.get('hits', 0)} claims · gold: {d.get('gold', '?')}",
                    style=PALETTE["chrome"],
                )
            elif spec.wall is not None:
                t.append("● speculative side has answered — ", style=PALETTE["spec"])
                t.append("serial is still paying for its calls…", style=PALETTE["chrome"])
            elif not self.chat.sent:
                t.append(
                    "one OOLONG trec-coarse question over 787 lines of 32k-token context…",
                    style=PALETTE["dim"],
                )
            else:
                t.append(
                    "same prompt, temperature 0, seeded server — both sides write "
                    "identical code; watch when each side's sub-calls start",
                    style=PALETTE["dim"],
                )
            if getattr(self, "_ui_error", None):
                t.append(f"  ⚠ {self._ui_error}", style=PALETTE["err"])
            self.query_one("#verdict", Static).update(t)

    return OolongRaceApp()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true", help="print the merged event timeline")
    args = ap.parse_args()
    if args.headless:
        run_headless()
    else:
        build_tui().run()


if __name__ == "__main__":
    main()
