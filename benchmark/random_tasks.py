"""Randomized task-family benchmark: distribution of speedups over many."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from demo.oolong import make_log_context
from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

FAMILIES = ("map", "two_stage", "chain", "vote", "branchy", "mixed")


def gen_task(seed: int) -> dict:
    rng = random.Random(seed)
    fam = rng.choice(FAMILIES)
    n = rng.randint(4, 16)
    k = rng.randint(2, 4)
    tag = f"t{seed}"
    timing = MockTiming(
        main_tok_per_s=rng.choice([80, 120, 180]),
        sub_base_s=rng.uniform(0.3, 0.9),
        sub_jitter_s=rng.uniform(0.1, 0.5),
        sub_tokens=6,
    )
    ctx = make_log_context(n_entries=rng.randint(80, 240), seed=seed)
    if fam == "map":
        code = (
            f"size = max(1, len(context) // {n})\n"
            f"chunks = [context[i:i+size] for i in range(0, len(context), size)][:{n}]\n"
            "rs = []\n"
            "for c in chunks:\n"
            f"    rs.append(llm_query('{tag} sum: ' + c))\n\n"
            f"answer['content'] = llm_query('{tag} reduce: ' + '|'.join(str(r) for r in rs))\n"
            "answer['ready'] = True"
        )
    elif fam == "two_stage":
        code = (
            f"size = max(1, len(context) // {n})\n"
            f"chunks = [context[i:i+size] for i in range(0, len(context), size)][:{n}]\n"
            "out = []\n"
            "for c in chunks:\n"
            f"    e = llm_query('{tag} extract: ' + c)\n"
            f"    out.append(llm_query('{tag} judge: ' + str(e)))\n\n"
            f"answer['content'] = '|'.join(str(o) for o in out)\n"
            "answer['ready'] = True"
        )
    elif fam == "chain":
        lines = [f"v0 = llm_query('{tag} step0: ' + context[:200])"]
        for i in range(1, k + 1):
            lines.append(f"v{i} = llm_query('{tag} step{i}: ' + str(v{i - 1}))")
        code = "\n".join(lines) + f"\nanswer['content'] = str(v{k})\nanswer['ready'] = True"
    elif fam == "vote":
        code = (
            f"votes = [llm_query('{tag} vote') for _ in range({n})]\n"
            "answer['content'] = '|'.join(str(v) for v in votes)\n"
            "answer['ready'] = True"
        )
    elif fam == "branchy":
        code = (
            f"cls = llm_query('{tag} classify: ' + context[:150])\n"
            "if 'a' in str(cls):\n"
            f"    r = llm_query('{tag} path A: ' + str(cls))\n"
            "else:\n"
            f"    r = llm_query('{tag} path B: ' + str(cls))\n\n"
            "answer['content'] = str(r)\n"
            "answer['ready'] = True"
        )
    else:  # mixed
        code = (
            f"size = max(1, len(context) // {n})\n"
            f"parts = [llm_query('{tag} p: ' + context[i:i+size]) "
            f"for i in range(0, len(context), size)][:{n}]\n"
            f"verdict = llm_query('{tag} verdict: ' + '|'.join(str(p) for p in parts))\n"
            "if len(str(verdict)) > 3:\n"
            f"    final = llm_query('{tag} final: ' + str(verdict))\n\n"
            "answer['content'] = str(final)\n"
            "answer['ready'] = True"
        )
    script = f"Working on task {tag} ({fam}).\n```repl\n{code}\n```\nDone."
    return {
        "seed": seed,
        "family": fam,
        "script": script,
        "code": code,
        "context": ctx,
        "timing": timing,
    }


# --------------------------------------------------------------- harness path
def run_harness(task: dict, mode: str) -> float:
    eng = MockLM(task["timing"])
    h = Harness(eng, mode, bus=EventBus(), context=task["context"])
    t0 = time.perf_counter()
    h.run_turn(eng.stream_main(task["script"]))
    wall = time.perf_counter() - t0
    h.launcher.shutdown()
    return wall


# --------------------------------------------------------------- RLM path
def run_rlm(task: dict, speculative: bool) -> float:
    """Through the real rlm LocalREPL machinery (stock vs our subclass)."""
    from rlm.environments.local_repl import LocalREPL

    from demo.rlm import SpeculativeLocalREPL

    timing: MockTiming = task["timing"]
    lm = MockLM(timing)

    def subcall(prompt, model=None):
        return lm.sub_call(str(prompt))

    t0 = time.perf_counter()
    if speculative:
        repl = SpeculativeLocalREPL(context_payload=task["context"], subcall_override=subcall)
        repl.begin_stream_turn()
        for delta in lm.stream_main(task["script"]):  # simulated model stream
            repl.feed(delta)
        repl.end_stream_turn()
        repl.execute_code(task["code"])
    else:
        repl = LocalREPL(context_payload=task["context"])
        repl.globals["llm_query"] = subcall  # stock serial behavior
        repl.globals["llm_query_batched"] = lambda ps, model=None: [subcall(p) for p in ps]
        for _ in lm.stream_main(task["script"]):  # same generation cost
            pass
        repl.execute_code(task["code"])
    wall = time.perf_counter() - t0
    repl.cleanup()
    return wall


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rlm-tasks", type=int, default=12)
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()

    rows = []
    speedups_all: list[float] = []
    by_family: dict[str, list[float]] = {}
    for seed in range(args.tasks):
        task = gen_task(seed)
        b = statistics.median(run_harness(task, "baseline") for _ in range(args.repeats))
        s = statistics.median(run_harness(task, "spec") for _ in range(args.repeats))
        sp = b / s if s else float("inf")
        speedups_all.append(sp)
        by_family.setdefault(task["family"], []).append(sp)
        rows.append(
            {
                "seed": seed,
                "family": task["family"],
                "baseline_s": round(b, 2),
                "spec_s": round(s, 2),
                "speedup": round(sp, 2),
                "path": "harness",
            }
        )
        print(
            f"seed={seed:<3d} {task['family']:<10s} base={b:6.2f}s spec={s:6.2f}s "
            f"-> {sp:5.2f}x",
            flush=True,
        )

    rlm_speedups = []
    for seed in range(args.rlm_tasks):
        task = gen_task(1000 + seed)
        b = run_rlm(task, speculative=False)
        s = run_rlm(task, speculative=True)
        sp = b / s if s else float("inf")
        rlm_speedups.append(sp)
        rows.append(
            {
                "seed": 1000 + seed,
                "family": task["family"],
                "baseline_s": round(b, 2),
                "spec_s": round(s, 2),
                "speedup": round(sp, 2),
                "path": "rlm-localrepl",
            }
        )
        print(
            f"RLM  {seed:<3d} {task['family']:<10s} base={b:6.2f}s spec={s:6.2f}s "
            f"-> {sp:5.2f}x",
            flush=True,
        )

    import math

    def geo(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0

    if speedups_all:
        print("\n== distribution over random tasks (harness path) ==")
        print(
            f"  n={len(speedups_all)}  geomean={geo(speedups_all):.2f}x  "
            f"median={statistics.median(speedups_all):.2f}x  "
            f"p10={pct(speedups_all, 10):.2f}x  p90={pct(speedups_all, 90):.2f}x"
        )
    for fam in sorted(by_family):
        xs = by_family[fam]
        print(
            f"  {fam:<10s} n={len(xs):<3d} geomean={geo(xs):.2f}x "
            f"median={statistics.median(xs):.2f}x"
        )
    if rlm_speedups:
        print(
            f"\n== through real RLM LocalREPL ==\n"
            f"  n={len(rlm_speedups)}  geomean={geo(rlm_speedups):.2f}x  "
            f"median={statistics.median(rlm_speedups):.2f}x  "
            f"p10={pct(rlm_speedups, 10):.2f}x  p90={pct(rlm_speedups, 90):.2f}x"
        )
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "random.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
