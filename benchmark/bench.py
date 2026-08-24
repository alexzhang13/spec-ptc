"""Bench: catalog x modes x repeats -> table + JSON. The 'never slower' proof."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark.scenarios import CATALOG, Scenario
from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM
from spec_ptc.runtime.harness import MODES, Harness, collect


def run_scenario(sc: Scenario, mode: str, engine=None) -> dict:
    """mode may also be 'spec-nopeek' — spec mode with tail-peeking disabled,
    to isolate what the streaming-level peek optimizations contribute."""
    eng = engine or MockLM(sc.timing)
    bus = EventBus()
    peek = mode != "spec-nopeek"
    h = Harness(eng, mode.replace("-nopeek", ""), bus=bus, context=sc.context, peek=peek)
    t0 = time.perf_counter()
    hits = misses = evictions = dispatched = 0
    answer = None
    for turn_script in sc.turns:
        start = len(bus.history)
        out = h.run_turn(eng.stream_main(turn_script))
        m = collect(bus.history[start:], h.store, mode)
        hits += m.hits
        misses += m.misses
        evictions += m.evictions
        dispatched += m.dispatched
        if out.final_answer:
            answer = out.final_answer
    wall = time.perf_counter() - t0
    wasted = sum(1 for s in h.store.all if s.state != "claimed")
    h.launcher.shutdown()
    return {
        "scenario": sc.name,
        "category": sc.category,
        "mode": mode,
        "wall_s": round(wall, 3),
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "dispatched": dispatched,
        "wasted": wasted,
        "answered": answer is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()

    modes = args.modes.split(",")
    rows = []
    scenarios = [s for s in CATALOG if args.only in s.name]
    for sc in scenarios:
        per_mode: dict[str, list[float]] = {}
        base_row = None
        for mode in modes:
            walls, last = [], None
            for _ in range(args.repeats):
                last = run_scenario(sc, mode)
                walls.append(last["wall_s"])
            last["wall_s"] = round(statistics.median(walls), 3)
            per_mode[mode] = walls
            rows.append(last)
            if mode == "baseline":
                base_row = last
            speed = (
                f"  {base_row['wall_s'] / last['wall_s']:.2f}x"
                if base_row and mode != "baseline" and last["wall_s"] > 0
                else ""
            )
            print(
                f"{sc.name:24s} {mode:9s} {last['wall_s']:7.2f}s "
                f"hit={last['hits']:<3d} miss={last['misses']:<3d} "
                f"evict={last['evictions']:<2d} waste={last['wasted']:<3d}"
                f"{speed}",
                flush=True,
            )

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(rows, indent=1))

    # summary + regression guard
    print("\n== summary (median wall seconds, speedup vs baseline) ==")
    guard_ok = True
    md = [
        "| scenario | baseline | lazy | spec | spec speedup | hits | waste |",
        "|---|---|---|---|---|---|---|",
    ]
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["mode"]] = r
    for name, m in by.items():
        b, lz, s = (m.get(x) for x in ("baseline", "lazy", "spec"))
        if not (b and s):
            continue
        sp = b["wall_s"] / s["wall_s"] if s["wall_s"] else float("inf")
        cat = next(x.category for x in CATALOG if x.name == name)
        if cat == "adversarial" and s["wall_s"] > b["wall_s"] * 1.05 + 0.25:
            guard_ok = False
            print(f"  REGRESSION: {name} spec {s['wall_s']}s > baseline {b['wall_s']}s")
        md.append(
            f"| {name} | {b['wall_s']:.2f} | "
            + (f"{lz['wall_s']:.2f}" if lz else "–")
            + f" | {s['wall_s']:.2f} | {sp:.2f}x | {s['hits']} | {s['wasted']} |"
        )
        print(
            f"  {name:24s} base={b['wall_s']:7.2f} "
            + (f"lazy={lz['wall_s']:7.2f} " if lz else "")
            + f"spec={s['wall_s']:7.2f}  speedup={sp:5.2f}x"
        )
    (outdir / "results.md").write_text("\n".join(md))
    geo = [
        b["wall_s"] / s["wall_s"]
        for n, m in by.items()
        for b, s in [(m.get("baseline"), m.get("spec"))]
        if b and s and s["wall_s"]
    ]
    if geo:
        import math

        g = math.exp(sum(math.log(x) for x in geo) / len(geo))
        print(f"\ngeomean speedup (spec vs baseline, all {len(geo)} scenarios): {g:.2f}x")
    print("regression guard:", "OK" if guard_ok else "FAILED")
    if not guard_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
