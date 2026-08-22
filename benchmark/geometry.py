"""Generation-geometry ("length-gen") sweeps: how the speedup depends on the."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

CTX = "alpha bravo charlie delta " * 200  # ~5k chars


def make_script(
    n_calls=8, pad_before=0, pad_after=0, prose_tail=0, crash_after_dispatch=False
) -> str:
    """A map-N block with tunable geometry. Pad lines are cheap arithmetic
    (~10 tokens each); prose_tail adds words after the closing fence."""
    pre = "\n".join(f"p{i} = ({i} * {i} + 17) % 101" for i in range(pad_before))
    post = "\n".join(f"q{i} = ({i} * 31 + 7) % 97" for i in range(pad_after))
    code = (
        (pre + "\n" if pre else "")
        + f"size = max(1, len(context) // {n_calls})\n"
        + f"chunks = [context[i:i+size] for i in range(0, len(context), size)][:{n_calls}]\n"
        + "rs = []\n"
        + "for c in chunks:\n"
        + "    rs.append(llm_query('sum: ' + c))\n"
        + ("\nboom = rs[9999]\n" if crash_after_dispatch else "")
        + (post + "\n" if post else "")
        + "\nanswer['content'] = '|'.join(str(r) for r in rs)\nanswer['ready'] = True"
    )
    tail = " ".join(f"word{i}" for i in range(prose_tail))
    return f"Working.\n```repl\n{code}\n```\n{tail}"


def run(script: str, mode: str, timing: MockTiming, max_inflight: int = 16) -> dict:
    eng = MockLM(timing)
    bus = EventBus()
    h = Harness(eng, mode, bus=bus, context=CTX, max_inflight=max_inflight)
    t0 = time.perf_counter()
    h.run_turn(eng.stream_main(script))
    wall = time.perf_counter() - t0
    hits = sum(1 for e in bus.history if e.kind == "claim_hit")
    wasted = sum(1 for s in h.store.all if s.state != "claimed")
    h.launcher.shutdown()
    return {"wall": round(wall, 2), "hits": hits, "wasted": wasted}


def point(
    name: str,
    script: str,
    timing: MockTiming,
    modes=("baseline", "lazy", "spec"),
    max_inflight: int = 16,
    **meta,
) -> list[dict]:
    rows = []
    base = None
    for mode in modes:
        r = run(script, mode, timing, max_inflight)
        r.update(axis=name, mode=mode, **meta)
        if mode == "baseline":
            base = r["wall"]
        r["speedup"] = round(base / r["wall"], 2) if base and r["wall"] else None
        rows.append(r)
        extra = f" hits={r['hits']:<3d} waste={r['wasted']}"
        sp = f" {r['speedup']}x" if mode != "baseline" and r["speedup"] else ""
        print(
            f"{name:<22s} {str(meta):<38s} {mode:9s} {r['wall']:7.2f}s{extra}{sp}", flush=True
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="rate,length,position,prose,latency,scale,waste")
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()
    axes = set(args.axes.split(","))
    T = lambda tps=120, lat=0.8: MockTiming(
        main_tok_per_s=tps, sub_base_s=lat, sub_jitter_s=lat / 4, sub_tokens=6
    )
    rows: list[dict] = []

    if "rate" in axes:  # frontier-slow .. draft-fast, fixed map-8 workload
        for tps in (30, 60, 120, 240, 480):
            rows += point("rate", make_script(prose_tail=120), T(tps), tps=tps)

    if "length" in axes:  # code AFTER the calls: the overlap tail
        for pad in (0, 40, 160):
            rows += point("length", make_script(pad_after=pad), T(60), pad_after=pad)

    if "position" in axes:  # calls late in a long block: worst-case lead time
        for pad in (0, 40, 160):
            rows += point("position", make_script(pad_before=pad), T(60), pad_before=pad)

    if "prose" in axes:  # model explains after the block: free overlap
        for words in (0, 150, 600):
            rows += point("prose", make_script(prose_tail=words), T(60), prose=words)

    if "latency" in axes:  # overhead floor .. long-call regime
        for lat in (0.05, 0.2, 0.8, 3.2):
            rows += point("latency", make_script(), T(120, lat), sub_lat=lat)

    if "scale" in axes:  # fan-out vs the concurrency budget
        for n, cap in ((2, 16), (8, 16), (32, 16), (32, 4)):
            rows += point(
                "scale", make_script(n_calls=n), T(120), max_inflight=cap, n_calls=n, cap=cap
            )

    if "waste" in axes:  # eviction storm: dispatch 16, claim none
        rows += point(
            "waste",
            make_script(n_calls=16, crash_after_dispatch=True),
            T(120),
            modes=("baseline", "spec"),
            crash=True,
        )

    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "sweeps.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {len(rows)} rows to {args.out}/sweeps.json")


if __name__ == "__main__":
    main()
