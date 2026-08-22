"""Runtime cost of speculation FAILURES."""

from __future__ import annotations

import time

from spec_ptc.contracts.events import EventBus
from spec_ptc.engine.speculation import Budget
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

T = MockTiming(main_tok_per_s=300, sub_base_s=0.5, sub_jitter_s=0.2, sub_tokens=4)


def blk(code):
    return f"go\n```repl\n{code}\n```\ndone"


def run(script, mode, budget=256):
    eng = MockLM(T)
    bus = EventBus()
    h = Harness(eng, mode, bus=bus, context="x" * 800)
    h.launcher.budget = Budget(max_inflight=32, max_dispatches_per_turn=budget)
    t0 = time.perf_counter()
    out = h.run_turn(eng.stream_main(script))
    wall = time.perf_counter() - t0
    waste = sum(1 for s in h.store.all if s.state != "claimed")
    hits = sum(1 for e in bus.history if e.kind == "claim_hit")
    misses = sum(1 for e in bus.history if e.kind == "claim_miss")
    # time the eviction itself (already ran inside run_turn; measure a second
    # storm separately below)
    h.launcher.shutdown()
    return wall, hits, misses, waste, out


CASES = {
    # healthy reference: clean map-12
    "healthy-map-12": (
        "rs = []\n"
        "for i in range(12):\n"
        "    rs.append(llm_query('h' + str(i) + ': ' + context[:40]))\n"
        "\n"
        "answer['content'] = '|'.join(str(r) for r in rs)\nanswer['ready'] = True"
    ),
    # failure mode: jail abort after 1 call -> 11 misses
    "jail-abort-11-miss": (
        "a = llm_query('h0: ' + context[:40])\n"
        "import math\n"
        "rs = [a]\n"
        "for i in range(1, 12):\n"
        "    rs.append(llm_query('h' + str(i) + ': ' + context[:40]))\n"
        "\n"
        "answer['content'] = '|'.join(str(r) for r in rs)\nanswer['ready'] = True"
    ),
    # failure mode, scaled: mutation blind spot -> ~23 phantom dispatches
    "phantom-waste-storm-24": (
        "data = ['d' + str(i) for i in range(24)]\n"
        "rs = []\n"
        "for i in range(24):\n"
        "    rs.append(llm_query('q: ' + data[i]))\n"
        "    data[(i + 7) % 24] = 'mut' + str(i)\n"
        "\n"
        "answer['content'] = str(len(rs))\nanswer['ready'] = True"
    ),
    # failure mode, scaled: unroll 32, break after 4 -> 28 evicted mid-flight
    "overshoot-evict-28": (
        "rs = []\n"
        "for i in range(32):\n"
        "    rs.append(llm_query('s' + str(i) + ': ' + context[:40]))\n"
        "    if i == 3:\n"
        "        break\n"
        "\n"
        "answer['content'] = str(len(rs))\nanswer['ready'] = True"
    ),
    # failure mode: budget 4 of 12
    "budget-abort-8-miss": (
        "rs = []\n"
        "for i in range(12):\n"
        "    rs.append(llm_query('b' + str(i) + ': ' + context[:40]))\n"
        "\n"
        "answer['content'] = str(len(rs))\nanswer['ready'] = True"
    ),
}


def main() -> None:
    print(
        f"{'case':26s} {'baseline':>9s} {'spec':>9s} {'vs base':>8s} "
        f"{'hit':>4s} {'miss':>5s} {'waste':>6s}"
    )
    for name, code in CASES.items():
        budget = 4 if name.startswith("budget") else 256
        wb, *_ = run(blk(code), "baseline")
        ws, hits, misses, waste, out = run(blk(code), "spec", budget=budget)
        assert out.final_answer is not None
        print(
            f"{name:26s} {wb:8.2f}s {ws:8.2f}s {wb / ws:7.2f}x {hits:4d} {misses:5d} {waste:6d}"
        )

    # eviction latency: how long does turn_end take to kill 28 IN-FLIGHT calls?
    eng = MockLM(T)
    bus = EventBus()
    h = Harness(eng, "spec", bus=bus, context="x" * 800)
    from spec_ptc.engine.shadow import ShadowRunner
    from spec_ptc.engine.streaming import StreamSegmenter  # feed shadow only, never exec
    from spec_ptc.runtime.harness import REAL_BUILTINS
    from spec_ptc.speculator import StreamTurn

    sh = ShadowRunner(
        {"context": "x" * 800},
        h.shadow_hooks,
        h.store,
        REAL_BUILTINS,
        bus,
        launcher=h.launcher,
        registry=h.reg,
    )
    turn = StreamTurn(StreamSegmenter(), sh)
    turn.feed(blk("rs = [llm_query('kill me ' + str(i)) for i in range(28)]"))
    turn.end(timeout=0.2)  # calls still in flight
    t0 = time.perf_counter()
    n = h.store.evict_unclaimed("storm")
    dt = (time.perf_counter() - t0) * 1000
    print(
        f"\nevict {n} in-flight speculations: {dt:.2f}ms "
        f"({dt / max(1, n):.3f}ms each) — cancellation is not a wall cost"
    )
    h.launcher.shutdown()


if __name__ == "__main__":
    main()
