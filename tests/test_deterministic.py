"""deterministic=True: identical calls route to one cached run (first
invocation is enough); default False keeps one independent run per call."""

import itertools
import time

from spec_ptc import Speculator


def run(code, deterministic):
    spec = Speculator()
    raw = []
    seq = itertools.count(1)

    @spec.tool(speculatable=True, pure=True, deterministic=deterministic)
    def llm_query(p):
        n = next(seq)
        raw.append(p)
        time.sleep(0.02)
        return f"R<{p}>#{n}"

    events = []
    spec.bus.subscribe(lambda ev: events.append(ev.kind))
    ns = {}
    ns.update(spec.hooks())
    stream = "```repl\n" + code + "```\n"
    with spec.turn(repl_locals=ns) as t:
        for i in range(0, len(stream), 6):
            t.feed(stream[i : i + 6])
    exec(code, ns)
    spec.close()
    return ns, raw, events


CODE = 'out = [llm_query("capital of France") for _ in range(3)]\n'


def test_deterministic_shares_one_run():
    ns, raw, events = run(CODE, deterministic=True)
    assert len(raw) == 1  # first invocation is enough
    assert events.count("dispatch") == 1
    assert ns["out"] == ["R<capital of France>#1"] * 3  # all claims hit the same run


def test_default_keeps_independent_samples():
    ns, raw, events = run(CODE, deterministic=False)
    assert len(raw) == 3  # one real run per call
    assert events.count("dispatch") == 3
    assert len(set(ns["out"])) == 3  # distinct samples, FIFO order
