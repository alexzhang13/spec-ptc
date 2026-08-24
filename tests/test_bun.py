"""Bun REPL parity: fan-out through JS Proxy laziness + Atomics forcing."""

import shutil
import time

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bun") is None, reason="bun not installed")


def make_stack(latency=0.4):
    from spec_ptc.contracts.events import EventBus
    from spec_ptc.contracts.tools import ToolRegistry
    from spec_ptc.engine.speculation import Launcher, SpecStore

    bus = EventBus()
    reg = ToolRegistry()

    def mock_llm(prompt):
        time.sleep(latency)
        return f"ans({str(prompt)[:20]})"

    reg.register("llm_query", mock_llm, speculatable=True, pure=True)
    store = SpecStore(bus)
    return reg, store, Launcher(store, bus), bus


def test_bun_real_exec_miss_path():
    from plugins.bun.driver import BunREPL

    repl = BunREPL(*make_stack(0.05))
    try:
        m = repl.execute_code("ns.x = llm_query('hello'); print(String(ns.x));")
        assert not m["error"], m
        assert "ans(hello)" in m["stdout"]
    finally:
        repl.close()


def test_bun_shadow_fanout_then_claim():
    from plugins.bun.driver import BunREPL

    stack = make_stack(0.5)
    repl = BunREPL(*stack)
    bus = stack[3]
    try:
        # streaming phase: two closed statements fed to the shadow twin
        t0 = time.perf_counter()
        repl.shadow_feed(
            "ns.rs = [];\nfor (let i = 0; i < 4; i++) { ns.rs.push(llm_query('chunk ' + i)); }"
        )
        dispatch_done = time.perf_counter() - t0
        assert dispatch_done < 0.5, f"dispatch was not lazy: {dispatch_done:.2f}s"
        # real phase claims all four
        m = repl.execute_code(
            "ns.out = ns.rs ? 'twin' : 'real';\n"
            "ns.vals = [llm_query('chunk 0'), llm_query('chunk 1'),"
            " llm_query('chunk 2'), llm_query('chunk 3')];\n"
            "print(ns.vals.join('|'));"
        )
        wall = time.perf_counter() - t0
        assert not m["error"], m
        assert m["stdout"].count("ans(") == 4
        hits = sum(1 for e in bus.history if e.kind == "claim_hit")
        assert hits == 4
        assert wall < 4 * 0.5, f"no fan-out: {wall:.2f}s"
    finally:
        repl.close()
