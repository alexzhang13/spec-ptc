"""Where speculation FAILS — misses (calls never speculated) and waste
(speculated incorrectly, never claimed). These tests assert the CURRENT
limitation on purpose: when we fix one, its test flips red and FAILED.md gets
updated. Correctness must hold in every case — failures degrade speed only.

Taxonomy (FAILED.md):
  MISS  — real run executes a call the speculative side never dispatched
  WASTE — speculative side dispatched a call the real run never claims
"""

import re
import time

from spec_ptc.contracts.events import EventBus
from spec_ptc.engine.speculation import Budget
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

FAST = MockTiming(main_tok_per_s=2000, sub_base_s=0.15, sub_jitter_s=0.1, sub_tokens=3)


def run(
    script, mode="spec", context="x" * 400, engine=None, max_inflight=16, budget_dispatches=64
):
    eng = engine or MockLM(FAST)
    bus = EventBus()
    h = Harness(eng, mode, bus=bus, context=context, max_inflight=max_inflight)
    h.launcher.budget = Budget(
        max_inflight=max_inflight, max_dispatches_per_turn=budget_dispatches
    )
    out = h.run_turn(eng.stream_main(script))
    h.launcher.shutdown()
    stats = {
        "hits": sum(1 for e in bus.history if e.kind == "claim_hit"),
        "misses": sum(1 for e in bus.history if e.kind == "claim_miss"),
        "dispatched": sum(1 for e in bus.history if e.kind == "dispatch"),
        "evicted": sum(1 for e in bus.history if e.kind == "evict"),
        "waste": sum(1 for s in h.store.all if s.state != "claimed"),
        "shadow_stopped": any(e.kind == "shadow_stop" for e in bus.history),
    }
    return out, stats


def blk(code, pre="go", post="done"):
    return f"{pre}\n```repl\n{code}\n```\n{post}"


def norm(s):
    return re.sub(r"\[\w+#\d+\] ", "", s or "")


# =====================================================================
# MISS class 1: jail-abort cascade — one blocked op silences the rest
# =====================================================================
def test_miss_jail_abort_silences_all_later_calls():
    """Non-whitelisted imports (socket/os/...) still opaque-abort: every call
    after the abort point is a MISS. Pure stdlib imports are whitelisted as of
    2026-08-22 (FAILED.md #1a); remaining target: dependency-aware resumption."""
    code = (
        "a = llm_query('before abort: ' + context[:40])\n"
        "import socket\n"  # shadow: blocked -> abort
        "b = llm_query('after abort 1')\n"
        "c = llm_query('after abort 2')\n"
        "answer['content'] = str(a) + str(b) + str(c) + str(socket is not None)\n"
        "answer['ready'] = True"
    )
    t0 = time.perf_counter()
    out, s = run(blk(code))
    spec_wall = time.perf_counter() - t0
    assert out.final_answer  # real run is fine (imports allowed there)
    assert s["shadow_stopped"]
    assert s["hits"] == 1 and s["misses"] == 2, s  # <- the failure we document
    t0 = time.perf_counter()
    base, _ = run(blk(code), mode="baseline")
    base_wall = time.perf_counter() - t0
    assert norm(out.final_answer) == norm(base.final_answer)
    # total speculation collapse must still not cost wall time (FAILED.md)
    assert spec_wall <= base_wall * 1.10 + 0.30, (spec_wall, base_wall)


# =====================================================================
# MISS class 2: un-deepcopyable state poisons the next turn's fork
# =====================================================================
def test_miss_opaque_namespace_value_from_prior_turn():
    """Turn 1 leaves a generator in the namespace (deepcopy fails ->
    Opaque). Turn 2's first statement touches it -> abort -> every call in
    turn 2 is a MISS. Iteration target: per-value copy fallbacks (tee
    generators? pickle round-trip?), or skip-only-tainted-statements."""
    eng = MockLM(FAST)
    bus = EventBus()
    h = Harness(eng, "spec", bus=bus, context="y" * 200)
    h.run_turn(eng.stream_main(blk("gen = (i * 2 for i in range(10))\nfirst = next(gen)")))
    n_before = len(bus.history)
    out = h.run_turn(
        eng.stream_main(
            blk(
                "vals = [str(next(gen)) for _ in range(3)]\n"
                "r = llm_query('gen says: ' + ','.join(vals))\n"
                "answer['content'] = str(r)\nanswer['ready'] = True"
            )
        )
    )
    h.launcher.shutdown()
    tail = bus.history[n_before:]
    misses = sum(1 for e in tail if e.kind == "claim_miss")
    assert out.final_answer and "mock-answer" in out.final_answer
    assert misses == 1, "expected the turn-2 call to MISS (Opaque abort)"


# =====================================================================
# MISS class 3: per-turn dispatch budget aborts the shadow mid-map
# =====================================================================
def test_miss_budget_exhaustion_aborts_remaining():
    """Budget of 3 dispatches, map of 6: calls 1-3 hit, the denial raises
    ShadowBudgetDenied -> abort -> calls 4-6 MISS. Iteration target: deny
    should stop *dispatching* without killing the shadow (later statements
    may still chain off already-dispatched values)."""
    code = (
        "rs = []\n"
        "for i in range(6):\n"
        "    rs.append(llm_query('b' + str(i) + ': ' + context[:30]))\n"
        "\n"
        "answer['content'] = '|'.join(str(r) for r in rs)\nanswer['ready'] = True"
    )
    out, s = run(blk(code), budget_dispatches=3)
    assert out.final_answer.count("|") == 5
    assert s["hits"] == 3 and s["misses"] == 3, s
    assert s["shadow_stopped"]


# =====================================================================
# MISS+WASTE class 4: sub-call fails while speculative
# =====================================================================
def test_failed_speculative_subcall_is_unclaimable_waste():
    """The engine 500s on its FIRST invocation of a prompt: the speculation
    fails (state='failed', unclaimable) -> real run MISSES and re-executes
    inline (which succeeds). Answer correct; cost = 1 wasted attempt + the
    call paid at exec time. Iteration target: retry-once in the launcher."""

    class FlakyEngine:
        def __init__(self):
            self.mock = MockLM(FAST)
            self.attempts = {}

        def stream_main(self, script):
            return self.mock.stream_main(script)

        def make_tools(self, reg, bus):
            def flaky(prompt, _spec=None):
                n = self.attempts.get(prompt, 0)
                self.attempts[prompt] = n + 1
                if n == 0:
                    raise RuntimeError("transient 500")
                return self.mock.sub_call(prompt)

            flaky.wants_spec = True
            reg.register("llm_query", flaky, speculatable=True, pure=True)
            reg.register(
                "llm_query_batched",
                lambda ps, **k: [flaky(p) for p in ps],
                speculatable=True,
                pure=True,
                batched=True,
            )

    code = (
        "r = llm_query('flaky: ' + context[:30])\n"
        "answer['content'] = str(r)\nanswer['ready'] = True"
    )
    out, s = run(blk(code), engine=FlakyEngine())
    assert out.final_answer and "mock-answer" in out.final_answer
    assert s["dispatched"] == 1 and s["misses"] == 1 and s["waste"] == 1, s


# =====================================================================
# WASTE class 5: peek's mutation blind spot (subscript store evades the
# stale-name rail: `data[i+1] = ...` does not mark `data` as assigned)
# =====================================================================
def test_peek_mutation_bets_are_retracted_mid_stream():
    """FIXED (was WASTE class 5): a loop body that mutates its own iterable
    through a subscript used to leave phantom peeks alive until turn end.
    Now: (a) the stale-name rail taints subscript/attribute stores and
    mutating method calls, so complete-body plans only bet on iteration 0;
    (b) BET RETRACTION — when the mutation line streams in and the re-plan
    no longer justifies earlier peeks, they are evicted immediately, while
    their sub-calls are barely started. Residual cost: ~one streamed line of
    sub-decode per phantom, not a full call."""
    code = (
        "data = ['d0', 'd1', 'd2', 'd3']\n"
        "rs = []\n"
        "for i in range(4):\n"
        "    rs.append(llm_query('q: ' + data[i]))\n"
        "    data[(i + 1) % 4] = 'CHANGED' + str(i)\n"
        "\n"
        "answer['content'] = '|'.join(str(r) for r in rs)\nanswer['ready'] = True"
    )
    out, s = run(blk(code))
    assert out.final_answer.count("|") == 3
    assert s["hits"] == 4 and s["misses"] == 0, s  # true args all claimed
    assert s["evicted"] == 3, s  # phantoms retracted...
    base, _ = run(blk(code), mode="baseline")
    assert norm(out.final_answer) == norm(base.final_answer)


def test_retraction_fires_before_real_execution():
    """Retraction is queued at the invalidating line and executes the first
    moment the shadow is free — worst case delayed by one in-progress
    force-wait, but always before real execution and always cancelling the
    phantoms while their sub-calls are still in flight."""
    from spec_ptc.contracts.events import EventBus
    from spec_ptc.runtime.engines import MockLM, MockTiming
    from spec_ptc.runtime.harness import Harness

    code = (
        "data = ['d0', 'd1', 'd2', 'd3']\n"
        "rs = []\n"
        "for i in range(4):\n"
        "    rs.append(llm_query('q: ' + data[i]))\n"
        "    data[(i + 1) % 4] = 'CHANGED' + str(i)\n"
        "\n"
        "answer['content'] = str(len(rs))\nanswer['ready'] = True"
    )
    # realistic stream rate so "mid-stream" is observable (at absurd test
    # rates the shadow drains the invalidating line just after stream_end)
    eng = MockLM(MockTiming(main_tok_per_s=60, sub_base_s=0.3, sub_jitter_s=0.1, sub_tokens=3))
    bus = EventBus()
    h = Harness(eng, "spec", bus=bus, context="x" * 400)
    h.run_turn(eng.stream_main(blk(code)))
    h.launcher.shutdown()
    t_exec = next(e.t for e in bus.history if e.kind == "exec_begin")
    retractions = [
        e for e in bus.history if e.kind == "evict" and e.data.get("reason") == "peek-retracted"
    ]
    assert len(retractions) == 3
    # the enforceable bound: retraction fires the first moment the shadow is
    # free after the invalidating line (worst case: after one force-wait),
    # ALWAYS before real execution — never left to the turn-end sweep
    assert all(e.t < t_exec for e in retractions)
    # and the phantoms were cancelled IN FLIGHT (never ready, never claimed)
    phantoms = [sp for sp in h.store.all if sp.state == "evicted"]
    assert len(phantoms) == 3
    assert all(sp.t_claim == 0.0 for sp in phantoms)


# =====================================================================
# WASTE class 6: data-independent early exit — unroll overshoots a break
# =====================================================================
def test_waste_unroll_overshoots_early_break():
    """`break` under a call-free condition doesn't stop the unroll: peek
    dispatches all 6 iterations, the real loop stops after 3 -> 3 WASTED.
    (When the break condition inspects the call result — the common case —
    it contains a Call node and peek already bails to zero waste.)
    Iteration target: cap unroll at 1 iteration when the body contains
    break/return/raise."""
    code = (
        "rs = []\n"
        "for i in range(6):\n"
        "    rs.append(llm_query('step ' + str(i) + ': ' + context[:20]))\n"
        "    if i == 2:\n"
        "        break\n"
        "\n"
        "answer['content'] = '|'.join(str(r) for r in rs)\nanswer['ready'] = True"
    )
    out, s = run(blk(code))
    assert out.final_answer.count("|") == 2
    assert s["hits"] == 3 and s["waste"] == 3, s


# =====================================================================
# WASTE class 7: abandoned turn — speculate, then nobody executes
# =====================================================================
def test_waste_abandoned_turn_evicts_everything():
    """A host feeds the stream but never runs the block (client abort,
    max-tokens cutoff policy, user cancel): every dispatch is WASTE, and
    turn_end must reclaim it all (eviction aborts in-flight sub-calls)."""
    from spec_ptc.contracts.tools import ToolRegistry
    from spec_ptc.runtime.harness import REAL_BUILTINS
    from spec_ptc.speculator import SpecSession

    eng = MockLM(FAST)
    reg = ToolRegistry()
    bus = EventBus()
    eng.make_tools(reg, bus)
    session = SpecSession(reg, bus)
    turn = session.begin_stream_turn({"context": "z" * 200}, REAL_BUILTINS)
    script = blk("rs = [llm_query('waste: ' + str(i)) for i in range(4)]")
    for i in range(0, len(script), 9):
        turn.feed(script[i : i + 9])
    turn.end(timeout=30)
    session.end_turn()  # no real execution happened
    dispatched = sum(1 for e in bus.history if e.kind == "dispatch")
    evicted = sum(1 for e in bus.history if e.kind == "evict")
    assert dispatched == 4 and evicted == 4, (dispatched, evicted)
    assert all(sp.state == "evicted" for sp in session.store.all)
    session.close()


# =====================================================================
# DEGENERATE class 8: hit with ~zero head start (not a miss, barely a win)
# =====================================================================
def test_degenerate_last_statement_call_has_no_overlap():
    """A single call in the block's LAST statement with nothing after it:
    speculation hits, but dispatch happens essentially at stream end — the
    head start is one statement-closure, worth ~a line of tokens. This is
    the honest floor of streaming value (lazy == spec here)."""
    code = (
        "answer['content'] = str(llm_query('only call: ' + context[:30]))\n"
        "answer['ready'] = True"
    )
    eng = MockLM(FAST)
    bus = EventBus()
    h = Harness(eng, "spec", bus=bus, context="w" * 200)
    h.run_turn(eng.stream_main(blk(code)))
    h.launcher.shutdown()
    t_dispatch = next(e.t for e in bus.history if e.kind == "dispatch")
    t_stream_end = next(e.t for e in bus.history if e.kind == "stream_end")
    head_start = t_stream_end - t_dispatch
    assert head_start < 0.25, f"expected near-zero head start, got {head_start:.2f}s"
