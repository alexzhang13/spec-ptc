"""Streaming-level speculation: peeks fire before statements close, and are
adopted (never duplicated) when the shadow reaches the closed statement."""

import time

from spec_ptc.contracts.events import EventBus
from spec_ptc.engine.streaming import plan_peeks, repair_tail, safe_eval
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

T = MockTiming(main_tok_per_s=2000, sub_base_s=0.3, sub_jitter_s=0.2, sub_tokens=3)


def drive(lines, mode="spec", context="c" * 400, hold_after=None, hold_s=0.35):
    """Feed lines one at a time; optionally pause after a given line index
    (simulating the model 'still writing') and snapshot dispatch count."""
    eng = MockLM(T)
    bus = EventBus()
    h = Harness(eng, mode, bus=bus, context=context)
    snapshot = {}

    def stream():
        for i, line in enumerate(lines):
            yield line + "\n"
            if hold_after is not None and i == hold_after:
                time.sleep(hold_s)
                snapshot["dispatched"] = sum(1 for e in bus.history if e.kind == "dispatch")

    out = h.run_turn(stream())
    h.launcher.shutdown()
    return out, bus, snapshot


def test_loop_unrolled_before_it_closes():
    lines = [
        "```repl",
        "chunks = ['a1', 'b2', 'c3', 'd4']",
        "results = []",
        "for c in chunks:",
        "    results.append(llm_query('sum: ' + c))",  # <- hold here
        "",
        "final = '|'.join(str(r) for r in results)",
        "answer['content'] = final",
        "answer['ready'] = True",
        "```",
    ]
    out, bus, snap = drive(lines, hold_after=4)
    # all 4 iterations dispatched while the loop statement was still OPEN
    assert snap["dispatched"] == 4, f"peek missed: {snap}"
    m = out.metrics
    assert m.hits == 4 and m.misses == 0
    # adoption, not duplication: exactly 4 dispatches total, none wasted
    assert m.dispatched == 4
    adopts = sum(1 for e in bus.history if e.kind == "adopt")
    assert adopts == 4
    assert out.final_answer.count("|") == 3


def test_peek_variable_args_from_live_state():
    lines = [
        "```repl",
        "topic = 'gpu clusters'",
        "prefix = 'Tell me about '",
        "r = llm_query(prefix + topic)",  # simple stmt: closes fast
        "s = llm_query(f'more on {topic}, please')",  # <- hold: f-string peek
        "combined = str(r) + str(s)",
        "answer['content'] = combined",
        "answer['ready'] = True",
        "```",
    ]
    out, bus, snap = drive(lines, hold_after=4)
    assert snap["dispatched"] == 2
    assert out.metrics.hits == 2 and out.metrics.dispatched == 2


def test_identical_calls_in_unrolled_loop_stay_independent():
    lines = [
        "```repl",
        "votes = []",
        "for _ in range(3):",
        "    votes.append(llm_query('same prompt'))",
        "",
        "answer['content'] = '|'.join(str(v) for v in votes)",
        "answer['ready'] = True",
        "```",
    ]
    out, bus, _ = drive(lines)
    assert out.metrics.dispatched == 3  # tally-based top-up, no dedup collapse
    assert out.metrics.hits == 3
    parts = out.final_answer.split("|")
    assert len(set(parts)) == 3  # independent samples survived


def test_peek_skips_conditionals_and_stale_names():
    lines = [
        "```repl",
        "flag = False",
        "if flag:",
        "    a = llm_query('should never fire')",
        "",
        "x = 'new'",
        "b = llm_query('uses ' + x)",
        "answer['content'] = str(b)",
        "answer['ready'] = True",
        "```",
    ]
    out, bus, _ = drive(lines)
    m = out.metrics
    assert m.dispatched == 1 and m.evictions == 0, (
        "conditional call must not be peeked; stale-name rail must hold"
    )
    assert out.metrics.hits == 1


def test_repair_and_safe_eval_units():
    import ast

    assert repair_tail("xs = [1, 2") is not None
    assert repair_tail("for c in chunks:") is not None
    env = {"chunks": ["a", "b"], "n": 3}
    v = safe_eval(ast.parse("[('x: ' + c) for c in chunks]", mode="eval").body, env)
    assert v == ["x: a", "x: b"]
    v2 = safe_eval(ast.parse("f'n is {n}'", mode="eval").body, env)
    assert v2 == "n is 3"
    plans = plan_peeks(
        "for c in chunks:\n    r.append(llm_query('q: ' + c))",
        {"llm_query"},
        {"chunks": ["u", "v"], "r": []},
    )
    assert [p.args for p in plans] == [("q: u",), ("q: v",)]
