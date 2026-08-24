"""Taint skip: a statement reading a NonSpeculated marker is skipped (its
targets poisoned) instead of aborting the shadow — later independent calls
keep speculating. EXAMPLE.md Case 5."""

import time

from spec_ptc import Speculator

CASE5 = (
    'rows = fetch_db("top errors")\n'
    'a = llm_query("triage: " + doc)\n'
    'b = llm_query("explain: " + str(rows))\n'  # uses the marker
    'c = llm_query("summarize: " + str(a))\n'  # independent of it
)


def run_case5(taint_skip):
    spec = Speculator(taint_skip=taint_skip)
    raw = []

    @spec.tool(speculatable=True, pure=True)
    def llm_query(p):
        raw.append(p)
        time.sleep(0.03)
        return "R<" + p[:16] + ">"

    @spec.tool()
    def fetch_db(q):
        raw.append("DB:" + q)
        return "rows(" + q + ")"

    events = []
    spec.bus.subscribe(lambda ev: events.append((ev.kind, dict(ev.data))))
    ns = {"doc": "the incident report"}
    ns.update(spec.hooks())

    stream = "```repl\n" + CASE5 + "```\n"
    with spec.turn(repl_locals=ns) as t:
        for i in range(0, len(stream), 6):
            t.feed(stream[i : i + 6])
    exec(CASE5, ns)
    spec.close()
    kinds = [k for k, _ in events]
    return ns, raw, events, kinds


def test_skip_recovers_independent_call_after_poison():
    ns, raw, events, kinds = run_case5(taint_skip=True)
    assert kinds.count("dispatch") == 2  # a and c
    assert kinds.count("claim_hit") == 2
    assert kinds.count("claim_miss") == 1  # only b
    assert kinds.count("shadow_stop") == 0
    skips = [d for k, d in events if k == "shadow_skip"]
    assert [s["index"] for s in skips] == [2]  # exactly the poisoned statement
    assert raw.count("DB:top errors") == 1  # side effect ran once, at real exec
    # real results byte-identical to baseline
    assert ns["b"] == "R<explain: rows(to>"
    assert ns["c"] == "R<summarize: R<tri>"


def test_flag_off_preserves_abort_behavior():
    ns, raw, events, kinds = run_case5(taint_skip=False)
    assert kinds.count("dispatch") == 1  # only a
    assert kinds.count("claim_hit") == 1
    assert kinds.count("claim_miss") == 2  # b and c
    assert kinds.count("shadow_stop") == 1
    assert kinds.count("shadow_skip") == 0
    assert ns["c"] == "R<summarize: R<tri>"


def test_taint_propagates_through_plain_assignment():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    def llm_query(p):
        return "R<" + p[:12] + ">"

    @spec.tool()
    def get_token():
        return "tok123"

    events = []
    spec.bus.subscribe(lambda ev: events.append(ev.kind))
    ns = {}
    ns.update(spec.hooks())
    code = (
        "token = get_token()\n"
        "alias = token\n"  # poison flows through the assignment
        'b = llm_query("auth " + str(alias))\n'
        'd = llm_query("independent question")\n'
    )
    stream = "```repl\n" + code + "```\n"
    with spec.turn(repl_locals=ns) as t:
        for i in range(0, len(stream), 6):
            t.feed(stream[i : i + 6])
    exec(code, ns)
    spec.close()
    assert events.count("shadow_skip") == 2  # the alias stmt and b's stmt
    assert events.count("claim_hit") == 1  # d
    assert events.count("claim_miss") == 1  # b
    assert ns["b"] == "R<auth tok123>"
