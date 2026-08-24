"""The speculated / non-speculated separation and the modular Tool interface
(tool_api.py, TOOLS.md)."""

import time

import pytest

from spec_ptc.contracts.events import EventBus
from spec_ptc.contracts.tools import Tool
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

FAST = MockTiming(main_tok_per_s=2000, sub_base_s=0.15, sub_jitter_s=0.05, sub_tokens=2)


class EngineWith:
    """MockLM plus extra tools registered per test."""

    def __init__(self, extra):
        self.mock = MockLM(FAST)
        self.extra = extra

    def stream_main(self, script):
        return self.mock.stream_main(script)

    def make_tools(self, reg, bus):
        self.mock.make_tools(reg, bus)
        for item in self.extra:
            if isinstance(item, Tool):
                reg.register_tool(item)
            else:
                reg.register(**item)


def run(script, extra=(), mode="spec"):
    eng = EngineWith(list(extra))
    bus = EventBus()
    h = Harness(eng, mode, bus=bus, context="c" * 300)
    out = h.run_turn(eng.stream_main(script))
    h.launcher.shutdown()
    disp = [e.data for e in bus.history if e.kind == "dispatch"]
    return out, bus, disp


def blk(code):
    return f"go\n```repl\n{code}\n```\n"


# ------------------------------------------------------------- separation
def test_unmarked_tool_is_never_speculated_but_real_run_works():
    calls = []

    def send_report(text):
        calls.append(text)  # side effect: must NOT run early
        return "sent"

    code = (
        "s = llm_query('summarize: ' + context[:40])\n"
        "receipt = send_report('report body')\n"
        "answer['content'] = str(s) + '|' + receipt\nanswer['ready'] = True"
    )
    out, bus, disp = run(
        blk(code), extra=[{"name": "send_report", "fn": send_report}]
    )  # default: speculatable=False
    assert calls == ["report body"], "side effect must run exactly once (real run)"
    assert all(d["tool"] != "send_report" for d in disp)
    assert out.final_answer.endswith("|sent")
    hits = sum(1 for e in bus.history if e.kind == "claim_hit")
    assert hits == 1  # llm_query still speculated


def test_unused_nonspec_result_does_not_kill_speculation():
    """A fire-and-forget non-spec call mid-block must not opaque-abort:
    later speculatable calls still hit (the NonSpeculated marker is inert
    until USED)."""
    code = (
        "a = llm_query('first: ' + context[:30])\n"
        "log_metric('checkpoint')\n"  # result unused
        "b = llm_query('second: ' + str(a))\n"
        "answer['content'] = str(b)\nanswer['ready'] = True"
    )
    out, bus, _ = run(blk(code), extra=[{"name": "log_metric", "fn": lambda t: None}])
    hits = sum(1 for e in bus.history if e.kind == "claim_hit")
    assert hits == 2, "speculation must survive an unused non-spec call"


def test_used_nonspec_result_aborts_at_use_not_before():
    code = (
        "a = llm_query('early: ' + context[:30])\n"  # dispatched before the use
        "token = get_token()\n"
        "b = llm_query('auth ' + str(token))\n"  # USES marker -> abort here
        "answer['content'] = str(a) + str(b)\nanswer['ready'] = True"
    )
    out, bus, _ = run(blk(code), extra=[{"name": "get_token", "fn": lambda: "tok123"}])
    hits = sum(1 for e in bus.history if e.kind == "claim_hit")
    misses = sum(1 for e in bus.history if e.kind == "claim_miss")
    assert hits == 1 and misses == 1  # a hit; b missed
    assert "tok123" in out.final_answer


def test_speculatable_requires_pure():
    from spec_ptc.contracts.tools import ToolRegistry

    with pytest.raises(ValueError, match="requires pure=True"):
        ToolRegistry().register("bad", lambda: None, speculatable=True)


# ------------------------------------------------------------- Tool class
class KVTool(Tool):
    """get() speculates, set() must not — per-call gating via one namespace."""

    name = "kv"
    speculatable = True
    pure = True  # get is pure; set is gated off per-call

    def __init__(self):
        self.data = {"k1": "v1"}
        self.spec_runs = 0
        self.cancelled = []

    def execute(self, op, key, value=None):
        if op == "set":
            self.data[key] = value
            return "ok"
        time.sleep(0.15)
        return self.data.get(key.lower(), "?")  # case-insensitive store

    def speculative_execute(self, *a, _spec=None, **k):
        self.spec_runs += 1
        return self.execute(*a, **k)

    def cancel(self, spec):
        self.cancelled.append(spec.seq)

    def speculatable_call(self, args, kwargs):
        return args and args[0] == "get"

    def claim_key(self, args, kwargs):
        # normalize: key lookup is case-insensitive in this store
        return tuple(a.lower() if isinstance(a, str) else a for a in args)


def test_tool_class_gating_and_custom_key():
    kv = KVTool()
    code = (
        "v = kv('get', 'K1')\n"  # gated ON; claim_key lowercases
        "st = kv('set', 'k2', 'zzz')\n"  # gated OFF: never early
        "answer['content'] = str(v) + '|' + str(st)\nanswer['ready'] = True"
    )
    out, bus, disp = run(blk(code), extra=[kv])
    assert out.final_answer == "v1|ok"
    assert [d["tool"] for d in disp].count("kv") == 1  # only the get dispatched
    hits = sum(1 for e in bus.history if e.kind == "claim_hit" and e.data["tool"] == "kv")
    assert hits == 1, "custom claim_key must match 'K1' dispatch to 'k1'-era claim"
    assert kv.spec_runs == 1  # speculative_execute used
    assert kv.data.get("k2") == "zzz"  # set ran exactly once (real)


def test_tool_cancel_called_on_eviction():
    kv = KVTool()
    from spec_ptc.contracts.tools import ToolRegistry
    from spec_ptc.runtime.harness import REAL_BUILTINS
    from spec_ptc.speculator import SpecSession

    reg = ToolRegistry()
    reg.register_tool(kv)
    session = SpecSession(reg)
    turn = session.begin_stream_turn({"context": "x"}, REAL_BUILTINS)
    turn.feed(blk("v = kv('get', 'k1')"))
    turn.end(timeout=10)
    session.end_turn()  # abandoned: evict
    assert kv.cancelled, "Tool.cancel must run when its speculation is evicted"
    session.close()
