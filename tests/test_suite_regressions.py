"""Two engine bugs the pattern suite found (benchmark/suite): class statements
aborted the shadow, and poisoning a comprehension's loop variable leaked taint
into later independent statements."""

import time

from spec_ptc import Speculator


def _spec():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    def llm_query(p):
        time.sleep(0.02)
        return "R<" + str(p)[:12] + ">"

    return spec


def _run(spec, code, ns_extra=None):
    events = []
    spec.bus.subscribe(lambda ev: events.append((ev.kind, dict(ev.data))))
    ns = dict(ns_extra or {})
    ns.update(spec.hooks())
    stream = "```repl\n" + code + "```\n"
    with spec.turn(repl_locals=ns) as t:
        for i in range(0, len(stream), 6):
            t.feed(stream[i : i + 6])
    exec(code, ns)
    spec.close()
    return ns, events, [k for k, _ in events]


CLASS_CODE = (
    "class Tagger:\n"
    "    def __init__(self, xs):\n"
    "        self.xs = xs\n"
    "    def run(self):\n"
    "        return [llm_query('tag: ' + x) for x in self.xs]\n"
    "\n"
    "outs = Tagger(['a', 'b', 'c']).run()\n"
)


def test_class_statement_does_not_abort_the_shadow():
    ns, events, kinds = _run(_spec(), CLASS_CODE)
    assert kinds.count("shadow_stop") == 0, [d for k, d in events if k == "shadow_stop"]
    assert kinds.count("dispatch") == 3
    assert kinds.count("claim_hit") == 3
    assert ns["outs"] == ["R<tag: a>", "R<tag: b>", "R<tag: c>"]


COMP_CODE = (
    "tainted = [llm_query('x: ' + str(secret) + i) for i in ['1', '2']]\n"
    "clean = [llm_query('y: ' + i) for i in ['3', '4']]\n"
)


def test_comprehension_local_taint_does_not_leak_to_later_statements():
    spec = _spec()

    @spec.tool()
    def probe():
        return "S"

    ns, events, kinds = _run(spec, "secret = probe()\n" + COMP_CODE)
    skips = [d for k, d in events if k == "shadow_skip"]
    assert [s["index"] for s in skips] == [1]  # only the tainted comprehension
    assert kinds.count("dispatch") == 2  # the two clean calls
    assert kinds.count("claim_hit") == 2
    assert kinds.count("claim_miss") == 2  # the two tainted ones
    assert ns["clean"] == ["R<y: 3>", "R<y: 4>"]
