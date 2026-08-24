"""Semantics: multiplicity, equivalence, segmentation, shadow containment, guards."""

import time

import pytest

from spec_ptc.engine.streaming import StreamSegmenter
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

FAST = MockTiming(main_tok_per_s=2000, sub_base_s=0.15, sub_jitter_s=0.1, sub_tokens=3)


def run(script, mode, context="x" * 400, timing=FAST):
    eng = MockLM(timing)
    h = Harness(eng, mode, context=context)
    t0 = time.perf_counter()
    out = h.run_turn(eng.stream_main(script))
    return out, time.perf_counter() - t0


def block(code):
    return f"text before\n```repl\n{code}\n```\ntext after"


# ---------------------------------------------------------------- equivalence
MAP_REDUCE = block(
    "chunks = [context[i:i+100] for i in range(0, len(context), 100)]\n"
    "results = []\n"
    "for c in chunks:\n"
    "    results.append(llm_query('summarize: ' + c))\n"
    "\n"
    "answer['content'] = llm_query('reduce: ' + '|'.join(str(r) for r in results))\n"
    "answer['ready'] = True"
)


@pytest.mark.parametrize("mode", ["baseline", "lazy", "spec"])
def test_map_reduce_answers(mode):
    out, _ = run(MAP_REDUCE, mode)
    assert out.final_answer and "mock-answer" in out.final_answer
    if mode != "baseline":
        m = out.metrics
        assert m.hits == 5 and m.misses == 0 and m.evictions == 0


def test_deterministic_content_equivalence():
    """With sample-ids stripped, all modes compute the same answer."""
    import re

    outs = [run(MAP_REDUCE, m)[0].final_answer for m in ("baseline", "lazy", "spec")]
    normalized = [re.sub(r"\[\w+#\d+\] ", "", o) for o in outs]
    assert normalized[0] == normalized[1] == normalized[2]


# ---------------------------------------------------------------- multiplicity
def test_multiplicity_independent_samples():
    code = block(
        "votes = [llm_query('vote now') for _ in range(4)]\n"
        "print('|'.join(str(v) for v in votes))"
    )
    out, _ = run(code, "spec")
    printed = out.results[0].stdout.strip().split("|")
    assert len(printed) == 4
    assert len(set(printed)) == 4, "identical calls must stay independent samples"
    assert out.metrics.hits == 4 and out.metrics.misses == 0


# ---------------------------------------------------------------- conditionals
def test_branch_resolved_by_shadow():
    code = block(
        "mode = 'summarize'\n"
        "if mode == 'summarize':\n"
        "    a = llm_query('branch taken: ' + mode)\n"
        "else:\n"
        "    a = llm_query('branch NOT taken')\n"
        "\n"
        "b = llm_query('depends on: ' + str(a))\n"
        "print(str(b))"
    )
    out, _ = run(code, "spec")
    m = out.metrics
    assert m.hits == 2 and m.misses == 0
    assert m.dispatched == 2, "untaken branch must not dispatch"


def test_branch_on_speculated_value():
    code = block(
        "v = llm_query('classify this')\n"
        "if str(v).startswith('['):\n"
        "    out = llm_query('yes path')\n"
        "else:\n"
        "    out = llm_query('no path')\n"
        "\n"
        "print(str(out))"
    )
    out, _ = run(code, "spec")
    assert out.metrics.hits == 2 and out.metrics.misses == 0


# ---------------------------------------------------------------- guards/jail
def test_rebind_evicts_and_aborts():
    code = block("a = llm_query('first')\nllm_query = None\nb = 1")
    out, _ = run(code, "spec")
    # shadow dispatched 'first' then hit the rebind: evicted, aborted; real run
    # errors on calling None — same as baseline would.
    assert out.metrics.evictions >= 1


def test_shadow_exception_leaves_real_run_intact():
    code = block("xs = [llm_query('p1'), llm_query('p2')]\nboom = xs[99]\nprint('never')")
    out, _ = run(code, "spec")
    assert "IndexError" in out.results[0].stderr  # real run fails identically
    ob, _ = run(code, "baseline")
    assert "IndexError" in ob.results[0].stderr


def test_pure_import_allowed_in_shadow():
    code = block(
        "import math\n"
        "x = math.floor(3.7)\n"
        "y = llm_query('after import: ' + str(x))\n"
        "print(str(y))"
    )
    out, _ = run(code, "spec")
    # pure stdlib imports are whitelisted in the shadow (FAILED.md #1a) -> hit
    assert "mock-answer" in out.results[0].stdout
    assert out.metrics.hits == 1 and out.metrics.misses == 0


def test_runaway_statement_does_not_hang_turn():
    code = block("n = 0\nwhile n < 10**7:\n    n += 1\n\nprint('done', n)")
    t0 = time.perf_counter()
    out, wall = run(code, "spec")
    assert "done" in out.results[0].stdout


# ---------------------------------------------------------------- never slower
def test_no_calls_overhead_tiny():
    code = block("total = sum(i * i for i in range(1000))\nprint(total)")
    _, t_base = run(code, "baseline")
    _, t_spec = run(code, "spec")
    assert t_spec < t_base + 0.25


# ---------------------------------------------------------------- segmenter
def test_segmenter_never_splits_multiline_constructs():
    src = (
        "```repl\n"
        "x = (1 +\n     2)\n"
        "s = '''multi\nline'''\n"
        "def f(a,\n      b):\n    return a + b\n"
        "z = f(1, 2)\n"
        "```\n"
    )
    seg = StreamSegmenter()
    got = []
    for i in range(0, len(src), 3):
        got += seg.feed(src[i : i + 3])
    got += seg.finish()
    sources = [g.source for g in got]
    assert any("1 +" in s and "2)" in s for s in sources)
    assert any("multi" in s and "line" in s for s in sources)
    assert any(s.startswith("def f") and "return" in s for s in sources)
    joined = "\n".join(sources)
    import ast

    ast.parse(joined)  # recombination is valid


def test_segmenter_streams_prefix_only_valid_statements():
    src = "```repl\nresults = []\nfor c in [1,2]:\n    results.append(c)\n\nprint(len(results))\n```\n"
    seg = StreamSegmenter()
    out = []
    for ch in src:
        for g in seg.feed(ch):
            import ast

            ast.parse(g.source)
            out.append(g)
    out += seg.finish()
    assert len(out) == 3


def test_prefill_char_budget_delays_but_never_drops():
    """EXPLOG EXP-2 mitigation: huge-prompt dispatches queue on the char
    budget instead of clogging; everything still completes and claims."""
    from spec_ptc.engine.speculation import Budget

    big = "z" * 30_000
    code = (
        "rs = [llm_query('A' + context), llm_query('B' + context), "
        "llm_query('C' + context), llm_query('D' + context)]\n"
        "answer['content'] = str(len(rs))\nanswer['ready'] = True"
    )
    eng = MockLM(FAST)
    from spec_ptc.contracts.events import EventBus

    bus = EventBus()
    h = Harness(eng, "spec", bus=bus, context=big)
    h.launcher.budget = Budget(
        max_inflight=16, max_dispatches_per_turn=64, max_inflight_chars=65_000
    )  # ~2 at a time
    out = h.run_turn(eng.stream_main(f"go\n```repl\n{code}\n```\n"))
    h.launcher.shutdown()
    hits = sum(1 for e in bus.history if e.kind == "claim_hit")
    assert out.final_answer == "4" and hits == 4
