"""rlm_query is never speculated, but nothing breaks around it — llm_query
keeps speculating at every recursion depth (outer RLM → sub-RLM → sub-sub-RLM)."""

import time


def slow_llm(prompt: str, model=None) -> str:
    time.sleep(0.3)
    return f"ans({str(prompt)[:20]})"


D3_CODE = (
    "z = llm_query('d3 leaf question')\n"
    "answer['content'] = 'leaf:' + str(z)\nanswer['ready'] = True"
)
D2_CODE = (
    "xs = [llm_query('d2 chunk ' + str(i)) for i in range(4)]\n"
    "deeper = rlm_query('go to depth 3')\n"
    "answer['content'] = '|'.join(str(x) for x in xs) + '||' + str(deeper)\n"
    "answer['ready'] = True"
)
D1_CODE = (
    "a = llm_query('d1-a: ' + context[:20])\n"
    "deep = rlm_query('summarize everything below')\n"
    "note = 'depth1 saw: ' + str(deep)\n"  # uses the marker -> shadow skips it
    "b = llm_query('d1-b independent')\n"  # after the marker use -> must still speculate
    "answer['content'] = str(a) + '|' + note + '|' + str(b)\nanswer['ready'] = True"
)


def make_repl(depth, buses):
    from demo.rlm import SpeculativeLocalREPL

    repl = SpeculativeLocalREPL(context_payload="ctx " * 30, subcall_override=slow_llm)
    buses.append((depth, repl.bus))
    if depth < 3:

        def fake_rlm_query(prompt, model=None):
            inner = make_repl(depth + 1, buses)
            return inner.execute_code(D2_CODE if depth == 1 else D3_CODE).final_answer

        repl.globals["rlm_query"] = fake_rlm_query  # stand-in for the recursive client
    return repl


def kinds_of(bus):
    return [e.kind for e in bus.history]


def test_llm_query_speculates_at_every_depth():
    buses = []
    outer = make_repl(1, buses)
    # depth 1 driven through the streaming API, like a live RLM loop
    response = "thinking\n```repl\n" + D1_CODE + "\n```\n"
    outer.begin_stream_turn()
    for i in range(0, len(response), 8):
        outer.feed(response[i : i + 8])
    outer.end_stream_turn()
    result = outer.execute_code(D1_CODE)

    assert result.final_answer.startswith("ans(d1-a")
    assert "leaf:ans(d3" in result.final_answer and "ans(d2 chunk 0)" in result.final_answer

    per_depth = {d: kinds_of(b) for d, b in buses}
    assert len(per_depth) == 3  # three nested REPLs actually ran
    for d, ks in per_depth.items():
        assert ks.count("shadow_stop") == 0, f"depth {d} shadow aborted: {ks}"
    # depth 1: both llm_queries hit — including b, AFTER the rlm_query use
    assert per_depth[1].count("claim_hit") == 2 and per_depth[1].count("claim_miss") == 0
    assert per_depth[1].count("shadow_skip") >= 1  # the statement using `deep`
    # depth 2: the 4-wide fan-out speculated despite its own rlm_query below
    assert per_depth[2].count("claim_hit") == 4
    # depth 3: leaf call speculated by the auto pre-pass
    assert per_depth[3].count("claim_hit") == 1
