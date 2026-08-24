"""SpeculativeLocalREPL against real RLM LocalREPL machinery (mocked sub-LM)."""

import time


def slow_mock_llm(prompt: str, model=None) -> str:
    time.sleep(0.4)
    return f"ans({str(prompt)[:24]})"


CODE = (
    "parts = []\n"
    "for i in range(6):\n"
    "    parts.append(llm_query('chunk ' + str(i) + ': ' + context[:40]))\n"
    "\n"
    "combined = '|'.join(str(p) for p in parts)\n"
    "answer['content'] = combined\n"
    "answer['ready'] = True"
)
RESPONSE = "Mapping chunks now.\n```repl\n" + CODE + "\n```\nDone."


def test_speculative_local_repl_end_to_end():
    from demo.rlm import SpeculativeLocalREPL

    repl = SpeculativeLocalREPL(
        context_payload="hello world " * 20, subcall_override=slow_mock_llm
    )
    # --- streaming phase (as a streaming client would drive it)
    repl.begin_stream_turn()
    t0 = time.perf_counter()
    for i in range(0, len(RESPONSE), 9):
        repl.feed(RESPONSE[i : i + 9])
        time.sleep(0.005)
    repl.end_stream_turn()
    # --- real execution through stock RLM execute_code
    result = repl.execute_code(CODE)
    wall = time.perf_counter() - t0
    assert result.final_answer and result.final_answer.count("|") == 5
    hits = sum(1 for e in repl.bus.history if e.kind == "claim_hit")
    assert hits == 6
    assert wall < 6 * 0.4, f"no fan-out: {wall:.2f}s"  # serial would be 2.4s+


def test_adapter_auto_speculates_without_streaming():
    """Stock RLM usage (no streaming API at all) still gets the fan-out."""
    from demo.rlm import SpeculativeLocalREPL

    repl = SpeculativeLocalREPL(context_payload="hello " * 30, subcall_override=slow_mock_llm)
    t0 = time.time()
    result = repl.execute_code(
        "xs = [llm_query('auto ' + str(i)) for i in range(5)]\n"
        "answer['content'] = '|'.join(str(x) for x in xs)\nanswer['ready'] = True"
    )
    wall = time.time() - t0
    assert result.final_answer.count("|") == 4
    hits = sum(1 for e in repl.bus.history if e.kind == "claim_hit")
    assert hits == 5
    assert wall < 5 * 0.4, f"auto prepass did not fan out: {wall:.2f}s"


def test_adapter_miss_path_matches_stock():
    from demo.rlm import SpeculativeLocalREPL

    repl = SpeculativeLocalREPL(
        context_payload="ctx", subcall_override=slow_mock_llm, auto_speculate=False
    )
    # speculation off, no streaming -> every call is a miss -> stock serial behavior
    result = repl.execute_code(
        "x = llm_query('direct')\nanswer['content'] = str(x)\nanswer['ready'] = True"
    )
    assert result.final_answer == "ans(direct)"
    misses = sum(1 for e in repl.bus.history if e.kind == "claim_miss")
    assert misses == 1
