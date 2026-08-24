"""The Speculator facade + patch_rlm: the 'incredibly easy' paths."""

import time


def test_speculator_decorator_flow():
    from spec_ptc import Speculator

    spec = Speculator()
    effects = []

    @spec.tool(speculatable=True, pure=True, latency_hint_ms=300)
    def llm_query(prompt: str) -> str:
        time.sleep(0.25)
        return f"ans({prompt[:12]})"

    @spec.tool()
    def notify(msg: str) -> str:
        effects.append(msg)
        return "sent"

    ns = {"__builtins__": __builtins__, "answer": {}}
    ns.update(spec.hooks())

    stream = (
        "thinking\n```repl\n"
        "rs = [llm_query('q' + str(i)) for i in range(4)]\n"
        "receipt = notify('done')\n"
        "answer['content'] = '|'.join(rs) + '+' + receipt\n"
        "```\n"
    )
    code = (
        "rs = [llm_query('q' + str(i)) for i in range(4)]\n"
        "receipt = notify('done')\n"
        "answer['content'] = '|'.join(rs) + '+' + receipt\n"
    )

    t0 = time.perf_counter()
    with spec.turn(repl_locals={"answer": ns["answer"]}) as t:
        for i in range(0, len(stream), 12):
            t.feed(stream[i : i + 12])
    exec(code, ns)
    wall = time.perf_counter() - t0
    spec.end_turn()

    assert ns["answer"]["content"].count("|") == 3
    assert ns["answer"]["content"].endswith("+sent")
    assert effects == ["done"], "side-effect tool ran exactly once, never early"
    s = spec.stats()
    assert s["claimed"] == 4 and s["evicted"] == 0
    assert wall < 4 * 0.25, f"no fan-out through the facade: {wall:.2f}s"
    spec.close()


def test_patch_rlm_one_call():
    import rlm.environments as envs

    from demo.rlm import patch_rlm

    original = envs.LocalREPL
    try:
        cls = patch_rlm()
        env = envs.get_environment("local", {"context_payload": "hello"})
        assert isinstance(env, cls), "factory must now build the speculative REPL"
        assert hasattr(env, "begin_stream_turn")
        env.cleanup()
    finally:
        envs.LocalREPL = original
        envs.local_repl.LocalREPL = original
