"""Live vLLM wiring: read .endpoints.env written by infra/serve.sbatch."""

from __future__ import annotations

from spec_ptc.runtime.engines import MockLM, VLLMEngine, engine_from_env  # noqa: F401


class HybridEngine:
    """Scripted main-model stream (deterministic demos) + REAL vLLM sub-calls.
    This is the honest live-demo config: the visible speedup comes from real
    GPU sub-LM latencies, while the trajectory stays reproducible."""

    def __init__(self, timing, vllm: VLLMEngine) -> None:
        self._mock = MockLM(timing)
        self._vllm = vllm

    def stream_main(self, script: str):
        return self._mock.stream_main(script)

    def make_tools(self, reg, bus) -> None:
        self._vllm.make_tools(reg, bus)


RLM_SYSTEM = """You are an assistant with a persistent Python REPL.
Variables already defined:
  context : a long document (string) relevant to the user's question
  answer  : dict; set answer['content'] = <final answer string> and answer['ready'] = True when done
Functions available inside the REPL:
  llm_query(prompt: str) -> str            # ask a sub-LLM one question (no tools)
  llm_query_batched(prompts: list) -> list # many at once
Rules:
- Emit Python inside ```repl ... ``` fenced blocks; I execute them and show you stdout.
- The context is too long to read at once: slice it, and map llm_query over chunks.
- Prefer simple loops that append llm_query results to a list, then combine.
- Use print() to inspect. Finish by setting answer['content'] and answer['ready'] = True.
"""


def instrument_inline_calls(h, bus, solve_dp_preview: str = "matrix") -> None:
    """Baseline hooks are silent; wrap them so a UI can show each call the
    moment real execution reaches it (call_begin/call_end, cid-keyed)."""
    import itertools
    import time

    cid = itertools.count()

    def preview_of(name, args):
        if name == "solve_dp":
            return solve_dp_preview
        if args and isinstance(args[0], str):
            return args[0][:56]
        if args and isinstance(args[0], (list, tuple)) and args[0]:
            return str(args[0][0])[:56]
        return ""

    def wrap(name, fn):
        def hook(*a, **kw):
            i = next(cid)
            n = (
                len(a[0])
                if name.endswith("_batched") and a and isinstance(a[0], (list, tuple))
                else 1
            )
            bus.emit("call_begin", cid=i, tool=name, n=n, preview=preview_of(name, a))
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                bus.emit("call_end", cid=i, tool=name, ms=(time.perf_counter() - t0) * 1000)

        return hook

    for name in list(h.exec_hooks):
        h.exec_hooks[name] = wrap(name, h.exec_hooks[name])


def rlm_turns(h, eng, messages: list, max_turns: int = 6) -> str | None:
    """Run the RLM loop on an existing Harness until answer['ready'] or the
    turn budget runs out. Mutates `messages` in place; returns the answer."""
    final = None
    for _ in range(max_turns):
        out = h.run_turn(eng.stream_main(messages))
        messages.append({"role": "assistant", "content": out.response})
        if out.final_answer:
            final = out.final_answer
            break
        stdout = (
            "\n".join(
                r.stdout + (("\nERR: " + r.stderr) if r.stderr.strip() else "")
                for r in out.results
            )
            or "(no repl block found — emit one)"
        )
        messages.append({"role": "user", "content": f"REPL output:\n{stdout[:2000]}"})
    return final
