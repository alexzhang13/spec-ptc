from spec_ptc.runtime.engines import MockLM, VLLMEngine, engine_from_env  # noqa: F401

"""Live vLLM wiring: read .endpoints.env written by infra/serve.sbatch."""

from __future__ import annotations


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


def run_live(scenario_name: str, mode: str = "spec", max_turns: int = 4, bus=None, engine=None):
    """Full-live RLM loop: the MAIN model writes the repl code for a real
    scenario context; sub-calls speculate as it streams."""
    from demo.scenarios import get_scenario
    from spec_ptc.contracts.events import EventBus
    from spec_ptc.runtime.harness import Harness

    sc = get_scenario(scenario_name)
    bus = bus or EventBus()
    eng = engine or engine_from_env(bus=bus)
    h = Harness(eng, mode, bus=bus, context=sc.context)
    question = sc.description
    messages = [
        {"role": "system", "content": RLM_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {question}\n"
            f"(context is loaded; len(context) = {len(sc.context)} chars)",
        },
    ]
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
    h.launcher.shutdown()
    return final, h, bus
