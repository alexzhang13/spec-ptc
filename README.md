# sPTC - Speculative programmatic tool / sub-agent calling

Many harness designs like RLMs and CodeAct rely on programmatic tool-calling (PTC), where all tools are embedded as functions inside a single code REPL tool that is generated per turn. For RLMs in particular, sub-LLM and sub-RLM calls are expensive, often blocking tools in code that take up a majority of the runtime. sPTC is the general technique of speculating tool and sub-LLM calls that will happen as the root LLM is generating the codeblock, allowing the RLM to batch and asynchronously compute these expensive calls while the full codeblock is still being generated to overlap these calls with the logic of the code REPL.

This repository is a simple library and demo for this technique.

```
baseline   tokens──────────────────▶ exec: call₁──▶call₂──▶…──▶callₙ──▶ answer
spec-ptc   tokens──────────────────▶ exec: claim·claim·claim ──▶ answer
                 ╲ call₁ ▶▶▶ done ╱
                  ╲ call₂ ▶▶▶ done╱     (calls run inside generation time)
```

Run the tool calls inside a model's code **while the model is still writing
the code**. For RLM/CodeAct-style harnesses where the model composes
`llm_query()` sub-calls in Python: spec-ptc watches the token stream, executes
the program speculatively in a jailed shadow of the REPL, dispatches sub-LM
calls the moment their arguments are knowable (often before their statement
even finishes streaming), and lets the real execution claim the results.
**Wrong speculation wastes a call; it can never corrupt a result. A miss costs
one hash lookup — it is never slower than not speculating.**


Measured (deterministic mock LMs): **2.4× geomean** over 30 fixed scenarios,
up to **8.7×** on wide maps, ~1.0× (never worse) on adversarial inputs.

## Use it (30 seconds)

```python
from spec_ptc import Speculator

spec = Speculator()

@spec.tool(speculatable=True, pure=True)     # explicit opt-in; pure required
def llm_query(prompt: str) -> str: ...

@spec.tool()                                 # side effects: never speculated
def send_report(text: str) -> str: ...

ns.update(spec.hooks())                      # claiming hooks for your REPL
with spec.turn(repl_locals=ns) as t:         # per model turn
    for delta in model_stream:
        t.feed(delta)
exec(code, ns)                               # claims land automatically
```

RLM integration is one call — `from demo.rlm import patch_rlm; patch_rlm()` — and
out-of-process harnesses run `spec-ptc-daemon` and speak 4 JSON-lines
messages (vendorable client: `plugins/client.py`; Pi/CC/OpenCode wrappers in `plugins/`).

## Try it

```bash
uv sync
just list                     # 34 demo scenarios
just play oolong-mood-agg     # console demo: watch dispatch/ready/claim live
just demo oolong-mood-agg     # 4-panel TUI: speculating vs actually-running
uv run python -m benchmark.bench   # full suite + never-slower guard
just serve                    # vLLM on slurm node4, then: just demo-live
```

## Plugging it in

`api.py` defines four integration levels: use the harness (L1); install
claim-hooks into your own Python REPL (L2 — `SpeculativeLocalREPL` is a
drop-in RLM `LocalREPL` subclass); speak 4 JSON-lines messages to the daemon
from any process (L3 — Claude Code / OpenCode / Pi-mono; vendorable client in
`plugins/client.py`); or implement the executor protocol natively (L4 —
`bun/` is a working JS reference).

## Layout

```
src/spec_ptc/   the technique: speculator (frontend) · tools · speculation ·
                streaming · shadow · harness · events · engines · daemon
demo/           a consumer: TUI, console player, live RLM loop, scenarios, RLM adapter
plugins/        another consumer: daemon client + Pi / Claude Code / OpenCode wrappers + Bun
benchmark/      all measurement: fixed/random/geometry/failure benches + live experiments
```
Core never imports the other layers.
