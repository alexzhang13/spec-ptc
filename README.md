# Speculative Programmatic Tool Calling

Speculative programmatic tool-calling (**sPTC**) is a technique for harnesses that use tools like sub-agents / sub-calls in code. While the LLM is streaming tokens to generate a REPL call, **sPTC** speculates and queues up tool calls in the partially-generated code that act as Futures when the actual code is executed.

Learn more in [**the blogpost here**](https://alexzhang13.github.io/blog/2026/spec-ptc/).

<img src="media/comparison.gif" alt="sPTC vs serial comparison" width="640">

Many harness designs like [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601) and [CodeAct](https://arxiv.org/abs/2402.01030) rely on [programmatic tool-calling (PTC)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling), where all tools are embedded as functions inside a single code REPL tool that is generated per turn. For RLMs in particular, sub-LLM and sub-RLM calls are expensive, often blocking tools in code that take up a majority of the runtime. sPTC is the general technique of speculating tool and sub-LLM calls that will happen as the root LLM is generating the codeblock, allowing the RLM to batch and asynchronously compute these expensive calls while the full codeblock is still being generated to overlap these calls with the logic of the code REPL.

```
baseline   tokens──────────────────▶ exec: call₁──▶call₂──▶…──▶callₙ──▶ answer
spec-ptc   tokens──────────────────▶ exec: claim·claim·claim ──▶ answer
                 ╲ call₁ ▶▶▶ done ╱
                  ╲ call₂ ▶▶▶ done╱     (calls run inside generation time)
```

This repository is a simple library and demo for this technique.

## Getting Started

You can either clone this repository (uses `uv`), or install with:

```bash
pip install spec-ptc
```

The `Speculator` object is used to track and store tools to be speculated, as well as the shadow REPL that is used to speculate. You can add tools with the `spec.tool` decorator and control whether you want them to be speculated or not.

The simplest example is to install tool hooks into the REPL you already have, feed tokens as they stream and feed them to the speculator, then `exec` as usual when finished:

```python
from spec_ptc import Speculator

spec = Speculator()


# tools can also be async
@spec.tool(speculatable=True, pure=True)  # add as tool to be speculated
def llm_query(prompt: str) -> str:
    return sub_lm.complete(prompt)


@spec.tool()  # side effects: never speculated
def send_report(text: str) -> str:
    return mail.send(text)


ns.update(spec.hooks())  # same names, claim-or-run

code = ""
with spec.turn(repl_locals=ns) as t:  # snapshot → discarded shadow fork
    for delta in model_stream:
        code += delta
        t.feed(delta)  # closed stmts launch calls now
exec(code, ns)  # hits return immediately
```

For the [RLM](https://github.com/alexzhang13/rlm) this is one line: `from demo.rlm import patch_rlm; patch_rlm()`.

`example.py` is an example you can start with for looking how this is done for the RLM.


For arbitrary harness, we provide a simple daemon `spec-ptc-daemon` that runs the same
shadow + store out of process (default socket `/tmp/spec-ptc.sock`) with four JSON-lines messages:

```
turn_begin {vars}      snapshot REPL variables into the shadow
feed {delta}           stream tokens; the daemon launches calls
resolve {tool, args}   → hit{result} | miss   (miss: run the tool yourself)
turn_end               evict leftovers, return hit/miss counts
```

```python
from plugins.client import SpecClient  # ~60 lines, stdlib only — copy it

c = SpecClient()
c.turn_begin({"context": doc})
c.feed(delta)  # per streamed token
hit = c.resolve("llm_query", [prompt])  # result, or None → call it yourself
c.turn_end()
```

Wrappers in `plugins/`: Claude Code (`PreToolUse`), OpenCode, Pi-mono.
