"""Render SUITE.md: design, per-test justification (from patterns.py), results."""

import sys
from pathlib import Path

import benchmark.suite.patterns_v2  # noqa: F401  (registers V2)
from benchmark.suite.patterns import PATTERNS

HEADER = """# The Speculation Suite

A rigorous, fixed-trajectory comparison of **speculating vs not speculating**
on real LLM calls. Every test is a *pre-defined* REPL program replayed as a
token-paced stream (default 60 tok/s) against a live vLLM sub-model — so the
trajectory is identical across methods and repeats, and variance comes only
from genuine sub-call latencies. Each (test × method) runs **5×**.

**What is measured per run** (from the event bus): wall time, stream/exec
split, dispatches (and how many came from pre-close peeks), claim hits /
misses, evictions (waste), adoptions, median dispatch *lead time* (how long
before generation ended the call was in flight), median sub-call latency,
and a per-test correctness score.

**Design principles**
1. *Fixed trajectories*: the code never varies, only execution strategy does —
   the cleanest possible isolation of the technique (motivated by the OOLONG
   campaign, where sampled trajectories varied up to 173× and swamped
   method effects).
2. *Real calls only*: sub-calls hit a live vLLM endpoint with streaming and
   temperature sampling; no mocks anywhere.
3. *Adversarial by construction*: the `hard` category encodes every failure
   class we discovered (FAILED.md) as a runnable test with a falsifiable
   hypothesis; the suite is only trustworthy if speculation is allowed to
   lose where it should.
4. *Hypotheses first*: every test states its expected mechanism BEFORE
   results; the results table marks each hypothesis supported / refuted.

"""


def render(results_md: str = "") -> str:
    out = [HEADER]
    for cat, title, blurb in (
        ("easy", "Category A — expected wins",
         "Patterns with parallel width; speculation should convert Σ(calls) into ≈max(call)."),
        ("hard", "Category B — deliberately difficult",
         "Serial dependencies, control flow on speculated values, self-mutation, "
         "early exits, jail violations, failure paths, multiplicity traps."),
        ("edge", "Category C — edge-case curriculum",
         "Parser/streaming stress with real calls: the machinery must neither "
         "misparse nor silently skip."),
    ):
        out.append(f"## {title}\n\n{blurb}\n")
        for p in [p for p in PATTERNS if p.category == cat]:
            out.append(f"### `{p.name}`\n{p.hypothesis}\n")
    if results_md:
        out.append(results_md)
    return "\n".join(out)


if __name__ == "__main__":
    results = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else ""
    Path(__file__).parent.joinpath("SUITE.md").write_text(render(results))
    print("wrote SUITE.md")
