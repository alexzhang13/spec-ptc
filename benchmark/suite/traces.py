"""Event traces for a few representative patterns -> results/<tag>/traces.md,
included as an appendix in SUITE.md. Run AFTER data collection (it issues real
calls, so it would otherwise perturb the timings it documents).

  uv run python -m benchmark.suite.traces [--tag main] [--endpoint '']
"""

import argparse
import collections
from pathlib import Path

import benchmark.suite.patterns_v2  # noqa: F401
import benchmark.suite.patterns_v3  # noqa: F401
import benchmark.suite.patterns_v4  # noqa: F401
import benchmark.suite.patterns_v5  # noqa: F401
import benchmark.suite.patterns_v6  # noqa: F401
from benchmark.suite.patterns import CONTEXT, PATTERNS, response_text
from benchmark.suite.run_suite import endpoint
from benchmark.suite.runner import SuiteEngine
from spec_ptc import EventBus, Harness

HERE = Path(__file__).parent
SHOW = [
    ("map16", "the canonical win: a for-loop unrolled by the pre-close peek"),
    (
        "mutating_loop",
        "divergence: guesses retracted when the loop mutates its "
        "own input, and the correct calls re-issued",
    ),
    (
        "taint_split",
        "taint: three args depend on a non-speculatable tool and "
        "are not guessed; the other three still fan out",
    ),
    (
        "turn2_generator",
        "unforkable state: the fork refuses a generator left by "
        "the previous turn and degrades to no speculation",
    ),
    (
        "tool_error_raises",
        "a tool that raises: the error surfaces at the use site, not at dispatch",
    ),
    (
        "jail_break_mid",
        "the shadow hits a forbidden import and stops; calls already dispatched are still used",
    ),
]
KEEP = (
    "shadow_exec",
    "shadow_skip",
    "shadow_stop",
    "dispatch",
    "evict",
    "claim_hit",
    "claim_miss",
    "adopt",
)


def trace(name: str, url: str, model: str) -> list[str]:
    p = {x.name: x for x in PATTERNS}[name]
    eng = SuiteEngine(url, model, tps=60.0, sub_max_tokens=p.sub_max_tokens)
    bus = EventBus()
    h = Harness(eng, "spec", bus=bus, context=CONTEXT)
    h.run_turn(eng.stream_main(response_text(p)))
    h.launcher.shutdown()
    hist = collections.Counter(e.kind for e in bus.history)
    lines = [f"events: {dict(hist)}", ""]
    t0 = bus.history[0].t
    for e in bus.history:
        if e.kind not in KEEP:
            continue
        d = dict(e.data)
        detail = d.get("src") or d.get("preview") or d.get("reason") or d.get("key") or ""
        if isinstance(detail, tuple):
            detail = detail[0]
        lines.append(f"{e.t - t0:6.2f}s  {e.kind:12s} {str(detail)[:78]}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--endpoint", default="")
    args = ap.parse_args()
    url, model = endpoint(args.endpoint)
    out = [
        "## Appendix: event traces",
        "",
        "One speculative run of six representative patterns, straight from "
        f"the event bus (`{model.split('/')[-1]}`, 60 tok/s). Times are "
        "seconds from the first token of the response.",
        "",
    ]
    for name, why in SHOW:
        out += [f"### `{name}`", "", why, "", "```text"]
        out += trace(name, url, model)
        out += ["```", ""]
    p = HERE / "results" / args.tag / "traces.md"
    p.write_text("\n".join(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
