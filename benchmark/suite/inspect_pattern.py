"""Single-run event trace for one pattern — the debugging counterpart to
run_suite. Prints the event histogram, every shadow abort reason, and each
dispatch's source (peek vs shadow).

  uv run python -m benchmark.suite.inspect_pattern class_method [spec|baseline]
"""

import collections
import sys

import benchmark.suite.patterns_v2  # noqa: F401
import benchmark.suite.patterns_v3  # noqa: F401
import benchmark.suite.patterns_v4  # noqa: F401
import benchmark.suite.patterns_v5  # noqa: F401
import benchmark.suite.patterns_v6  # noqa: F401
from benchmark.suite.patterns import CONTEXT, PATTERNS, response_text
from benchmark.suite.run_suite import endpoint
from benchmark.suite.runner import SuiteEngine
from spec_ptc import EventBus, Harness


def main() -> None:
    name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "spec"
    which = sys.argv[3] if len(sys.argv) > 3 else ""  # '2' = second endpoint
    p = {x.name: x for x in PATTERNS}[name]
    url, model = endpoint(which)
    eng = SuiteEngine(url, model, tps=60.0, sub_max_tokens=p.sub_max_tokens)
    bus = EventBus()
    h = Harness(eng, mode, bus=bus, context=CONTEXT)
    out = h.run_turn(eng.stream_main(response_text(p)))
    h.launcher.shutdown()

    hist = collections.Counter(e.kind for e in bus.history)
    print(f"== {name} [{mode}] {model.split('/')[-1]}")
    print("events:", dict(hist))
    src = collections.Counter(e.data.get("source") for e in bus.history if e.kind == "dispatch")
    print("dispatch sources:", dict(src))
    for e in bus.history:
        if e.kind in ("shadow_stop", "abort", "evict", "shadow_skip", "shadow_exec"):
            print(f"  {e.kind}: {e.data}")
    print("score:", p.check(h.repl.locals) if p.check else "n/a")
    for r in out.results:
        if r.stderr.strip():
            print("stderr:", r.stderr.strip().splitlines()[-1][:160])


if __name__ == "__main__":
    main()
