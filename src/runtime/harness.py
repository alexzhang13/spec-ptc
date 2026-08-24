"""Real execution: SpecREPL, per-turn metrics, and the Harness turn loop
(stream -> segment -> shadow(+peek) -> real exec; modes: baseline | lazy | spec)."""

from __future__ import annotations

import builtins as _builtins
import io
import time
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

from spec_ptc.contracts.events import EventBus, SpecEvent
from spec_ptc.contracts.tools import ToolRegistry
from spec_ptc.engine.shadow import ShadowRunner
from spec_ptc.engine.speculation import (
    Budget,
    Launcher,
    SpecStore,
    make_baseline_hooks,
    make_real_hooks,
    make_shadow_hooks,
)
from spec_ptc.engine.streaming import Segment, StreamSegmenter

# The real REPL's builtins: permissive like RLM's _SAFE_BUILTINS (imports and
# open allowed — this is the environment the user already approved).

REAL_BUILTINS = {
    name: getattr(_builtins, name, None)
    for name in [
        "print",
        "len",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "bool",
        "type",
        "isinstance",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "range",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "any",
        "all",
        "repr",
        "format",
        "hash",
        "iter",
        "next",
        "slice",
        "callable",
        "hasattr",
        "getattr",
        "setattr",
        "chr",
        "ord",
        "divmod",
        "pow",
        "object",
        "super",
        "__import__",
        "open",
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "NameError",
        "StopIteration",
        "ZeroDivisionError",
        "ArithmeticError",
        "LookupError",
        "OSError",
    ]
}


@dataclass
class REPLResult:
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    final_answer: str | None = None
    locals: dict = field(default_factory=dict)


class SpecREPL:
    def __init__(self, context=None) -> None:
        self.globals: dict = {"__builtins__": dict(REAL_BUILTINS), "__name__": "__main__"}
        self.locals: dict = {"answer": {"content": "", "ready": False}}
        if context is not None:
            self.locals["context"] = context
            self.locals["context_0"] = context

    def execute_code(self, code: str, hooks: dict) -> REPLResult:
        t0 = time.perf_counter()
        ns = {**self.globals, **hooks, **self.locals}
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                exec(code, ns, ns)
        except Exception as e:
            err.write(f"\n{type(e).__name__}: {e}")
        # persist new/changed variables (hooks + dunders excluded)
        for k, v in ns.items():
            if k in hooks or k.startswith("__"):
                continue
            if k not in self.globals or self.globals.get(k) is not v:
                self.locals[k] = v
        # restore scaffold names the model may have clobbered
        if "context_0" in self.locals:
            self.locals["context"] = self.locals["context_0"]
        ans = self.locals.get("answer")
        final = None
        if isinstance(ans, dict) and ans.get("ready"):
            final = str(ans.get("content", ""))
        return REPLResult(
            stdout=out.getvalue(),
            stderr=err.getvalue(),
            execution_time=time.perf_counter() - t0,
            final_answer=final,
            locals=self.locals,
        )


@dataclass
class TurnMetrics:
    mode: str = ""
    t_stream: float = 0.0  # generation wall time
    t_exec: float = 0.0  # real execution wall time
    t_turn: float = 0.0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    dispatched: int = 0
    saved_ms_total: float = 0.0  # per-claim head start actually banked
    wasted_calls: int = 0  # dispatched but never claimed

    def row(self) -> dict:
        return {
            k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()
        }


def collect(events: list[SpecEvent], store: SpecStore, mode: str) -> TurnMetrics:
    m = TurnMetrics(mode=mode)
    t_turn0 = t_stream0 = t_stream1 = t_exec0 = t_exec1 = None
    for ev in events:
        d = ev.data
        if ev.kind == "turn_begin":
            t_turn0 = ev.t
        elif ev.kind == "stream_begin":
            t_stream0 = ev.t
        elif ev.kind == "stream_end":
            t_stream1 = ev.t
        elif ev.kind == "exec_begin":
            t_exec0 = ev.t
        elif ev.kind == "exec_end":
            t_exec1 = ev.t
            m.t_turn = ev.t - (t_turn0 or ev.t)
        elif ev.kind == "dispatch":
            m.dispatched += 1
        elif ev.kind == "claim_hit":
            m.hits += 1
        elif ev.kind == "claim_miss":
            m.misses += 1
        elif ev.kind == "claim_done":
            m.saved_ms_total += d.get("saved_ms", 0.0)
        elif ev.kind == "evict":
            m.evictions += 1
    if t_stream0 and t_stream1:
        m.t_stream = t_stream1 - t_stream0
    if t_exec0 and t_exec1:
        m.t_exec = t_exec1 - t_exec0
    m.wasted_calls = sum(
        1 for s in store.all if s.state in ("evicted", "failed", "pending", "running", "ready")
    )
    return m


MODES = ("baseline", "lazy", "spec")


def _worthwhile(tail: str, last: str, shadow) -> bool:
    from spec_ptc.speculator import StreamTurn

    return StreamTurn._peek_worthwhile(
        type("T", (), {"_last_tail": last, "shadow": shadow})(), tail
    )


@dataclass
class TurnOutcome:
    response: str = ""
    results: list = field(default_factory=list)  # REPLResult per block
    final_answer: str | None = None
    metrics: TurnMetrics | None = None


class Harness:
    def __init__(
        self,
        engine,
        mode: str,
        bus: EventBus | None = None,
        context=None,
        max_inflight: int = 16,
        peek: bool = True,
        taint_skip: bool = True,
    ):
        self.peek = peek
        self.taint_skip = taint_skip
        assert mode in MODES
        self.mode = mode
        self.bus = bus or EventBus()
        self.engine = engine
        self.reg = ToolRegistry()
        engine.make_tools(self.reg, self.bus)
        self.store = SpecStore(self.bus)
        self.launcher = Launcher(self.store, self.bus, Budget(max_inflight=max_inflight))
        self.repl = SpecREPL(context=context)
        if mode == "baseline":
            self.exec_hooks = make_baseline_hooks(self.reg)
        else:
            self.exec_hooks = make_real_hooks(self.reg, self.store, self.launcher, self.bus)
        self.shadow_hooks = make_shadow_hooks(self.reg, self.launcher)

    # ------------------------------------------------------------------ turn
    def run_turn(
        self, stream: Iterator[str], on_delta: Callable[[str], None] | None = None
    ) -> TurnOutcome:
        bus = self.bus
        bus.emit("turn_begin", mode=self.mode)
        seg = StreamSegmenter()
        shadow: ShadowRunner | None = None
        segments: list[Segment] = []

        def ensure_shadow() -> ShadowRunner:
            nonlocal shadow
            if shadow is None:
                shadow = ShadowRunner(
                    self.repl.locals,
                    self.shadow_hooks,
                    self.store,
                    REAL_BUILTINS,
                    bus,
                    launcher=self.launcher,
                    registry=self.reg,
                    taint_skip=self.taint_skip,
                )
            return shadow

        # ---- stream
        bus.emit("stream_begin")
        parts: list[str] = []
        last_tail = ""
        for delta in stream:
            parts.append(delta)
            bus.emit("token", text=delta)
            if on_delta:
                on_delta(delta)
            for s in seg.feed(delta):
                segments.append(s)
                bus.emit("stmt_closed", block=s.block_id, index=s.index, src=s.source)
                if self.mode == "spec":
                    ensure_shadow().feed(s)
            # streaming-level peek: each completed LINE inside an open block
            # re-plans speculation over the unclosed tail (see peek.py); the
            # throttle logic lives in StreamTurn._peek_worthwhile
            if self.mode == "spec" and self.peek and "\n" in delta and shadow is not None:
                tail = seg.pending_tail()
                if tail.strip() and tail != last_tail and _worthwhile(tail, last_tail, shadow):
                    last_tail = tail
                    shadow.feed_peek(tail)
        for s in seg.finish():
            segments.append(s)
            bus.emit("stmt_closed", block=s.block_id, index=s.index, src=s.source)
            if self.mode == "spec":
                ensure_shadow().feed(s)
        bus.emit("stream_end")
        response = "".join(parts)

        # ---- lazy mode: shadow pre-pass now, over the complete blocks
        if self.mode == "lazy" and segments:
            sh = ensure_shadow()
            for s in segments:
                sh.feed(s)

        # ---- wait for shadow only when it can produce claims (Invariant N:
        # a pure-compute turn must not pay for the shadow's duplicate work)
        if shadow is not None:
            shadow.finish()
            if any(s.has_call for s in segments):
                shadow.join(timeout=600)

        # ---- real execution, block by block
        outcome = TurnOutcome(response=response)
        blocks: dict[int, list[Segment]] = {}
        for s in segments:
            blocks.setdefault(s.block_id, []).append(s)
        bus.emit("exec_begin")
        for bid in sorted(blocks):
            code = "\n".join(s.source for s in blocks[bid])
            bus.emit("real_block_begin", block=bid, code=code)
            res = self.repl.execute_code(code, self.exec_hooks)
            outcome.results.append(res)
            bus.emit(
                "real_block_end",
                block=bid,
                stdout=res.stdout[:800],
                stderr=res.stderr[:400],
                ms=res.execution_time * 1000,
                answer=res.final_answer,
            )
            if res.final_answer is not None:
                outcome.final_answer = res.final_answer
        bus.emit("exec_end")

        # ---- cleanup: stop a straggling shadow, evict anything unclaimed
        if shadow is not None:
            shadow.abort("turn_end")
        self.store.evict_unclaimed("turn_end")
        self.launcher.budget.reset()
        outcome.metrics = collect(bus.history, self.store, self.mode)
        return outcome
