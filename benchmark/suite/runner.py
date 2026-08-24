"""Suite runner: one fixed pattern, streamed at a realistic rate, real vLLM
sub-calls, baseline vs spec. All metrics from the event bus."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from benchmark.suite.patterns import CONTEXT, Pattern, response_text
from spec_ptc import EventBus, Harness
from spec_ptc.contracts.tools import ToolRegistry


class SuiteEngine:
    """Scripted main stream (the pattern text, token-paced) + REAL vLLM subs."""

    def __init__(self, base_url: str, model: str, tps: float = 60.0, sub_max_tokens: int = 96):
        import threading

        from openai import OpenAI

        self._lock = threading.Lock()
        self.calls = 0  # real HTTP sub-calls issued
        self.calls_aborted = 0  # sub-calls closed early by an eviction
        self.chunks = 0  # streamed deltas ~ tokens generated
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=120)
        self.model = model
        self.tps = tps
        self.sub_max_tokens = sub_max_tokens
        self.bus = None

    def stream_main(self, text: str):
        # pace by characters (~4 chars/token) so dense code lines don't
        # stream unrealistically fast; ~6-token chunks
        chunk = 24
        delay = (chunk / 4) / self.tps
        for i in range(0, len(text), chunk):
            time.sleep(delay)
            yield text[i : i + chunk]

    def _call(self, prompt, _spec=None):
        with self._lock:
            self.calls += 1
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": str(prompt)}],
            stream=True,
            max_tokens=self.sub_max_tokens,
            temperature=0.7,
        )
        parts = []
        for ch in resp:
            if _spec is not None and _spec.state == "evicted":
                resp.close()
                with self._lock:
                    self.calls_aborted += 1
                raise RuntimeError("evicted")
            d = ch.choices[0].delta.content if ch.choices else None
            if d:
                parts.append(d)
                with self._lock:
                    self.chunks += 1
        return "".join(parts).strip()

    def make_tools(self, reg: ToolRegistry, bus: EventBus) -> None:
        self.bus = bus

        def single(prompt, _spec=None):
            return self._call(prompt, _spec)

        single.wants_spec = True
        reg.register("llm_query", single, speculatable=True, pure=True, latency_hint_ms=1500)

        def batched(prompts, **kw):
            with ThreadPoolExecutor(max_workers=8) as pool:
                return list(pool.map(self._call, prompts))

        reg.register(
            "llm_query_batched",
            batched,
            speculatable=True,
            pure=True,
            batched=True,
            latency_hint_ms=1500,
        )

        self.notes = []
        reg.register("log_note", self.notes.append)  # side effect: NOT speculatable

        reg.register("env_probe", lambda k: f"[{k}=cloud spend]")  # non-spec, returns

        def fragile(prompt, _spec=None):
            if "BOOM" in str(prompt):
                raise ValueError("tool failure: BOOM")
            return self._call(prompt, _spec)

        fragile.wants_spec = True
        reg.register(
            "llm_query_fragile", fragile, speculatable=True, pure=True, latency_hint_ms=1500
        )


@dataclass
class RunResult:
    wall_s: float
    stream_s: float
    exec_s: float
    dispatched: int
    peeked: int
    hits: int
    misses: int
    evicted: int
    adopted: int
    lead_s_median: float  # dispatch -> stream_end head start
    call_s_median: float  # real sub-call execution latency
    shadow_stops: int
    calls_made: int
    calls_aborted: int
    chunks_out: int
    abort_reason: str
    score: float
    stderr_head: str


def run_pattern(
    p: Pattern, mode: str, base_url: str, model: str, tps: float = 60.0, conc: int = 1
) -> RunResult:
    """conc>1 runs that many identical turns at once and reports the MAKESPAN
    (outer wall of the whole batch) — speculation under a loaded server."""
    if conc > 1:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            rs = [
                f.result()
                for f in [
                    pool.submit(_one_turn, p, mode, base_url, model, tps) for _ in range(conc)
                ]
            ]
        span = time.perf_counter() - t0
        add = lambda k: sum(getattr(x, k) for x in rs)
        med = lambda k: statistics.median([getattr(x, k) for x in rs])
        return RunResult(
            wall_s=round(span, 2),
            stream_s=round(med("stream_s"), 2),
            exec_s=round(med("exec_s"), 2),
            dispatched=add("dispatched"),
            peeked=add("peeked"),
            hits=add("hits"),
            misses=add("misses"),
            evicted=add("evicted"),
            adopted=add("adopted"),
            lead_s_median=round(med("lead_s_median"), 2),
            call_s_median=round(med("call_s_median"), 2),
            shadow_stops=add("shadow_stops"),
            calls_made=add("calls_made"),
            calls_aborted=add("calls_aborted"),
            chunks_out=add("chunks_out"),
            abort_reason=next((x.abort_reason for x in rs if x.abort_reason), ""),
            score=round(statistics.mean(x.score for x in rs), 3),
            stderr_head=next((x.stderr_head for x in rs if x.stderr_head), ""),
        )
    return _one_turn(p, mode, base_url, model, tps)


def _one_turn(p: Pattern, mode: str, base_url: str, model: str, tps: float = 60.0) -> RunResult:
    eng = SuiteEngine(base_url, model, tps=tps, sub_max_tokens=p.sub_max_tokens)
    bus = EventBus()
    h = Harness(eng, "baseline" if mode == "aa" else mode, bus=bus, context=CONTEXT)
    t0 = time.perf_counter()
    for text in p.turns or (response_text(p),):
        out = h.run_turn(eng.stream_main(text))
    wall = time.perf_counter() - t0
    ev = bus.history
    tget = lambda k: [e.t for e in ev if e.kind == k]
    ends = tget("stream_end")
    disp = [e for e in ev if e.kind == "dispatch"]
    # head start = time from dispatch to the end of the stream it overlapped
    leads = [min(x for x in ends if x > e.t) - e.t for e in disp if any(x > e.t for x in ends)]
    calls = [e.data.get("ms", 0) / 1000 for e in ev if e.kind == "ready"]
    ns = h.repl.locals
    score = p.check(ns) if p.check else float(bool(out.final_answer))
    stderr = ""
    for r in out.results:
        if r.stderr.strip():
            stderr = r.stderr.strip().splitlines()[-1][:120]
    res = RunResult(
        wall_s=round(wall, 2),
        stream_s=round(sum(b - a for a, b in zip(tget("stream_begin"), ends, strict=False)), 2),
        exec_s=round(
            sum(b - a for a, b in zip(tget("exec_begin"), tget("exec_end"), strict=False)), 2
        ),
        dispatched=len(disp),
        peeked=sum(1 for e in disp if e.data.get("source") == "peek"),
        hits=sum(1 for e in ev if e.kind == "claim_hit"),
        misses=sum(1 for e in ev if e.kind == "claim_miss"),
        evicted=sum(1 for e in ev if e.kind == "evict"),
        adopted=sum(1 for e in ev if e.kind == "adopt"),
        lead_s_median=round(statistics.median(leads), 2) if leads else 0.0,
        call_s_median=round(statistics.median(calls), 2) if calls else 0.0,
        shadow_stops=sum(1 for e in ev if e.kind == "shadow_stop"),
        calls_made=eng.calls,
        calls_aborted=eng.calls_aborted,
        chunks_out=eng.chunks,
        abort_reason=next(
            (str(e.data.get("reason", ""))[:60] for e in ev if e.kind == "shadow_stop"), ""
        ),
        score=round(score, 3),
        stderr_head=stderr,
    )
    h.launcher.shutdown()
    return res
