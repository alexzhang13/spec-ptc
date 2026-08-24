"""Suite runner: one fixed pattern, streamed at a realistic rate, real vLLM
sub-calls, baseline vs spec. All metrics from the event bus."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from spec_ptc import EventBus, Harness
from spec_ptc.contracts.tools import ToolRegistry

from benchmark.suite.patterns import CONTEXT, Pattern, response_text


class SuiteEngine:
    """Scripted main stream (the pattern text, token-paced) + REAL vLLM subs."""

    def __init__(self, base_url: str, model: str, tps: float = 60.0,
                 sub_max_tokens: int = 96):
        from openai import OpenAI
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
            yield text[i:i + chunk]

    def _call(self, prompt, _spec=None):
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": str(prompt)}],
            stream=True, max_tokens=self.sub_max_tokens, temperature=0.7)
        parts = []
        for ch in resp:
            if _spec is not None and _spec.state == "evicted":
                resp.close()
                raise RuntimeError("evicted")
            d = ch.choices[0].delta.content if ch.choices else None
            if d:
                parts.append(d)
        return "".join(parts).strip()

    def make_tools(self, reg: ToolRegistry, bus: EventBus) -> None:
        self.bus = bus

        def single(prompt, _spec=None):
            return self._call(prompt, _spec)
        single.wants_spec = True
        reg.register("llm_query", single, speculatable=True, pure=True,
                     latency_hint_ms=1500)

        def batched(prompts, **kw):
            with ThreadPoolExecutor(max_workers=8) as pool:
                return list(pool.map(self._call, prompts))
        reg.register("llm_query_batched", batched, speculatable=True, pure=True,
                     batched=True, latency_hint_ms=1500)


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
    lead_s_median: float     # dispatch -> stream_end head start
    call_s_median: float     # real sub-call execution latency
    score: float
    stderr_head: str


def run_pattern(p: Pattern, mode: str, base_url: str, model: str,
                tps: float = 60.0) -> RunResult:
    eng = SuiteEngine(base_url, model, tps=tps,
                      sub_max_tokens=p.sub_max_tokens)
    bus = EventBus()
    h = Harness(eng, mode, bus=bus, context=CONTEXT)
    t0 = time.perf_counter()
    out = h.run_turn(eng.stream_main(response_text(p)))
    wall = time.perf_counter() - t0
    ev = bus.history
    tget = lambda k: [e.t for e in ev if e.kind == k]
    stream_end = (tget("stream_end") or [t0])[-1]
    disp = [e for e in ev if e.kind == "dispatch"]
    leads = [stream_end - e.t for e in disp if e.t < stream_end]
    calls = [e.data.get("ms", 0) / 1000 for e in ev if e.kind == "ready"]
    ns = h.repl.locals
    score = p.check(ns) if p.check else float(bool(out.final_answer))
    stderr = ""
    for r in out.results:
        if r.stderr.strip():
            stderr = r.stderr.strip().splitlines()[-1][:120]
    res = RunResult(
        wall_s=round(wall, 2),
        stream_s=round((tget("stream_end")[-1] - tget("stream_begin")[0]), 2) if tget("stream_begin") else 0,
        exec_s=round((tget("exec_end")[-1] - tget("exec_begin")[0]), 2) if tget("exec_begin") else 0,
        dispatched=len(disp),
        peeked=sum(1 for e in disp if e.data.get("source") == "peek"),
        hits=sum(1 for e in ev if e.kind == "claim_hit"),
        misses=sum(1 for e in ev if e.kind == "claim_miss"),
        evicted=sum(1 for e in ev if e.kind == "evict"),
        adopted=sum(1 for e in ev if e.kind == "adopt"),
        lead_s_median=round(statistics.median(leads), 2) if leads else 0.0,
        call_s_median=round(statistics.median(calls), 2) if calls else 0.0,
        score=round(score, 3),
        stderr_head=stderr,
    )
    h.launcher.shutdown()
    return res
