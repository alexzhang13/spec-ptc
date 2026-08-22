"""EXPLOG experiments: where does speculation break the engine?."""

import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from spec_ptc.runtime.engines import engine_from_env

eng = engine_from_env()
CLIENT, MODEL = eng.sub, eng.sub_model


def stream_probe(
    prompt="Write a detailed 300-word essay about distributed systems.", max_tokens=256
):
    """One streaming request: (ttft_s, decode_tok_per_s, total_s)."""
    t0 = time.perf_counter()
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    n, t_first = 0, None
    for chunk in resp:
        d = chunk.choices[0].delta.content if chunk.choices else None
        if d:
            if t_first is None:
                t_first = time.perf_counter()
            n += 1
    t_end = time.perf_counter()
    return (t_first - t0, n / max(1e-6, t_end - t_first), t_end - t0)


def one_call(prompt, max_tokens=128):
    t0 = time.perf_counter()
    CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return time.perf_counter() - t0


def background_load(n, prompt, max_tokens, stop_evt, started_evt):
    def worker(i):
        first = True
        while not stop_evt.is_set():
            try:
                one_call(f"[{i}] " + prompt, max_tokens)
            except Exception:
                pass
            if first:
                first = False
        return None

    pool = ThreadPoolExecutor(max_workers=n)
    futs = [pool.submit(worker, i) for i in range(n)]
    time.sleep(2.5)
    started_evt.set()
    return pool, futs


def with_load(n, load_prompt, load_tokens, probe_fn):
    if n == 0:
        return probe_fn()
    stop, started = threading.Event(), threading.Event()
    pool, _ = background_load(n, load_prompt, load_tokens, stop, started)
    started.wait()
    try:
        return probe_fn()
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        time.sleep(1.0)


SHORT = "Summarize: " + "data point " * 40
LONG = "Summarize this document:\n" + (
    "The quarterly metrics improved across all measured dimensions. " * 700
)  # ~7k tok

print("== EXP-1: decode-batch saturation (short-prompt load, 128-tok gens) ==", flush=True)
for n in (0, 4, 12, 24, 48, 96):
    ttfts, rates = [], []
    for _ in range(2):
        ttft, rate, _ = with_load(n, SHORT, 128, stream_probe)
        ttfts.append(ttft), rates.append(rate)
    print(
        f"  load n={n:<3d} main ttft={min(ttfts):5.2f}s decode={max(rates):6.1f} tok/s",
        flush=True,
    )

print("== EXP-2: prefill clogging (7k-token-prompt load) ==", flush=True)
for n in (0, 4, 16):
    ttft, rate, _ = with_load(n, LONG, 32, stream_probe)
    print(f"  load n={n:<3d} main ttft={ttft:5.2f}s decode={rate:6.1f} tok/s", flush=True)

print("== EXP-3: per-call latency inflation under fan-out ==", flush=True)
serial = statistics.median(one_call(f"inflate {i}: " + SHORT) for i in range(3))
for n in (8, 32, 96):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        lats = list(pool.map(lambda i: one_call(f"fan {n}-{i}: " + SHORT), range(n)))
    batch_wall = time.perf_counter() - t0
    print(
        f"  n={n:<3d} serial-one={serial:5.2f}s  first={min(lats):5.2f}s "
        f"(inflation {min(lats) / serial:4.2f}x)  p50={statistics.median(lats):5.2f}s "
        f"p max={max(lats):5.2f}s  all-done={batch_wall:5.2f}s "
        f"(vs serial-all={serial * n:6.1f}s)",
        flush=True,
    )

print("== EXP-4: eviction-storm recovery ==", flush=True)
base = statistics.median(one_call(f"recov base {i}: " + SHORT) for i in range(3))
streams = []
for i in range(48):
    streams.append(
        CLIENT.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": f"storm {i}: " + SHORT}],
            stream=True,
            max_tokens=512,
            temperature=0.7,
        )
    )
time.sleep(1.0)
t0 = time.perf_counter()
for s in streams:
    s.close()  # the eviction: abort all 48 mid-generation
close_ms = (time.perf_counter() - t0) * 1000
after = one_call("recov after storm: " + SHORT)
after2 = one_call("recov after storm 2: " + SHORT)
print(
    f"  abort 48 in-flight streams: {close_ms:.0f}ms; call latency "
    f"base={base:.2f}s -> just-after={after:.2f}s, next={after2:.2f}s",
    flush=True,
)
print("done", flush=True)
