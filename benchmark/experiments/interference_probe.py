"""GPU-interference probe: does speculative sub-call load slow the MAIN
stream when both run on the SAME vLLM instance? (Invariant N at the engine
level.)"""

import threading
import time

from spec_ptc.runtime.engines import engine_from_env

eng = engine_from_env()
PROMPT = "Write a detailed 400-word essay about GPU clusters."


def measure_stream_tps() -> float:
    resp = eng.sub.chat.completions.create(
        model=eng.sub_model,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        max_tokens=384,
        temperature=0.7,
    )
    n, t0, t_first = 0, time.perf_counter(), None
    for chunk in resp:
        d = chunk.choices[0].delta.content if chunk.choices else None
        if d:
            if t_first is None:
                t_first = time.perf_counter()
            n += 1
    return n / (time.perf_counter() - t_first)


def spec_load(stop):
    while not stop.is_set():
        try:
            eng.sub.chat.completions.create(
                model=eng.sub_model,
                messages=[{"role": "user", "content": "Summarize: " + "data " * 300}],
                max_tokens=256,
                temperature=0.7,
            )
        except Exception:
            pass


alone = min(measure_stream_tps() for _ in range(2))
stop = threading.Event()
threads = [threading.Thread(target=spec_load, args=(stop,), daemon=True) for _ in range(12)]
[t.start() for t in threads]
time.sleep(2)  # let the batch fill
loaded = min(measure_stream_tps() for _ in range(2))
stop.set()
print(f"main-stream decode alone : {alone:6.1f} tok/s")
print(f"with 12 concurrent specs : {loaded:6.1f} tok/s")
print(f"degradation              : {100 * (1 - loaded / alone):5.1f}%")
