"""Preflight a model on this node's vLLM: load, one completion, unload."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from model_sweep import PORT, start_server, stop_server, wait_health  # noqa: E402

CONFIGS = [("Qwen/Qwen3.5-27B", 1, 0.90), ("Qwen/Qwen3.5-122B-A10B-FP8", 2, 0.92)]

for model, tp, util in CONFIGS:
    print(f"== preflight {model} tp={tp}", flush=True)
    proc = start_server(model, tp, util)
    if not wait_health(proc, timeout=1800):
        print(f"   LOAD FAILED (infra/sweep-{model.split('/')[-1]}-tp{tp}.log)", flush=True)
        stop_server(proc)
        continue
    import time
    from openai import OpenAI
    c = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="EMPTY")
    t0 = time.perf_counter()
    r = c.chat.completions.create(model=model, max_tokens=60, messages=[
        {"role": "user", "content": "Write one line of python that sums 1..10."}])
    print(f"   OK {time.perf_counter()-t0:.1f}s -> {r.choices[0].message.content[:100]!r}",
          flush=True)
    stop_server(proc)
print("PREFLIGHT DONE", flush=True)
