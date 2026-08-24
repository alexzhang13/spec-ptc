"""Model-size / TP sweep: how speculation speedup depends on the sub-model."""

import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/shared/home/altzhang-de4f8c/spec-ptc")
VLLM_PY = str(ROOT / ".venv-vllm/bin/python")
PORT = 8210

CONFIGS = [  # TP-only rerun (TP=1 rows measured in the 2026-08-22 sweep)
    ("Qwen/Qwen2.5-7B-Instruct", 2, 0.85),
    ("Qwen/Qwen2.5-14B-Instruct", 2, 0.90),
    ("Qwen/Qwen2.5-32B-Instruct", 2, 0.90),
]
SCENARIOS = ["oolong-mood-agg", "map-8", "oolong-two-stage", "majority-5", "chain-2"]


def idle_gpus(n: int) -> str:
    """No slurm device isolation here: pick the n emptiest physical GPUs."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    pairs = sorted(
        (int(u), int(i)) for i, u in (ln.split(", ") for ln in out.strip().splitlines())
    )
    chosen = [i for u, i in pairs[:n] if u < 2000]  # MiB: refuse busy GPUs
    if len(chosen) < n:
        raise RuntimeError(f"not enough idle GPUs: {pairs}")
    return ",".join(map(str, chosen))


def start_server(model: str, tp: int, util: float):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = idle_gpus(tp)
    print(f"  using GPUs {env['CUDA_VISIBLE_DEVICES']}", flush=True)
    env.update(
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_ATTENTION_BACKEND="FLASH_ATTN",
        PATH=f"{ROOT}/.venv-vllm/bin:" + env.get("PATH", ""),
    )
    if tp > 1:
        # EXPLOG: node4's Fabric Manager is broken for NVLink SHARP multicast
        # (nvls.cc bind fails, CUDA error 2). Disabling NVLS keeps NVLink P2P
        # and makes TP work; P2P/IB disables were red herrings.
        env.update(NCCL_NVLS_ENABLE="0")
    cmd = [
        VLLM_PY,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        str(util),
        "--max-model-len",
        "8192",
        "--tensor-parallel-size",
        str(tp),
        "--enforce-eager",
        "--kernel-config",
        '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, '
        '"enable_jit_warmup": false}',
    ]
    log = open(ROOT / f"infra/sweep-{model.split('/')[-1]}-tp{tp}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    return proc


def wait_health(proc, timeout=2400) -> bool:
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=3)
            return True
        except Exception:
            time.sleep(10)
    return False


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(12)  # EXPLOG: let GPU memory actually free before the next load


def bench_config(model: str, tp: int) -> list[dict]:
    from benchmark.bench import run_scenario
    from benchmark.scenarios import get_scenario
    from demo.live import HybridEngine
    from spec_ptc.runtime.engines import VLLMEngine

    url = f"http://localhost:{PORT}/v1"
    vllm = lambda: VLLMEngine(url, model, url, model)
    rows = []

    # call-latency probe: what does one representative sub-call cost here?
    eng = vllm()

    def one(p):
        t0 = time.perf_counter()
        eng.sub.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": p}],
            max_tokens=128,
            temperature=0.7,
        )
        return time.perf_counter() - t0

    lat = statistics.median(one(f"probe {i}: summarize: " + "data " * 150) for i in range(3))

    # interference probe: main-stream decode alone vs under 12-way spec load
    def stream_rate():
        r = eng.sub.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write a 300-word essay about clusters."}],
            stream=True,
            max_tokens=256,
            temperature=0.7,
        )
        n, t1 = 0, None
        for ch in r:
            d = ch.choices[0].delta.content if ch.choices else None
            if d:
                t1 = t1 or time.perf_counter()
                n += 1
        return n / max(1e-6, time.perf_counter() - t1)

    alone = stream_rate()
    stop_evt = threading.Event()

    def loadgen(i):
        while not stop_evt.is_set():
            try:
                one(f"load {i}: " + "data " * 100)
            except Exception:
                pass

    pool = ThreadPoolExecutor(max_workers=12)
    for i in range(12):
        pool.submit(loadgen, i)
    time.sleep(2)
    loaded = stream_rate()
    stop_evt.set()
    pool.shutdown(wait=False, cancel_futures=True)
    time.sleep(1)

    meta = {
        "model": model.split("/")[-1],
        "tp": tp,
        "call_latency_s": round(lat, 2),
        "decode_alone": round(alone, 1),
        "decode_12way": round(loaded, 1),
    }
    print(f"### {meta}", flush=True)

    for name in SCENARIOS:
        sc = get_scenario(name)
        row = {"scenario": name, **meta}
        for mode in ("baseline", "spec"):
            r = run_scenario(sc, mode, engine=HybridEngine(sc.timing, vllm()))
            row[mode + "_s"] = r["wall_s"]
            row[mode + "_hits"] = r["hits"]
            row["answered"] = r["answered"]
        row["speedup"] = round(row["baseline_s"] / row["spec_s"], 2) if row["spec_s"] else None
        rows.append(row)
        print(
            f"  {name:20s} base={row['baseline_s']:7.2f}s spec={row['spec_s']:7.2f}s "
            f"-> {row['speedup']}x hits={row['spec_hits']}",
            flush=True,
        )
    return rows


def main():
    out = []
    outfile = ROOT / "bench_out/model_sweep.json"
    for model, tp, util in CONFIGS:
        print(f"\n===== {model} TP={tp} =====", flush=True)
        proc = start_server(model, tp, util)
        if not wait_health(proc):
            print(
                f"  SERVER FAILED (see infra/sweep-{model.split('/')[-1]}-tp{tp}.log)",
                flush=True,
            )
            out.append({"model": model.split("/")[-1], "tp": tp, "failed": True})
            stop_server(proc)
            continue
        try:
            out.extend(bench_config(model, tp))
        except Exception as e:
            print(f"  BENCH ERROR: {type(e).__name__}: {e}", flush=True)
            out.append(
                {"model": model.split("/")[-1], "tp": tp, "failed": True, "error": str(e)[:200]}
            )
        finally:
            stop_server(proc)
        outfile.write_text(json.dumps(out, indent=1))
    print("\nSWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
