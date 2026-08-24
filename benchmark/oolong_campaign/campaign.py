"""Sequential campaign driver: one server at a time on this node, resumable."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark" / "experiments"))

from model_sweep import PORT, start_server, stop_server, wait_health  # noqa: E402

from benchmark.oolong_campaign.data import load_tasks  # noqa: E402
from benchmark.oolong_campaign.hints import HINT_NAME  # noqa: E402
from benchmark.oolong_campaign.runner import run_unit  # noqa: E402

MODELS = [
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", 1, 0.92),
    ("Qwen/Qwen3.5-27B", 1, 0.90),
    ("Qwen/Qwen3.5-122B-A10B-FP8", 2, 0.92),
]
CONCURRENCIES = (8, 4, 1)
REPEATS = 3
N_TREC, N_PAIRS = 4, 4


def main(campaign_id: str = "main") -> None:
    base = ROOT / "benchmark" / "oolong_campaign" / "runs" / campaign_id
    base.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(N_TREC, N_PAIRS)
    (base / "manifest.json").write_text(json.dumps({
        "hint": HINT_NAME, "models": MODELS, "concurrencies": CONCURRENCIES,
        "repeats": REPEATS, "tasks": [t.task_id for t in tasks],
        "started": time.time()}, indent=1, default=str))

    for model, tp, util in MODELS:
        mtag = model.split("/")[-1]
        pending = [(spec, conc, rep)
                   for spec in (False, True)
                   for conc in CONCURRENCIES
                   for rep in range(1, REPEATS + 1)
                   if not (base / f"{mtag}/spec={spec}_c{conc}_r{rep}/DONE").exists()]
        if not pending:
            print(f"### {mtag}: all units done, skipping server", flush=True)
            continue
        print(f"### starting server {mtag} tp={tp} ({len(pending)} units pending)",
              flush=True)
        proc = start_server(model, tp, util)
        if not wait_health(proc, timeout=3600):
            print(f"### SERVER FAILED for {mtag} — skipping model", flush=True)
            stop_server(proc)
            continue
        url = f"http://localhost:{PORT}/v1"
        try:
            for spec, conc, rep in pending:
                unit = base / f"{mtag}/spec={spec}_c{conc}_r{rep}"
                print(f"--- unit {mtag} spec={spec} conc={conc} rep={rep}", flush=True)
                t0 = time.time()
                stats = run_unit(model, url, spec, conc, rep, tasks, unit)
                (unit / "DONE").write_text(str(time.time()))
                print(f"    unit_wall={stats['unit_wall_s']:.0f}s "
                      f"task_mean={stats['task_wall_mean_s']:.0f}s "
                      f"score={stats['score_mean']:.2f} errors={stats['n_errors']}",
                      flush=True)
                _ = t0
        finally:
            stop_server(proc)
    print("CAMPAIGN DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
