"""Run the suite: patterns × {baseline, spec} × repeats, resumable, CSV out.

  uv run python -m benchmark.suite.run_suite --repeats 5 [--only name] [--tps 60]
                                             [--tag v1] [--model-tag 4B]
"""

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import benchmark.suite.patterns_v2  # noqa: F401  (registers V2)
from benchmark.suite.patterns import PATTERNS
from benchmark.suite.runner import run_pattern

HERE = Path(__file__).parent


def endpoint():
    env = dict(l.split("=", 1) for l in
               (HERE / ".endpoint.env").read_text().strip().splitlines())
    return env["SUITE_URL"], env["SUITE_MODEL"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--tps", type=float, default=60.0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    url, model = endpoint()
    out = HERE / "results" / f"{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "runs.csv"
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done = {(r["pattern"], r["mode"], int(r["rep"]), float(r["tps"]))
                    for r in csv.DictReader(f)}
    first = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    writer = None
    for p in PATTERNS:
        if args.only and args.only not in p.name:
            continue
        for rep in range(1, args.repeats + 1):
            for mode in ("baseline", "spec"):
                key = (p.name, mode, rep, args.tps)
                if key in done:
                    continue
                t0 = time.time()
                try:
                    r = run_pattern(p, mode, url, model, tps=args.tps)
                    row = {"pattern": p.name, "category": p.category, "mode": mode,
                           "rep": rep, "tps": args.tps, "model": model.split("/")[-1],
                           **dataclasses.asdict(r)}
                except Exception as e:
                    row = {"pattern": p.name, "category": p.category, "mode": mode,
                           "rep": rep, "tps": args.tps, "model": model.split("/")[-1],
                           "wall_s": round(time.time() - t0, 2), "score": 0.0,
                           "stderr_head": f"RUNNER: {type(e).__name__}: {e}"[:120]}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if first:
                        writer.writeheader()
                        first = False
                writer.writerow(row)
                f.flush()
                print(f"{p.name:22s} {mode:9s} r{rep} wall={row.get('wall_s')}s "
                      f"score={row.get('score')} hits={row.get('hits', '-')} "
                      f"miss={row.get('misses', '-')} evict={row.get('evicted', '-')} "
                      f"err={row.get('stderr_head', '')[:60]}", flush=True)
    f.close()
    print("SUITE RUN DONE", flush=True)


if __name__ == "__main__":
    main()
