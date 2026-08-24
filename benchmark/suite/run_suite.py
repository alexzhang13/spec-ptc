"""Run the suite: patterns × {baseline, spec} × repeats, resumable, CSV out.

uv run python -m benchmark.suite.run_suite --repeats 5 [--only name] [--tps 60]
                                           [--tag v1] [--model-tag 4B]
"""

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import benchmark.suite.patterns_v2  # noqa: F401
import benchmark.suite.patterns_v3  # noqa: F401
import benchmark.suite.patterns_v4  # noqa: F401
import benchmark.suite.patterns_v5  # noqa: F401
import benchmark.suite.patterns_v6  # noqa: F401
from benchmark.suite.patterns import PATTERNS
from benchmark.suite.runner import RunResult, run_pattern

HERE = Path(__file__).parent


def endpoint(which: str = ""):
    """which='' -> .endpoint.env; which='2' -> .endpoint2.env (2nd model)."""
    f = HERE / f".endpoint{which}.env"
    env = dict(line.split("=", 1) for line in f.read_text().strip().splitlines())
    return env["SUITE_URL"], env["SUITE_MODEL"]


def _align(csv_path: Path, fields: list[str]) -> list[str]:
    """Keep appends schema-safe: if the file's header is missing columns this
    run produces, rewrite it once with the union (blanks for old rows)."""
    if not csv_path.exists():
        return fields
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    old = list(rows[0].keys()) if rows else []
    if not old or set(fields) <= set(old):
        return old or fields
    union = old + [c for c in fields if c not in old]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=union)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in union})
    return union


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--tps", type=float, default=60.0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--endpoint", default="", help="'' or '2' (second model)")
    ap.add_argument(
        "--conc", type=int, default=1, help="run N identical turns at once; wall_s = makespan"
    )
    ap.add_argument(
        "--aa", action="store_true", help="A/A control: run baseline twice instead of vs spec"
    )
    args = ap.parse_args()
    url, model = endpoint(args.endpoint)
    out = HERE / "results" / f"{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "runs.csv"
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done = {
                (r["pattern"], r["mode"], int(r["rep"]), float(r["tps"]))
                for r in csv.DictReader(f)
            }
    fields = ["pattern", "category", "mode", "rep", "tps", "conc", "model"] + [
        x.name for x in dataclasses.fields(RunResult)
    ]
    fields = _align(csv_path, fields)
    first = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    writer = None
    for p in PATTERNS:
        pats = [s for s in args.only.split(",") if s]
        if pats and not any(s in p.name for s in pats):
            continue
        for rep in range(1, args.repeats + 1):
            for mode in ("baseline", "aa") if args.aa else ("baseline", "spec"):
                key = (p.name, mode, rep, args.tps)
                if key in done:
                    continue
                t0 = time.time()
                try:
                    r = run_pattern(p, mode, url, model, tps=args.tps, conc=args.conc)
                    row = {
                        "pattern": p.name,
                        "category": p.category,
                        "mode": mode,
                        "rep": rep,
                        "tps": args.tps,
                        "conc": args.conc,
                        "model": model.split("/")[-1],
                        **dataclasses.asdict(r),
                    }
                except Exception as e:
                    row = {
                        "pattern": p.name,
                        "category": p.category,
                        "mode": mode,
                        "rep": rep,
                        "tps": args.tps,
                        "conc": args.conc,
                        "model": model.split("/")[-1],
                        "wall_s": round(time.time() - t0, 2),
                        "score": 0.0,
                        "stderr_head": f"RUNNER: {type(e).__name__}: {e}"[:120],
                    }
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    if first:
                        writer.writeheader()
                        first = False
                writer.writerow({k: row.get(k, "") for k in fields})
                f.flush()
                print(
                    f"{p.name:22s} {mode:9s} r{rep} wall={row.get('wall_s')}s "
                    f"score={row.get('score')} hits={row.get('hits', '-')} "
                    f"miss={row.get('misses', '-')} evict={row.get('evicted', '-')} "
                    f"err={row.get('stderr_head', '')[:60]}",
                    flush=True,
                )
    f.close()
    print("SUITE RUN DONE", flush=True)


if __name__ == "__main__":
    main()
