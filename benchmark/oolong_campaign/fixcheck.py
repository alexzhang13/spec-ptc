"""Re-run the affected OOLONG units with the shadow __globals__ fix on vs off.

Three arms per repeat: baseline (no spec), spec without the fix (pre-fix
behavior restored by stubbing _rebind), spec with it. Same server, same tasks.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path("/shared/home/altzhang-de4f8c/spec-ptc")
sys.path.insert(0, str(ROOT))

from benchmark.oolong_campaign.data import load_tasks  # noqa: E402
from benchmark.oolong_campaign.runner import run_unit  # noqa: E402

URL = os.environ["FIXCHECK_URL"]
MODEL = os.environ.get("FIXCHECK_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
CONC = int(os.environ.get("FIXCHECK_CONC", "8"))
REPS = int(os.environ.get("FIXCHECK_REPS", "3"))
BASE = ROOT / "benchmark/oolong_campaign/runs/fixcheck"

import spec_ptc.engine.shadow as sh  # noqa: E402

_REAL_REBIND = sh._rebind


def set_arm(arm: str) -> None:
    # "nofix" restores the pre-fix behavior: cross-turn fns keep their real __globals__
    sh._rebind = (lambda fn, ns: fn) if arm == "nofix" else _REAL_REBIND


ARMS = [("base", False), ("nofix", True), ("fix", True)]


def main() -> None:
    tasks = load_tasks(4, 4)
    print(f"tasks: {[t.task_id for t in tasks]}", flush=True)
    for rep in range(1, REPS + 1):
        for arm, spec in ARMS:
            unit = BASE / arm / f"spec={spec}_c{CONC}_r{rep}"
            if (unit / "DONE").exists():
                print(f"--- skip {arm} r{rep} (done)", flush=True)
                continue
            set_arm(arm)
            print(f"--- {arm} spec={spec} c{CONC} r{rep}", flush=True)
            t0 = time.time()
            stats = run_unit(MODEL, URL, spec, CONC, rep, tasks, unit)
            (unit / "DONE").write_text(str(time.time()))
            print(
                f"    unit_wall={stats['unit_wall_s']:.0f}s "
                f"task_mean={stats['task_wall_mean_s']:.0f}s "
                f"score={stats['score_mean']:.2f} errors={stats['n_errors']} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    print("FIXCHECK DONE", flush=True)


if __name__ == "__main__":
    main()
