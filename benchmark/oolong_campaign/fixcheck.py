"""Re-run the affected OOLONG units with the shadow __globals__ fix on vs off.

Three arms per repeat: baseline (no spec), spec without the fix (pre-fix
behavior restored by stubbing _rebind), spec with it. Same server, same tasks.
"""

import io
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path("/shared/home/altzhang-de4f8c/spec-ptc")
sys.path.insert(0, str(ROOT))

# --- campaign-era LocalREPL concurrency patches (commit 3f84bd6, lost in the
# layered-repo refactor and absent from HEAD's runner.py). Without these the
# _temp_cwd chdir race poisons the process cwd at conc>1 and every later unit
# dies with FileNotFoundError. Replicated here so shared code stays untouched.
import rlm.environments as rlm_envs  # noqa: E402


@contextmanager
def _stable_cwd(self):
    yield


class _ThreadTee:
    """sys.stdout/err proxy: per-thread capture buffer, else the real stream."""

    def __init__(self, real):
        self.real = real
        self.local = threading.local()

    def write(self, s):
        buf = getattr(self.local, "buf", None)
        return (buf if buf is not None else self.real).write(s)

    def flush(self):
        buf = getattr(self.local, "buf", None)
        (buf if buf is not None else self.real).flush()

    def __getattr__(self, k):
        return getattr(self.real, k)


if not isinstance(sys.stdout, _ThreadTee):
    sys.stdout = _ThreadTee(sys.stdout)
    sys.stderr = _ThreadTee(sys.stderr)


@contextmanager
def _thread_capture(self):
    out, err = io.StringIO(), io.StringIO()
    sys.stdout.local.buf, sys.stderr.local.buf = out, err
    try:
        yield out, err
    finally:
        sys.stdout.local.buf = None
        sys.stderr.local.buf = None


rlm_envs.local_repl.LocalREPL._temp_cwd = _stable_cwd
rlm_envs.local_repl.LocalREPL._capture_output = _thread_capture
assert rlm_envs.local_repl.LocalREPL._temp_cwd is _stable_cwd, "cwd patch not live"
assert rlm_envs.local_repl.LocalREPL._capture_output is _thread_capture, (
    "capture patch not live"
)
print("LocalREPL concurrency patches ACTIVE", flush=True)

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
