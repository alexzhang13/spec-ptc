"""Run one campaign unit: N OOLONG tasks through RLM at a given concurrency,
with or without speculation, everything logged."""

import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rlm.environments as rlm_envs
from rlm import RLM
from rlm.logger.rlm_logger import RLMLogger

from benchmark.oolong_campaign.data import Task, score
from benchmark.oolong_campaign.logsys import TrajectoryLogger

_STOCK_LOCAL_REPL = rlm_envs.LocalREPL


def _set_speculation(enabled: bool) -> None:
    from demo.rlm import SpeculativeLocalREPL

    cls = SpeculativeLocalREPL if enabled else _STOCK_LOCAL_REPL
    rlm_envs.LocalREPL = cls
    rlm_envs.local_repl.LocalREPL = cls


def _walk_subcall_times(obj, acc: list) -> None:
    """Collect every execution_time under any rlm_calls list, whatever the shape."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rlm_calls" and isinstance(v, list):
                for c in v:
                    if isinstance(c, dict) and c.get("execution_time") is not None:
                        acc.append(round(float(c["execution_time"]), 3))
            else:
                _walk_subcall_times(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk_subcall_times(v, acc)


def _traj_stats(metadata) -> dict:
    out = {"n_turns": 0, "n_subcalls": 0, "subcall_times_s": [], "turn_ts": []}
    if not isinstance(metadata, dict):
        return out
    iters = metadata.get("iterations")
    if isinstance(iters, list):
        out["n_turns"] = len(iters)
        for it in iters:
            if isinstance(it, dict):
                out["turn_ts"].append(it.get("timestamp"))
    elif isinstance(iters, int):
        out["n_turns"] = iters
    _walk_subcall_times(metadata, out["subcall_times_s"])
    out["n_subcalls"] = len(out["subcall_times_s"])
    return out


def _run_task(
    task: Task,
    model: str,
    base_url: str,
    spec: bool,
    log: TrajectoryLogger,
    outdir: Path,
    max_iterations: int = 12,
    max_timeout: float = 900.0,
) -> dict:
    tid = task.task_id
    env_kwargs = {}
    if spec:
        from spec_ptc.contracts.events import EventBus

        bus = EventBus()
        bus.record = False
        bus.subscribe(
            lambda ev: log.log(
                "spec_" + ev.kind, task_id=tid, **{k: str(v)[:120] for k, v in ev.data.items()}
            )
        )
        env_kwargs["spec_bus"] = bus

    rlogger = RLMLogger(log_dir=str((outdir / "rlm_trajs").resolve()), file_name=tid)
    r = RLM(
        backend="openai",
        backend_kwargs={"model_name": model, "base_url": base_url, "api_key": "EMPTY"},
        environment="local",
        environment_kwargs=env_kwargs,
        max_iterations=max_iterations,
        max_timeout=max_timeout,
        sub_sampling_args={"max_tokens": 4096},
        logger=rlogger,
        verbose=False,
    )
    t0 = time.time()
    log.log(
        "task_begin",
        task_id=tid,
        dataset=task.dataset,
        spec=spec,
        ctx_chars=len(task.context),
        question=task.question[:120],
    )
    row = {"task_id": tid, "dataset": task.dataset, "spec": spec}
    try:
        out = r.completion(
            prompt=task.context, root_prompt=f"{task.hint}\n\nQuestion: {task.question}"
        )
        final = getattr(out, "response", "") or ""
        meta = getattr(out, "metadata", None)
        ts = _traj_stats(meta)
        sc = ts["subcall_times_s"]
        row.update(
            ok=True,
            score=score(task, str(final)),
            final_answer=str(final)[:300],
            rlm_execution_time_s=round(getattr(out, "execution_time", 0.0), 2),
            usage=str(getattr(out, "usage_summary", ""))[:300],
            n_turns=ts["n_turns"],
            n_subcalls=ts["n_subcalls"],
            subcall_time_sum_s=round(sum(sc), 2),
            subcall_time_p50_s=round(statistics.median(sc), 3) if sc else 0.0,
            subcall_time_max_s=round(max(sc), 3) if sc else 0.0,
            subcall_times_s=sc,
        )
    except Exception as e:
        row.update(
            ok=False,
            error=f"{type(e).__name__}: {e}",
            trace=traceback.format_exc()[-800:],
            score=0.0,
        )
    row["wall_s"] = round(time.time() - t0, 2)
    log.log("task_end", **row)
    return row


def run_unit(
    model: str,
    base_url: str,
    spec: bool,
    concurrency: int,
    repeat: int,
    tasks: list[Task],
    outdir: Path,
) -> dict:
    outdir = Path(outdir).resolve()
    log = TrajectoryLogger(outdir)
    _set_speculation(spec)
    log.log(
        "unit_begin",
        model=model,
        spec=spec,
        concurrency=concurrency,
        repeat=repeat,
        n_tasks=len(tasks),
    )
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(
                pool.map(lambda t: _run_task(t, model, base_url, spec, log, outdir), tasks)
            )
    finally:
        _set_speculation(False)
    unit_wall = time.time() - t0
    walls = [r["wall_s"] for r in rows]
    stats = {
        "model": model,
        "spec": spec,
        "concurrency": concurrency,
        "repeat": repeat,
        "unit_wall_s": round(unit_wall, 2),
        "task_wall_mean_s": round(statistics.mean(walls), 2),
        "task_wall_p50_s": round(statistics.median(walls), 2),
        "task_wall_max_s": round(max(walls), 2),
        "score_mean": round(statistics.mean(r["score"] for r in rows), 3),
        "n_errors": sum(1 for r in rows if not r.get("ok")),
        "tasks": rows,
    }
    log.log("unit_end", **{k: v for k, v in stats.items() if k != "tasks"})
    log.write_json("unit_stats.json", stats)
    log.close()
    return stats
