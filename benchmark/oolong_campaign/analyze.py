"""Aggregate campaign stats: per-config means + variance over repeats."""

import json
import sys
from pathlib import Path

import pandas as pd


def load_units(base: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(base.rglob("unit_stats.json")):
        u = json.loads(f.read_text())
        for t in u["tasks"]:
            rows.append({
                "model": u["model"].split("/")[-1], "spec": u["spec"],
                "concurrency": u["concurrency"], "repeat": u["repeat"],
                "unit_wall_s": u["unit_wall_s"], **{k: t.get(k) for k in (
                    "task_id", "dataset", "wall_s", "score", "ok", "n_turns",
                    "n_subcalls", "subcall_time_sum_s", "subcall_time_p50_s")},
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "spec", "concurrency", "repeat"]).agg(
        unit_wall_s=("unit_wall_s", "first"),
        task_wall_mean_s=("wall_s", "mean"),
        score=("score", "mean"),
        errors=("ok", lambda s: int((~s.astype(bool)).sum())),
        turns=("n_turns", "mean"), subcalls=("n_subcalls", "mean"),
    ).reset_index()
    out = g.groupby(["model", "spec", "concurrency"]).agg(
        unit_wall_mean_s=("unit_wall_s", "mean"),
        unit_wall_std_s=("unit_wall_s", "std"),
        task_wall_mean_s=("task_wall_mean_s", "mean"),
        task_wall_std_s=("task_wall_mean_s", "std"),
        score_mean=("score", "mean"), errors=("errors", "sum"),
        turns_mean=("turns", "mean"), subcalls_mean=("subcalls", "mean"),
        n_repeats=("repeat", "nunique"),
    ).reset_index()
    return out.round(2)


def speedups(summary: pd.DataFrame) -> pd.DataFrame:
    piv = summary.pivot_table(index=["model", "concurrency"], columns="spec",
                              values=["unit_wall_mean_s", "task_wall_mean_s"])
    piv[("speedup", "unit")] = piv[("unit_wall_mean_s", False)] / piv[("unit_wall_mean_s", True)]
    piv[("speedup", "task")] = piv[("task_wall_mean_s", False)] / piv[("task_wall_mean_s", True)]
    return piv.round(2)


if __name__ == "__main__":
    base = Path(sys.argv[1] if len(sys.argv) > 1 else
                "benchmark/oolong_campaign/runs/main")
    df = load_units(base)
    s = summarize(df)
    print(s.to_string(index=False))
    print("\n== speedups (baseline / spec) ==")
    print(speedups(s).to_string())
    s.to_csv(base / "summary.csv", index=False)
    df.to_csv(base / "task_rows.csv", index=False)
