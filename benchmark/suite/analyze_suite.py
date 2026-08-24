"""Suite aggregation: per-pattern speedups, waste, lead time, score parity."""

import sys
from pathlib import Path

import pandas as pd


def load(tag: str = "v1") -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "results" / tag / "runs.csv")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["pattern", "category", "mode"]).agg(
        wall_mean=("wall_s", "mean"), wall_std=("wall_s", "std"),
        score=("score", "mean"), hits=("hits", "mean"), misses=("misses", "mean"),
        evicted=("evicted", "mean"), peeked=("peeked", "mean"),
        adopted=("adopted", "mean"), lead_s=("lead_s_median", "mean"),
        n=("rep", "count")).reset_index()
    piv = g.pivot_table(index=["pattern", "category"], columns="mode")
    piv[("speedup", "")] = (piv[("wall_mean", "baseline")] /
                            piv[("wall_mean", "spec")]).round(2)
    return g, piv


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    df = load(tag)
    g, piv = summarize(df)
    cols = [("wall_mean", "baseline"), ("wall_std", "baseline"),
            ("wall_mean", "spec"), ("wall_std", "spec"), ("speedup", ""),
            ("score", "baseline"), ("score", "spec"),
            ("evicted", "spec"), ("misses", "spec"), ("lead_s", "spec")]
    print(piv[cols].round(2).sort_values(("speedup", ""), ascending=False).to_string())
