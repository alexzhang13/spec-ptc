"""Seaborn figures for the OOLONG campaign."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from benchmark.oolong_campaign.analyze import load_units, summarize  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", palette=["#4C72B0", "#DD8452"])
LBL = {False: "baseline", True: "speculative"}


def plot_all(base: Path) -> None:
    out = base / "figures"
    out.mkdir(exist_ok=True)
    df = load_units(base)
    df["mode"] = df["spec"].map(LBL)
    s = summarize(df)
    s["mode"] = s["spec"].map(LBL)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=180)
        plt.close(fig)
        print("wrote", out / name)

    # 1. unit wall time by concurrency x model (bars + repeat variance)
    g = sns.catplot(data=df.drop_duplicates(["model", "spec", "concurrency", "repeat"]),
                    x="concurrency", y="unit_wall_s", hue="mode", col="model",
                    kind="bar", errorbar="sd", capsize=0.12, height=5, aspect=0.85)
    g.set_axis_labels("concurrent tasks", "experiment wall time (s)")
    g.figure.suptitle("Whole-experiment runtime (8 OOLONG tasks), ±sd over 3 runs", y=1.04)
    save(g.figure, "unit_wall.png")

    # 2. per-task wall distributions
    g = sns.catplot(data=df, x="concurrency", y="wall_s", hue="mode", col="model",
                    kind="box", height=5, aspect=0.85, showfliers=True)
    g.set_axis_labels("concurrent tasks", "per-task wall time (s)")
    g.figure.suptitle("Per-task runtime distribution", y=1.04)
    save(g.figure, "task_wall_box.png")

    # 3. speedup heatmap
    piv = s.pivot_table(index="model", columns="concurrency",
                        values="task_wall_mean_s", aggfunc="first")  # placeholder shape
    sp = (s[~s.spec].set_index(["model", "concurrency"])["task_wall_mean_s"] /
          s[s.spec].set_index(["model", "concurrency"])["task_wall_mean_s"]).unstack()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(sp, annot=True, fmt=".2f", cmap="crest", cbar_kws={"label": "×"}, ax=ax)
    ax.set_title("Speculation speedup (mean task wall, baseline/spec)")
    save(fig, "speedup_heatmap.png")

    # 4. sub-call latency distributions (from per-task lists in unit_stats)
    if "subcall_time_p50_s" in df:
        g = sns.catplot(data=df, x="mode", y="subcall_time_p50_s", col="model",
                        kind="violin", height=5, aspect=0.8, cut=0)
        g.set_axis_labels("", "per-task median sub-call time (s)")
        g.figure.suptitle("Sub-call latency (per-task medians)", y=1.04)
        save(g.figure, "subcall_violin.png")

    # 5. per-dataset scores (sanity: speculation must not change quality)
    g = sns.catplot(data=df, x="dataset", y="score", hue="mode", col="model",
                    kind="bar", errorbar="sd", height=5, aspect=0.9)
    g.set_axis_labels("", "score")
    g.figure.suptitle("Task scores by dataset (quality invariance check)", y=1.04)
    save(g.figure, "scores.png")

    # 6. turns/sub-calls per task
    g = sns.catplot(data=df, x="mode", y="n_subcalls", col="model", kind="strip",
                    height=5, aspect=0.8, jitter=0.25)
    g.set_axis_labels("", "# sub-calls per task")
    save(g.figure, "subcall_counts.png")


if __name__ == "__main__":
    plot_all(Path(sys.argv[1] if len(sys.argv) > 1 else
                  "benchmark/oolong_campaign/runs/main"))
