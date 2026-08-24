"""Seaborn figures for the OOLONG campaign."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
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
    g = sns.catplot(
        data=df.drop_duplicates(["model", "spec", "concurrency", "repeat"]),
        x="concurrency",
        y="unit_wall_s",
        hue="mode",
        col="model",
        kind="bar",
        errorbar="sd",
        capsize=0.12,
        height=5,
        aspect=0.85,
    )
    g.set_axis_labels("concurrent tasks", "experiment wall time (s)")
    g.figure.suptitle("Whole-experiment runtime (8 OOLONG tasks), ±sd over 3 runs", y=1.04)
    save(g.figure, "unit_wall.png")

    # 2. per-task wall distributions
    g = sns.catplot(
        data=df,
        x="concurrency",
        y="wall_s",
        hue="mode",
        col="model",
        kind="box",
        height=5,
        aspect=0.85,
        showfliers=True,
    )
    g.set_axis_labels("concurrent tasks", "per-task wall time (s)")
    g.figure.suptitle("Per-task runtime distribution", y=1.04)
    save(g.figure, "task_wall_box.png")

    # 3. speedup heatmap
    piv = s.pivot_table(
        index="model", columns="concurrency", values="task_wall_mean_s", aggfunc="first"
    )  # placeholder shape
    sp = (
        s[~s.spec].set_index(["model", "concurrency"])["task_wall_mean_s"]
        / s[s.spec].set_index(["model", "concurrency"])["task_wall_mean_s"]
    ).unstack()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(sp, annot=True, fmt=".2f", cmap="crest", cbar_kws={"label": "×"}, ax=ax)
    ax.set_title("Speculation speedup (mean task wall, baseline/spec)")
    save(fig, "speedup_heatmap.png")

    # 4. sub-call latency distributions (from per-task lists in unit_stats)
    if "subcall_time_p50_s" in df:
        g = sns.catplot(
            data=df,
            x="mode",
            y="subcall_time_p50_s",
            col="model",
            kind="violin",
            height=5,
            aspect=0.8,
            cut=0,
        )
        g.set_axis_labels("", "per-task median sub-call time (s)")
        g.figure.suptitle("Sub-call latency (per-task medians)", y=1.04)
        save(g.figure, "subcall_violin.png")

    # 5. per-dataset scores (sanity: speculation must not change quality)
    g = sns.catplot(
        data=df,
        x="dataset",
        y="score",
        hue="mode",
        col="model",
        kind="bar",
        errorbar="sd",
        height=5,
        aspect=0.9,
    )
    g.set_axis_labels("", "score")
    g.figure.suptitle("Task scores by dataset (quality invariance check)", y=1.04)
    save(g.figure, "scores.png")

    # 6. turns/sub-calls per task
    g = sns.catplot(
        data=df,
        x="mode",
        y="n_subcalls",
        col="model",
        kind="strip",
        height=5,
        aspect=0.8,
        jitter=0.25,
    )
    g.set_axis_labels("", "# sub-calls per task")
    save(g.figure, "subcall_counts.png")


_PAL = {"Base RLM": "#4C72B0", "Speculative PTC": "#DD8452"}
_CELLS_CSV = Path("benchmark/oolong_campaign/runs/task_rows_corrected.csv")
_STORM_TASK_S = 2000.0
_TITLE = "Speculative PTC vs. Base RLM(Qwen3-30B-A3B-Instruct) on OOLONG/OOLONG-Pairs"


def _drop_storm_units(df: pd.DataFrame, max_task_s: float = _STORM_TASK_S) -> pd.DataFrame:
    key = ["block", "spec", "concurrency", "repeat"]
    keep = df.groupby(key)["wall_s"].transform("max") <= max_task_s
    return df[keep].copy()


def _repeat_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block, spec, conc, _rep), g in df.groupby(
        ["block", "spec", "concurrency", "repeat"]
    ):
        rows.append(
            {
                "temperature": "0.0" if block == "temp0" else "0.7",
                "mode": "Speculative PTC" if spec else "Base RLM",
                "concurrency": int(conc),
                "wall_s": float(g["unit_wall_s"].iloc[0]),
                "n_subcalls": float(g["n_subcalls"].mean()),
                "n_turns": float(g["n_turns"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_mean_speedup(
    csv: Path | None = None,
    out: Path | None = None,
    drop_storms: bool = True,
    filename: str = "mean_runtime_traj_no_outliers.png",
) -> None:
    """Two temperatures × wall / sub-calls / turns. Concurrency 4 and 8 only."""
    if csv is None:
        csv = _CELLS_CSV
    if out is None:
        out = Path("benchmark/oolong_campaign/runs/main/figures")
    out.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    df = df[df.model.str.contains("30B")].copy()
    if drop_storms:
        df = _drop_storm_units(df)
    long = _repeat_frame(df)
    long = long[long["concurrency"].isin((4, 8))]

    metrics = (
        ("wall_s", "Wall time (s)"),
        ("n_subcalls", "Sub-calls per task"),
        ("n_turns", "Turns per task"),
    )
    temps = ("0.0", "0.7")
    bar_kws = dict(
        x="concurrency",
        hue="mode",
        order=[4, 8],
        hue_order=["Base RLM", "Speculative PTC"],
        palette=_PAL,
        errorbar=("ci", 95),
        n_boot=8000,
        seed=0,
        capsize=0.08,
        err_kws={"linewidth": 1.1, "color": "#2b2b2b"},
        saturation=1,
        width=0.72,
        legend=False,
        zorder=3,
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.28)
    fig = plt.figure(figsize=(11.8, 6.8))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.14, 1.22, 1.05, 1.05],
        wspace=0.26,
        hspace=0.34,
        left=0.035,
        right=0.985,
        top=0.80,
        bottom=0.13,
    )
    axes = [[fig.add_subplot(gs[r, c + 1]) for c in range(3)] for r in range(2)]
    for c in range(3):
        axes[1][c].sharex(axes[0][c])
        if c < 2:
            axes[1][c].sharey(axes[0][c])

    for row, temp in enumerate(temps):
        lab = fig.add_subplot(gs[row, 0])
        lab.set_axis_off()
        lab.text(
            0.5,
            0.5,
            f"Temperature = {temp}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            color="#222222",
        )
        sub = long[long["temperature"] == temp]
        for col, (y, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            sns.barplot(data=sub, y=y, ax=ax, **bar_kws)
            ax.set_axisbelow(True)
            ax.set_xlabel("Concurrent tasks" if row == 1 else "")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=11)
            if row == 0:
                ax.set_title(ylabel, pad=12, fontsize=12)
                ax.tick_params(labelbottom=False)
            if y == "n_turns":
                ax.set_ylim(0, 13)
            ax.set_xticks([4, 8])

    fig.legend(
        [Patch(facecolor=_PAL[k], edgecolor="none") for k in _PAL],
        list(_PAL),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.91),
        fontsize=11,
        handlelength=1.2,
        columnspacing=1.8,
    )
    fig.suptitle(_TITLE, fontsize=13.5, y=0.985)
    fig.text(
        0.54,
        0.035,
        "Mean over repeats  ·  95% bootstrap CI  ·  temp 0.7 Base RLM at n=4 "
        "(units with a task >2000 s excluded)",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    path = out / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--mean-ci":
        plot_mean_speedup()
    else:
        plot_all(Path(args[0] if args else "benchmark/oolong_campaign/runs/main"))
