"""Seaborn figures for the OOLONG campaign."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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


_BASE = "Base RLM"
_SPEC = "Speculative PTC + RLM"
_FILL = {_BASE: "#3D4A54", _SPEC: "#C45C3E"}
_EDGE = {_BASE: "#2A333B", _SPEC: "#8B3A28"}
_HATCH = {_BASE: None, _SPEC: "///"}
_CELLS_CSV = Path("benchmark/oolong_campaign/runs/task_rows_corrected.csv")
_STORM_TASK_S = 2000.0
_TITLE = "Speculative PTC vs. Base RLM(Qwen3-30B-A3B-Instruct) on OOLONG/OOLONG-Pairs"
_TCRIT = {3: 4.302652729911275, 4: 3.182446305284341, 5: 2.7764451051977936}


def _drop_storm_units(df: pd.DataFrame, max_task_s: float = _STORM_TASK_S) -> pd.DataFrame:
    key = ["block", "spec", "concurrency", "repeat"]
    keep = df.groupby(key)["wall_s"].transform("max") <= max_task_s
    return df[keep].copy()


def _repeat_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block, spec, conc, _rep), g in df.groupby(["block", "spec", "concurrency", "repeat"]):
        rows.append(
            {
                "temperature": "0.0" if block == "temp0" else "0.7",
                "mode": _SPEC if spec else _BASE,
                "concurrency": int(conc),
                "wall_s": float(g["unit_wall_s"].iloc[0]),
                "n_subcalls": float(g["n_subcalls"].mean()),
                "n_turns": float(g["n_turns"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _mean_ci(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    mean = float(vals.mean())
    if n < 2:
        return mean, mean, mean
    se = float(vals.std(ddof=1) / np.sqrt(n))
    t = _TCRIT.get(n, 1.96)
    return mean, mean - t * se, mean + t * se


def plot_mean_speedup(
    csv: Path | None = None,
    out: Path | None = None,
    drop_storms: bool = True,
    filename: str = "mean_runtime_traj_no_outliers.png",
) -> None:
    """Two temperatures × wall / sub-calls / turns. Concurrency 4 and 8."""
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
    concs = (4, 8)
    modes = (_BASE, _SPEC)
    x = np.arange(len(concs))
    width = 0.34
    offsets = (-width / 2 - 0.02, width / 2 + 0.02)

    ink = "#1C1917"
    mute = "#57534E"
    rule = "#D6D1C7"
    paper = "#FAF8F5"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.facecolor": paper,
            "axes.facecolor": paper,
            "text.color": ink,
            "axes.edgecolor": rule,
            "axes.linewidth": 0.8,
            "xtick.color": mute,
            "ytick.color": mute,
            "hatch.linewidth": 0.7,
            "hatch.color": "#F4E6DC",
        }
    )

    fig = plt.figure(figsize=(12.0, 6.9))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.13, 1.22, 1.05, 1.05],
        wspace=0.28,
        hspace=0.36,
        left=0.04,
        right=0.985,
        top=0.80,
        bottom=0.13,
    )
    axes = [[fig.add_subplot(gs[r, c + 1]) for c in range(3)] for r in range(2)]
    row_hi = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    for row, temp in enumerate(temps):
        lab = fig.add_subplot(gs[row, 0])
        lab.set_axis_off()
        lab.set_facecolor(paper)
        lab.text(
            0.55,
            0.5,
            f"Temperature = {temp}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11.5,
            color=ink,
        )
        sub = long[long["temperature"] == temp]
        for col, (y, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            for i, mode in enumerate(modes):
                means, lo_err, hi_err = [], [], []
                for c in concs:
                    m, lo, hi = _mean_ci(
                        sub.loc[(sub["mode"] == mode) & (sub["concurrency"] == c), y]
                    )
                    lo = max(lo, 0.0)
                    means.append(m)
                    lo_err.append(m - lo)
                    hi_err.append(hi - m)
                    row_hi[row][col] = max(row_hi[row][col], hi)
                xpos = x + offsets[i]
                ax.bar(
                    xpos,
                    means,
                    width,
                    facecolor=_FILL[mode],
                    edgecolor=_EDGE[mode],
                    linewidth=0.7,
                    hatch=_HATCH[mode],
                    zorder=3,
                )
                ax.errorbar(
                    xpos,
                    means,
                    yerr=np.vstack([lo_err, hi_err]),
                    fmt="none",
                    ecolor=ink,
                    elinewidth=1.45,
                    capsize=4.0,
                    capthick=1.45,
                    zorder=5,
                )
            ax.set_xticks(x, [str(c) for c in concs])
            ax.set_xlim(-0.62, 1.62)
            ax.set_ylabel("")
            ax.set_xlabel("Concurrent tasks" if row == 1 else "")
            ax.yaxis.grid(True, color="#E8E4DC", linewidth=0.8)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(rule)
            ax.spines["bottom"].set_color(rule)
            ax.tick_params(length=0, labelsize=10.5, pad=4)
            if row == 0:
                ax.set_title(ylabel, pad=12, fontsize=12, color=ink, loc="center")
                ax.tick_params(labelbottom=False)

    for row in range(2):
        for col, (y, _) in enumerate(metrics):
            top = 13.0 if y == "n_turns" else row_hi[row][col] * 1.16
            axes[row][col].set_ylim(0, top)

    fig.legend(
        [
            Patch(
                facecolor=_FILL[m],
                edgecolor=_EDGE[m],
                linewidth=0.7,
                hatch=_HATCH[m],
                label=m,
            )
            for m in modes
        ],
        modes,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.915),
        fontsize=11,
        handlelength=1.6,
        handleheight=1.05,
        columnspacing=1.8,
    )
    fig.suptitle(_TITLE, fontsize=13.5, y=0.985, color=ink)
    note = (
        "Mean over repeats  ·  95% t-interval  ·  temp 0.7 Base RLM at n=4 "
        "(units with a task >2000 s excluded)"
        if drop_storms
        else "Mean over repeats  ·  95% t-interval  ·  all n=5 units included"
    )
    fig.text(
        0.54,
        0.035,
        note,
        ha="center",
        va="center",
        fontsize=9,
        color=mute,
    )
    path = out / filename
    fig.savefig(path, dpi=220, facecolor=paper)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--mean-ci":
        plot_mean_speedup()
        plot_mean_speedup(drop_storms=False, filename="mean_runtime_traj_with_outliers.png")
    else:
        plot_all(Path(args[0] if args else "benchmark/oolong_campaign/runs/main"))
