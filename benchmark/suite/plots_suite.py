"""Suite figures (seaborn). One tag -> results/<tag>/figures/*.png

uv run python -m benchmark.suite.plots_suite main [--rates main,rate20,rate150]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import benchmark.suite.patterns_v2  # noqa: E402,F401
import benchmark.suite.patterns_v3  # noqa: E402,F401
import benchmark.suite.patterns_v4  # noqa: E402,F401
import benchmark.suite.patterns_v5  # noqa: E402,F401
import benchmark.suite.patterns_v6  # noqa: E402,F401
from benchmark.suite.analyze_suite import load, summarize  # noqa: E402
from benchmark.suite.patterns import PATTERNS  # noqa: E402

NCALLS = {p.name: (p.n_expected_calls or 0) for p in PATTERNS}

HERE = Path(__file__).parent
sns.set_theme(style="whitegrid", context="talk")
PAL = {"baseline": "#8896a6", "spec": "#2f7f4f"}


def _fig(path: Path, w: float, h: float):
    f, ax = plt.subplots(figsize=(w, h))
    return f, ax, path


def _save(f, path: Path) -> None:
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)
    print("wrote", path)


def walls(t: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    curated = t[t["category"] != "sweep"].sort_values("base_s", ascending=False)
    long = df[df["pattern"].isin(curated["pattern"])]
    order = list(curated["pattern"])
    f, ax = plt.subplots(figsize=(13, 0.42 * len(order) + 2.5))
    sns.barplot(
        data=long,
        y="pattern",
        x="wall_s",
        hue="mode",
        order=order,
        palette=PAL,
        errorbar="sd",
        ax=ax,
        capsize=0.25,
        err_kws={"lw": 1.2},
    )
    for i, p in enumerate(order):
        r = curated[curated["pattern"] == p].iloc[0]
        ax.text(
            long["wall_s"].max() * 1.01,
            i,
            f"{r['speedup']:.2f}x",
            va="center",
            fontsize=11,
            color="#2f7f4f" if r["speedup"] >= 1.3 else "#555",
        )
    n = int(curated["n"].max())
    ax.set(
        xlabel="turn wall-clock (s)",
        ylabel="",
        title=f"Curated patterns: baseline vs speculative (mean ± sd, n={n})",
    )
    ax.legend(title="", loc="lower right")
    _save(f, out / "walls.png")


def speedups(t: pd.DataFrame, out: Path) -> None:
    c = t[t["category"] != "sweep"].sort_values("speedup", ascending=False)
    colors = ["#2f7f4f" if v.startswith("SUPPORTED") else "#b3402f" for v in c["verdict"]]
    f, ax = plt.subplots(figsize=(11, 0.4 * len(c) + 2.5))
    ax.barh(
        c["pattern"],
        c["speedup"],
        xerr=c["speedup_sd"],
        color=colors,
        error_kw={"lw": 1, "ecolor": "#333"},
    )
    ax.axvline(1.0, color="#333", lw=1.5)
    ax.axvline(1.3, color="#888", lw=1, ls="--")
    ax.invert_yaxis()
    ax.set(
        xlabel="speedup (paired per-rep baseline/spec)",
        ylabel="",
        title="Speedup by pattern (green = pre-registered claim met)",
    )
    _save(f, out / "speedups.png")


def sweeps(t: pd.DataFrame, out: Path) -> None:
    w = t[t["pattern"].str.startswith("sweep_w")].copy()
    if not w.empty:
        w["width"] = w["pattern"].str[7:].astype(int)
        w = w.sort_values("width")
        f, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].errorbar(
            w["width"],
            w["speedup"],
            yerr=w["speedup_sd"],
            marker="o",
            color="#2f7f4f",
            capsize=4,
        )
        axes[0].axhline(1.0, color="#333", lw=1)
        axes[0].axvline(16, color="#b3402f", ls="--", lw=1.2)
        axes[0].text(16.4, w["speedup"].min(), "max_inflight=16", color="#b3402f", fontsize=11)
        axes[0].set(
            xscale="log",
            xlabel="independent calls in block (width)",
            ylabel="speedup",
            title="Speedup vs width",
        )
        axes[0].set_xticks(list(w["width"]))
        axes[0].set_xticklabels([str(x) for x in w["width"]])
        for m, col in (("base_s", "#8896a6"), ("spec_s", "#2f7f4f")):
            axes[1].plot(
                w["width"],
                w[m],
                marker="o",
                color=col,
                label="baseline" if m == "base_s" else "spec",
            )
        axes[1].set(xscale="log", xlabel="width", ylabel="wall (s)", title="Wall vs width")
        axes[1].set_xticks(list(w["width"]))
        axes[1].set_xticklabels([str(x) for x in w["width"]])
        axes[1].legend()
        _save(f, out / "sweep_width.png")

    ln = t[t["pattern"].str.startswith("sweep_len")].copy()
    if not ln.empty:
        ln["cap"] = ln["pattern"].str[9:].astype(int)
        ln = ln.sort_values("cap")
        f, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].errorbar(
            ln["cap"],
            ln["speedup"],
            yerr=ln["speedup_sd"],
            marker="o",
            color="#2f7f4f",
            capsize=4,
        )
        axes[0].axhline(1.0, color="#333", lw=1)
        axes[0].set(
            xlabel="sub-call token cap",
            ylabel="speedup",
            title="Speedup vs call length (width=8)",
        )
        axes[1].plot(ln["call_s"], ln["saved_s"], marker="o", color="#2f5f8f")
        for _, r in ln.iterrows():
            axes[1].annotate(
                f"{int(r['cap'])}t",
                (r["call_s"], r["saved_s"]),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=10,
            )
        axes[1].set(
            xlabel="measured median sub-call latency (s)",
            ylabel="wall-clock saved (s)",
            title="Absolute savings grow with call latency",
        )
        _save(f, out / "sweep_length.png")


def structure(t: pd.DataFrame, out: Path) -> None:
    """The headline claim: speedup tracks the block's parallel width."""
    c = t[t["category"] != "sweep"].copy()
    c["width"] = c["pattern"].map(NCALLS)
    c = c[c["width"] > 0]
    f, ax = plt.subplots(figsize=(11, 7.5))
    sns.scatterplot(
        data=c,
        x="width",
        y="speedup",
        hue="category",
        size="call_s",
        sizes=(60, 420),
        alpha=0.85,
        ax=ax,
        palette="deep",
    )
    ax.axhline(1.0, color="#333", lw=1.2)
    ax.set(
        xscale="log",
        xlabel="independent tool calls available in the turn",
        ylabel="speedup",
        title="Speedup tracks parallel width (marker size = sub-call latency)",
    )
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    for _, r in c.iterrows():
        if r["speedup"] > 2.4 or r["speedup"] < 1.05 or r["width"] >= 12:
            ax.annotate(
                r["pattern"],
                (r["width"], r["speedup"]),
                fontsize=8.5,
                textcoords="offset points",
                xytext=(8, 4),
            )
    ax.legend(fontsize=9, loc="upper left")
    _save(f, out / "structure.png")


def cost(t: pd.DataFrame, out: Path) -> None:
    """What speculation costs the server when it guesses wrong."""
    if "tokens_x" not in t:
        return
    c = t.dropna(subset=["tokens_x"]).copy()
    c = c[c["dispatched"] > 0]  # patterns with nothing to speculate: no cost axis
    f, ax = plt.subplots(figsize=(10.5, 7))
    sns.scatterplot(
        data=c, x="tokens_x", y="speedup", hue="category", s=140, ax=ax, palette="deep"
    )
    ax.axhline(1.0, color="#333", lw=1)
    ax.axvline(1.0, color="#333", lw=1, ls=":")
    for _, r in c.iterrows():
        if r["tokens_x"] > 1.1 or r["speedup"] < 1.02:
            ax.annotate(
                f"{r['pattern']} ({r['calls_x']:.2f}x calls)",
                (r["tokens_x"], r["speedup"]),
                fontsize=8.5,
                textcoords="offset points",
                xytext=(8, 3),
            )
    ax.set(
        xlabel="sub-call tokens generated, relative to baseline",
        ylabel="speedup",
        title="Extra GPU work is confined to the patterns that diverge",
    )
    _save(f, out / "cost.png")


def waste(t: pd.DataFrame, out: Path) -> None:
    c = t[t["category"] != "sweep"].copy()
    c["wasted"] = c["misses"].fillna(0) + c["evicted"].fillna(0)
    f, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=c, x="wasted", y="speedup", hue="category", s=140, ax=ax, palette="deep"
    )
    for _, r in c.iterrows():
        if r["wasted"] > 0 or r["speedup"] < 1.2 or r["speedup"] > 3:
            ax.annotate(
                r["pattern"],
                (r["wasted"], r["speedup"]),
                fontsize=9,
                textcoords="offset points",
                xytext=(7, 3),
            )
    ax.axhline(1.0, color="#333", lw=1)
    ax.set(
        xlabel="wasted speculation per turn (misses + evictions)",
        ylabel="speedup",
        title="Wasted work does not cost wall-clock",
    )
    _save(f, out / "waste.png")


def lead(t: pd.DataFrame, out: Path) -> None:
    c = t[t["category"] != "sweep"]
    f, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=c, x="lead_s", y="saved_s", hue="category", s=140, ax=ax, palette="deep"
    )
    lim = max(c["lead_s"].max(), c["saved_s"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls=":", color="#666")
    for _, r in c.iterrows():
        if r["saved_s"] > 0.6:
            ax.annotate(
                r["pattern"],
                (r["lead_s"], r["saved_s"]),
                fontsize=9,
                textcoords="offset points",
                xytext=(7, 3),
            )
    ax.set(
        xlabel="median head start per dispatch (s)",
        ylabel="wall-clock saved (s)",
        title="Savings vs head start (dotted = 1:1)",
    )
    _save(f, out / "lead_time.png")


def parity(t: pd.DataFrame, out: Path) -> None:
    f, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1.02], [0, 1.02], ls=":", color="#666")
    sns.scatterplot(
        data=t, x="score_b", y="score_s", hue="category", s=150, ax=ax, palette="deep"
    )
    ax.set(
        xlabel="baseline score",
        ylabel="speculative score",
        title="Correctness parity (all points on the diagonal)",
    )
    _save(f, out / "parity.png")


def rates(tags: list[str], out: Path) -> None:
    frames = []
    for tg in tags:
        p = HERE / "results" / tg / "runs.csv"
        if p.exists():
            frames.append(summarize(load(tg)))
    if len(frames) < 2:
        return
    a = pd.concat(frames)
    a = a[a["pattern"].isin(set.intersection(*[set(f["pattern"]) for f in frames]))]
    f, ax = plt.subplots(figsize=(11, 7))
    sns.lineplot(data=a, x="tps", y="speedup", hue="pattern", marker="o", ax=ax)
    ax.axhline(1.0, color="#333", lw=1)
    ax.set(
        xlabel="main-stream generation rate (tok/s)",
        ylabel="speedup",
        title="Speedup vs how fast the model emits the block",
    )
    ax.legend(fontsize=9, ncol=2)
    _save(f, out / "rate_sweep.png")


def crossmodel(tag_a: str, tag_b: str, out: Path) -> None:
    a = summarize(load(tag_a))[["pattern", "category", "speedup", "call_s"]]
    b = summarize(load(tag_b))[["pattern", "speedup", "call_s"]]
    m = a.merge(b, on="pattern", suffixes=("_a", "_b"))
    if m.empty:
        return
    ma = load(tag_a)["model"].iloc[0]
    mb = load(tag_b)["model"].iloc[0]
    f, ax = plt.subplots(figsize=(9.5, 8.5))
    lim = max(m["speedup_a"].max(), m["speedup_b"].max()) * 1.08
    ax.plot([1, lim], [1, lim], ls=":", color="#666")
    sns.scatterplot(
        data=m, x="speedup_a", y="speedup_b", hue="category", s=150, ax=ax, palette="deep"
    )
    for _, r in m.iterrows():
        if abs(r["speedup_b"] - r["speedup_a"]) > 1.0 or r["speedup_b"] > 6:
            ax.annotate(
                r["pattern"],
                (r["speedup_a"], r["speedup_b"]),
                fontsize=8.5,
                textcoords="offset points",
                xytext=(7, 3),
            )
    ax.set(
        xlabel=f"speedup on {ma}",
        ylabel=f"speedup on {mb}",
        title="Same programs, two sub-models (dotted = equal)",
    )
    _save(f, out / "crossmodel.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--rates", default="")
    ap.add_argument("--tag2", default="")
    ap.add_argument("--max-rep", type=int, default=0)
    args = ap.parse_args()
    out = HERE / "results" / args.tag.split(",")[0] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.tag, args.max_rep)
    t = summarize(df)
    walls(t, df, out)
    speedups(t, out)
    structure(t, out)
    cost(t, out)
    sweeps(t, out)
    waste(t, out)
    lead(t, out)
    parity(t, out)
    if args.rates:
        rates(args.rates.split(","), out)
    if args.tag2:
        crossmodel(args.tag, args.tag2, out)


if __name__ == "__main__":
    main()
