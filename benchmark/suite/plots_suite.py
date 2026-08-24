"""Suite figures: paired walls per pattern, mechanism panels."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", palette=["#4C72B0", "#DD8452"])


def plot(tag: str = "v1") -> None:
    base = Path(__file__).parent / "results" / tag
    df = pd.read_csv(base / "runs.csv")
    out = base / "figures"
    out.mkdir(exist_ok=True)
    order = (df[df["mode"] == "baseline"].groupby("pattern").wall_s.mean()
             .sort_values(ascending=False).index)

    def save(fig, name):
        fig.savefig(out / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out / name)

    # 1. paired walls per pattern (the headline)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(data=df, x="pattern", y="wall_s", hue="mode", order=order,
                errorbar="sd", capsize=0.08, ax=ax)
    ax.set_ylabel("wall time (s), ±sd over 5 runs")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    cats = df.drop_duplicates("pattern").set_index("pattern").category
    for i, p in enumerate(order):
        ax.text(i, -0.02, {"easy": "A", "hard": "B", "edge": "C"}[cats[p]],
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=11, color="gray")
    ax.set_title(f"Speculation suite — wall time per test ({tag})")
    save(fig, "walls.png")

    # 2. speedup vs waste (does winning cost waste?)
    g = df.groupby(["pattern", "mode"]).agg(w=("wall_s", "mean"),
                                            ev=("evicted", "mean")).unstack()
    g["speedup"] = g[("w", "baseline")] / g[("w", "spec")]
    g["waste"] = g[("ev", "spec")]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=g.reset_index(), x="waste", y="speedup", s=140, ax=ax)
    for _, r in g.reset_index().iterrows():
        ax.annotate(r["pattern"].item() if hasattr(r["pattern"], "item") else r["pattern"],
                    (r["waste"], r["speedup"]), fontsize=10,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xlabel("evicted speculations per run (waste)")
    ax.set_ylabel("speedup (baseline/spec)")
    ax.set_title("Speedup vs waste")
    save(fig, "speedup_vs_waste.png")

    # 3. dispatch lead time (how early do calls get in flight?)
    sp = df[df["mode"] == "spec"]
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=sp, x="pattern", y="lead_s_median", order=order,
                errorbar="sd", ax=ax, color="#DD8452")
    ax.set_ylabel("median dispatch lead (s before stream end)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    save(fig, "lead_time.png")

    # 4. score parity check
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.barplot(data=df, x="pattern", y="score", hue="mode", order=order,
                errorbar="sd", ax=ax)
    ax.set_ylabel("score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.set_title("Correctness parity")
    save(fig, "scores.png")


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else "v1")
