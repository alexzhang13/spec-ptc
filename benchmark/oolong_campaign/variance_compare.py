"""temp0 vs temp0.7: how much variance did greedy decoding remove?"""

import sys
from pathlib import Path

import pandas as pd

from benchmark.oolong_campaign.analyze import load_units

ROOT = Path("benchmark/oolong_campaign/runs")


def cells(run: str, model_substr: str = "30B") -> pd.DataFrame:
    df = load_units(ROOT / run)
    df = df[df.model.str.contains(model_substr)].copy()
    df["mode"] = df["spec"].map({False: "baseline", True: "spec"})
    return df


def main(model_substr: str = "30B") -> None:
    hot = cells("main", model_substr)
    cold = cells("temp0", model_substr)

    print("== per-cell task-wall std (s): temp0.7 -> temp0 ==")
    for (m, c), g7 in hot.groupby(["mode", "concurrency"]):
        g0 = cold[(cold["mode"] == m) & (cold.concurrency == c)]
        if g0.empty:
            continue
        print(
            f"  {m:9s} c{c}: std {g7.wall_s.std():7.0f} -> {g0.wall_s.std():6.0f}   "
            f"mean {g7.wall_s.mean():6.0f} -> {g0.wall_s.mean():6.0f}   "
            f"CV {g7.wall_s.std() / g7.wall_s.mean():4.2f} -> {g0.wall_s.std() / g0.wall_s.mean():4.2f}"
        )

    print("\n== repeat-consistency at temp0: do the 3 repeats produce the same trajectory? ==")
    print("(per task+mode+conc: spread of turns and sub-calls across repeats)")
    g = cold.groupby(["mode", "concurrency", "task_id"]).agg(
        turns_spread=("n_turns", lambda s: int(s.max() - s.min())),
        subs_spread=("n_subcalls", lambda s: int(s.max() - s.min())),
        wall_cv=("wall_s", lambda s: round(s.std() / max(s.mean(), 1), 2)),
        n=("wall_s", "size"),
    )
    ident = (g.turns_spread == 0) & (g.subs_spread == 0)
    print(f"  identical trajectory shape across repeats: {ident.sum()}/{len(g)} cells")
    print(
        f"  median wall CV within identical-shape cells: "
        f"{g[ident].wall_cv.median() if ident.any() else float('nan')}"
    )
    worst = g.sort_values("subs_spread", ascending=False).head(6)
    print("  worst residual divergences:")
    print(worst.to_string())

    print("\n== spec vs baseline at temp0 (unit walls, median [min-max]) ==")
    u = cold.drop_duplicates(["mode", "concurrency", "repeat"])
    piv = u.pivot_table(
        index="concurrency", columns="mode", values="unit_wall_s", aggfunc="median"
    )
    if {"baseline", "spec"} <= set(piv.columns):
        piv["speedup"] = (piv.baseline / piv.spec).round(2)
    print(piv.round(0).to_string())
    for (m, c), g in u.groupby(["mode", "concurrency"]):
        ws = sorted(g.unit_wall_s)
        print(f"  {m:9s} c{c}: {[round(w) for w in ws]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "30B")
