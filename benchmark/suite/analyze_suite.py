"""Summarise a suite tag: paired per-rep speedups, variance, engine counters,
correctness parity, and the pre-registered verdict.

  uv run python -m benchmark.suite.analyze_suite main [--md out.md]
"""

import argparse
from pathlib import Path

import pandas as pd

import benchmark.suite.patterns_v2  # noqa: F401
import benchmark.suite.patterns_v3  # noqa: F401
import benchmark.suite.patterns_v4  # noqa: F401
import benchmark.suite.patterns_v5  # noqa: F401
import benchmark.suite.patterns_v6  # noqa: F401
from benchmark.suite.expectations import (
    FLOOR,
    PARITY_EPS,
    WIN_MIN,
    claim,
    expected_dispatch,
    verdict,
)
from benchmark.suite.patterns import PATTERNS

NCALLS = {p.name: p.n_expected_calls for p in PATTERNS}

HERE = Path(__file__).parent
COUNTERS = [
    "dispatched",
    "peeked",
    "hits",
    "misses",
    "evicted",
    "shadow_stops",
    "calls_made",
    "calls_aborted",
    "chunks_out",
]


def _sign_flip_p(diffs: list[float]) -> float:
    """Exact one-sided permutation test on paired differences (base - spec).
    p = P(mean >= observed) when signs are exchangeable, i.e. the chance a
    speedup this large came from noise. Falls back to sampling above 2^16."""
    n = len(diffs)
    if n == 0 or all(d == 0 for d in diffs):
        return 1.0
    obs = sum(diffs)
    if n <= 16:
        hits = total = 0
        for mask in range(1 << n):
            s = sum(-d if (mask >> i) & 1 else d for i, d in enumerate(diffs))
            hits += s >= obs
            total += 1
        return hits / total
    import random

    rng = random.Random(0)
    hits = 0
    for _ in range(20000):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        hits += s >= obs
    return hits / 20000


def load(tag: str, max_rep: int = 0) -> pd.DataFrame:
    """tag may be comma-separated: runs collected under identical conditions
    (same model, same tps) are pooled for one table."""
    df = pd.concat(
        [pd.read_csv(HERE / "results" / t / "runs.csv") for t in tag.split(",")],
        ignore_index=True,
    )
    for c in COUNTERS + ["wall_s", "score", "lead_s_median", "call_s_median"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if max_rep:  # uniform n across patterns when a run was cut short
        df = df[df["rep"] <= max_rep]
    return df


def summarize(df: pd.DataFrame, treat: str = "spec") -> pd.DataFrame:
    """One row per (pattern, tps): paired speedup + variance + counters."""
    rows = []
    for (pat, tps), g in df.groupby(["pattern", "tps"], sort=False):
        b = g[g["mode"] == "baseline"].set_index("rep")
        s = g[g["mode"] == treat].set_index("rep")
        reps = sorted(set(b.index) & set(s.index))
        if not reps:
            continue
        ratios = (b.loc[reps, "wall_s"] / s.loc[reps, "wall_s"]).astype(float)
        sd = float(ratios.std(ddof=1)) if len(reps) > 1 else 0.0
        speed = float(ratios.mean())
        dscore = float(s.loc[reps, "score"].mean() - b.loc[reps, "score"].mean())
        row = {
            "pattern": pat,
            "category": g["category"].iloc[0],
            "tps": tps,
            "n": len(reps),
            "base_s": round(float(b.loc[reps, "wall_s"].mean()), 2),
            "base_sd": round(float(b.loc[reps, "wall_s"].std(ddof=1) or 0), 2),
            "spec_s": round(float(s.loc[reps, "wall_s"].mean()), 2),
            "spec_sd": round(float(s.loc[reps, "wall_s"].std(ddof=1) or 0), 2),
            "saved_s": round(float((b.loc[reps, "wall_s"] - s.loc[reps, "wall_s"]).mean()), 2),
            "speedup": round(speed, 2),
            "speedup_sd": round(sd, 2),
            "speedup_min": round(float(ratios.min()), 2),
            "base_cv": round(
                float(b.loc[reps, "wall_s"].std(ddof=1) / b.loc[reps, "wall_s"].mean()), 3
            ),
            "spec_cv": round(
                float(s.loc[reps, "wall_s"].std(ddof=1) / s.loc[reps, "wall_s"].mean()), 3
            ),
            "score_b": round(float(b.loc[reps, "score"].mean()), 3),
            "score_s": round(float(s.loc[reps, "score"].mean()), 3),
            "d_score": round(dscore, 3),
            "lead_s": round(float(s.loc[reps, "lead_s_median"].mean()), 2),
            "call_s": round(float(s.loc[reps, "call_s_median"].mean()), 2),
            "p_val": round(
                _sign_flip_p(
                    [float(x) for x in (b.loc[reps, "wall_s"] - s.loc[reps, "wall_s"])]
                ),
                4,
            ),
            "claim": claim(pat),
            "verdict": verdict(pat, speed, dscore),
        }
        for c in COUNTERS:
            if c in s:
                row[c] = round(float(s.loc[reps, c].mean()), 1)
        exp = expected_dispatch(pat, NCALLS.get(pat))
        got = row.get("dispatched")
        row["disp"] = (
            f"{int(got)}/{exp}"
            if exp is not None and got is not None
            else (f"{int(got)}/-" if got is not None else "-")
        )
        row["disp_ok"] = (
            "" if exp is None or got is None else ("yes" if int(got) == exp else "NO")
        )
        if "calls_made" in s:
            bc = float(b.loc[reps, "calls_made"].mean()) or 1.0
            bt = float(b.loc[reps, "chunks_out"].mean()) or 1.0
            row["calls_x"] = round(float(s.loc[reps, "calls_made"].mean()) / bc, 2)
            row["tokens_x"] = round(float(s.loc[reps, "chunks_out"].mean()) / bt, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(t: pd.DataFrame, cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in t.iterrows()]
    return "\n".join([head, rule, *body])


MAIN_COLS = [
    "pattern",
    "category",
    "n",
    "base_s",
    "spec_s",
    "speedup",
    "speedup_sd",
    "speedup_min",
    "saved_s",
    "lead_s",
    "dispatched",
    "peeked",
    "hits",
    "misses",
    "evicted",
    "shadow_stops",
    "disp",
    "disp_ok",
    "calls_x",
    "tokens_x",
    "p_val",
    "score_b",
    "score_s",
    "claim",
    "verdict",
]


def report(tag: str, max_rep: int = 0) -> str:
    df = load(tag, max_rep)
    t = summarize(df)
    t = t.sort_values(["category", "speedup"], ascending=[True, False])
    out = [
        f"### Per-pattern results ({tag}, n={int(t['n'].max())} paired reps)",
        "",
        md_table(t, MAIN_COLS),
        "",
    ]
    par = t[t["d_score"].abs() > PARITY_EPS]
    slow = t[t["speedup"] < FLOOR]
    out += [
        "### Invariants",
        "",
        f"- correctness parity (|Δscore| ≤ {PARITY_EPS}): "
        f"**{len(t) - len(par)}/{len(t)}** patterns"
        + ("" if par.empty else " — violations: " + ", ".join(par["pattern"])),
        f"- never-slower (speedup ≥ {FLOOR}): **{len(t) - len(slow)}/{len(t)}**"
        + ("" if slow.empty else " — violations: " + ", ".join(slow["pattern"])),
        f"- dispatch-count checks passed: "
        f"**{int((t['disp_ok'] == 'yes').sum())}/"
        f"{int((t['disp_ok'] != '').sum())}**"
        + (
            ""
            if (t["disp_ok"] != "NO").all()
            else " — mismatches: "
            + ", ".join(
                t[t["disp_ok"] == "NO"]["pattern"]
                + " ("
                + t[t["disp_ok"] == "NO"]["disp"]
                + ")"
            )
        ),
        f"- speedups significant at p<0.05 (exact paired sign-flip test): "
        f"**{int(((t['p_val'] < 0.05) & (t['speedup'] > 1)).sum())}/{len(t)}**",
        f"- worst SINGLE repetition below the {FLOOR}x floor: "
        f"**{int((t['speedup_min'] < FLOOR).sum())}/{len(t)}**"
        + (
            ""
            if (t["speedup_min"] >= FLOOR).all()
            else " — "
            + ", ".join(
                f"{r['pattern']} ({r['speedup_min']}x)"
                for _, r in t[t["speedup_min"] < FLOOR].iterrows()
            )
        ),
        f"- claimed wins met (≥ {WIN_MIN}x): "
        f"**{int((t[t['claim'] == 'win']['speedup'] >= WIN_MIN).sum())}"
        f"/{int((t['claim'] == 'win').sum())}**",
        "",
        "### Aggregates",
        "",
    ]
    for cat, g in t.groupby("category"):
        gm = float((g["speedup"].apply(lambda x: max(x, 1e-9))).pow(1 / len(g)).prod())
        out.append(
            f"- **{cat}** (n={len(g)}): geomean speedup {gm:.2f}x, "
            f"total wall {g['base_s'].sum():.0f}s -> {g['spec_s'].sum():.0f}s, "
            f"worst {g['speedup'].min():.2f}x"
        )
    gm_all = float(t["speedup"].apply(lambda x: max(x, 1e-9)).pow(1 / len(t)).prod())
    out += [
        "",
        f"- **all {len(t)} patterns**: geomean {gm_all:.2f}x, "
        f"wall {t['base_s'].sum():.0f}s -> {t['spec_s'].sum():.0f}s "
        f"({t['saved_s'].sum():.0f}s saved), "
        f"median run-to-run CV {t['base_cv'].median():.3f} (baseline) vs "
        f"{t['spec_cv'].median():.3f} (spec)",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--md", default="")
    ap.add_argument(
        "--max-rep", type=int, default=0, help="use only reps 1..N (uniform n across patterns)"
    )
    args = ap.parse_args()
    txt = report(args.tag, args.max_rep)
    print(txt)
    if args.md:
        Path(args.md).write_text(txt)
    summarize(load(args.tag, args.max_rep)).to_csv(
        HERE / "results" / args.tag.split(",")[0] / "summary.csv", index=False
    )


if __name__ == "__main__":
    main()
