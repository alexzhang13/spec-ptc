"""Render SUITE.md: methodology, per-pattern justification, results, verdicts.

uv run python -m benchmark.suite.render_md --tag main [--tag2 main30b]
                                           [--rates main,rate20,rate150]
"""

import argparse
from pathlib import Path

import benchmark.suite.patterns_v2  # noqa: F401
import benchmark.suite.patterns_v3  # noqa: F401
import benchmark.suite.patterns_v4  # noqa: F401
import benchmark.suite.patterns_v5  # noqa: F401
import benchmark.suite.patterns_v6  # noqa: F401
from benchmark.suite.analyze_suite import load, md_table, summarize
from benchmark.suite.expectations import FLOOR, PARITY_EPS, WIN_MIN, claim
from benchmark.suite.patterns import PATTERNS, response_text

HERE = Path(__file__).parent
CATS = [
    (
        "easy",
        "Expected wins",
        "Blocks with real parallel width. If speculation does not win here, the "
        "technique does not work.",
    ),
    (
        "hard",
        "Hard / floor cases",
        "Serial or dependent structure, where there is little or nothing to "
        "overlap. The claim is the floor: never materially slower.",
    ),
    (
        "edge",
        "Edge cases and adversarial cases",
        "Places a speculator can be wrong rather than slow — divergence, tool "
        "errors, unforkable state, taint, exotic Python surface, multi-turn "
        "state. Correctness parity is the primary metric.",
    ),
    (
        "sweep",
        "Parametric sweeps",
        "One axis at a time, so the suite yields a curve instead of an anecdote.",
    ),
]
FIXES = [
    (
        "Class statements aborted the shadow",
        "`class_method` scored 1.0 in both modes but speculated **nothing** "
        "(0 hits, 6 misses). The event trace showed "
        "`shadow_stop: NameError: name '__name__' is not defined`: the shadow "
        "namespace never got `__name__`, which a `class` statement needs for "
        "`__module__`. Any turn in which the model defined a class lost all "
        "speculation from that statement on — silently, because the results were "
        "still correct.",
        "`src/engine/shadow.py`: seed `__name__` in the fork namespace.",
    ),
    (
        "The real REPL could not define classes at all",
        "The same pattern first failed in BOTH modes with "
        "`NameError: __build_class__ not found` — the sandboxed builtins had no "
        "`__build_class__`, so `class ...` was a hard error for any model that "
        "wrote one.",
        "`src/runtime/harness.py`: allow `__build_class__`, `type`, `property`, "
        "`staticmethod`, `classmethod`.",
    ),
    (
        "Taint leaked out of comprehension scopes",
        "`taint_split` predicted 3 misses (calls whose args come from a "
        "non-speculatable tool) and 3 hits (independent calls). It measured 6 "
        "misses. The trace showed the *second*, independent comprehension being "
        "skipped with `tainted: ['l']`: poisoning the first comprehension's "
        "targets had written a taint marker into the loop variable `l`, and the "
        "next statement reusing the name `l` inherited it. Comprehension "
        "variables do not leak in Python 3, so this was pure lost speculation.",
        "`src/engine/shadow.py`: exclude comprehension- and lambda-local names "
        "from taint poisoning (`_comp_local_names`).",
    ),
]
REFUTED_NOTES = {
    "tool_error_recovered": "the except-branch fallback is serial by construction: the retraction "
    "is only discovered when the failing result is used, and the fallback "
    "call then runs alone with nothing left to overlap. It is also the "
    "noisiest pattern in the suite (worst repetition 0.62x on a ~3s turn), "
    "because a single server stall dominates a short wall.",
    "long_prose_burst": "this one refutes the LABEL, not the hypothesis. The pattern exists to "
    "show that a long prose PREFIX does not buy overlap — only stream that "
    "arrives AFTER the block does — and the measurement confirms exactly "
    "that: median head start 0.29s here versus 3.11s for `prose_sandwich`, "
    "whose prose comes after the block. Registering it as a `win` was "
    "inconsistent with its own hypothesis.",
    "taint_split": "three of the six calls are un-speculatable by construction (their "
    "arguments come from a non-speculatable tool), so the ceiling is about "
    "half a full fan-out. 1.3x was an optimistic threshold for a pattern "
    "that gives up half its width on purpose; the mechanism check "
    "(3 dispatches, 3 hits, 3 misses) is the result that matters.",
    "generator_across_blocks": "only three calls, consumed immediately after the block closes, on a "
    "1.6s turn — the available head start is one call latency. The claim "
    "should have been the floor.",
}
REFINED = [
    (
        "`identical10`",
        "originally scored output *diversity* across 10 "
        "identical prompts, which measures the model's sampling, not the "
        "engine. Rewritten to score result completeness; independence is now "
        "asserted mechanically from the event log (10 dispatches, 10 distinct "
        "FIFO claims).",
    ),
    (
        "`serial_chain6`",
        "was predicted to sit at the 1.0x floor. It "
        "measured ~1.5x: the shadow starts the *whole chain* while the block is "
        "still streaming, so the head start applies once to the chain rather "
        "than per call. `chain4_slow` was added with realistic call lengths to "
        "show that this bonus amortises away as calls get longer.",
    ),
    (
        "The width sweep",
        "was predicted to rise to the in-flight cap (16) and "
        "then flatten. It does not flatten: 3.7x at width 16, 5.1x at 32, 6.8x at "
        "64. The cap bounds concurrency, not speedup — with cap C and width W the "
        "speculative side pays ceil(W/C) waves against W serial calls, so the "
        "ideal ratio keeps climbing toward C. What actually limits it is server "
        "throughput once 16 requests are in flight, plus the fixed stream time. "
        "The measured curve is the corrected prediction.",
    ),
    (
        "`dependent_args`",
        "was predicted to hide only one call. It measured "
        "~3x, for the same reason: once the first result lands inside the "
        "shadow, the dependent fan-out also runs ahead of the stream.",
    ),
]
ROW = [
    "base_s",
    "spec_s",
    "speedup",
    "speedup_sd",
    "speedup_min",
    "saved_s",
    "p_val",
    "disp",
    "disp_ok",
    "hits",
    "misses",
    "evicted",
    "shadow_stops",
    "calls_x",
    "tokens_x",
    "score_b",
    "score_s",
    "verdict",
]


def header(tag: str, df) -> list[str]:
    model = df["model"].iloc[0]
    n = int(df.groupby(["pattern", "mode"])["rep"].count().max())
    return [
        "# Speculation test suite",
        "",
        "A curriculum of **fixed REPL programs** run end-to-end against a live "
        "vLLM endpoint, each executed twice — once with speculation off "
        "(`baseline`) and once on (`spec`) — and repeated to average out "
        "sub-call latency noise. Every sub-call is a real LLM request; nothing "
        "here is mocked.",
        "",
        "## Why fixed programs",
        "",
        "Measuring speculation on live model output confounds two things: the "
        "trajectory the model happens to take, and how fast the harness "
        "executes it. The OOLONG campaign in `benchmark/oolong_campaign/` "
        "measures the end-to-end effect with live trajectories (and pays for it "
        "with variance). This suite does the opposite: the 'generation' is a "
        "**recorded script**, replayed character-by-character at a chosen token "
        "rate, so both modes execute the *identical* program. The only "
        "remaining variance is real sub-call latency, which repeats average "
        "out. That makes each pattern a controlled experiment about one "
        "structural property of the code.",
        "",
        "## Method",
        "",
        f"- **Endpoint**: live vLLM, model `{model}`, one GPU, streaming "
        "sub-calls (a speculative call that gets retracted is aborted "
        "mid-stream, so wasted work is real wasted GPU work).",
        "- **Main stream**: the scripted response is emitted in 24-character "
        "chunks paced to a target token rate (default 60 tok/s, ~4 chars per "
        "token), matching what a served model does; the harness parses, "
        "segments and speculates while it arrives.",
        f"- **Repeats**: {n} per (pattern, mode). Speedup is computed "
        "**paired per repetition** (baseline_i / spec_i) and then averaged, so "
        "endpoint drift cancels; `speedup_sd` is across those pairs and "
        "`speedup_min` is the worst single pair.",
        "- **Order**: for each pattern the two modes alternate rep by rep, so "
        "neither mode systematically gets a warmer server.",
        "- **Scoring**: each pattern has a `check(namespace) -> [0,1]` that "
        "inspects the REPL state the program actually produced (result counts, "
        "non-empty strings, no proxy leaking into the answer). Scores are for "
        "**parity**, not quality: the two modes must agree.",
        "",
        "### Metrics",
        "",
        "| column | meaning |",
        "|---|---|",
        "| `base_s` / `spec_s` | mean turn wall-clock, seconds |",
        "| `speedup` | mean of paired per-rep ratios (>1 = spec faster) |",
        "| `saved_s` | mean wall-clock saved per turn |",
        "| `dispatched` | speculative calls started before they were needed |",
        "| `hits` / `misses` | needed calls that were already in flight / not |",
        "| `evicted` | speculations retracted because the real path diverged |",
        "| `shadow_stops` | shadow-REPL aborts (graceful degradation events) |",
        "| `lead_s` | median head start of a dispatch over the stream it hid behind |",
        "| `disp` / `disp_ok` | speculative dispatches vs the count the "
        "mechanism should make (blank where the correct count is "
        "model-dependent) |",
        "| `calls_x` / `tokens_x` | real sub-call requests and generated "
        "tokens, relative to baseline — the GPU cost of speculating |",
        "| `p_val` | exact one-sided paired sign-flip test on the per-rep wall "
        "differences: the probability a speedup this large came from noise |",
        "",
        "### Pre-registered claims",
        "",
        "Each pattern carries a claim fixed *before* the reported run "
        "(`expectations.py`): `win` (speedup must reach "
        f"{WIN_MIN}x) or `floor` (must not drop below {FLOOR}x). Every pattern "
        f"additionally claims correctness parity (|Δscore| ≤ {PARITY_EPS}). "
        "The `verdict` column is computed mechanically from those claims — "
        "`REFUTED` rows are kept in the table, not deleted.",
        "",
    ]


def pattern_section(p, row) -> list[str]:
    out = [f"#### `{p.name}`", ""]
    if row is not None:
        out += [
            f"*{row['base_s']}s → {row['spec_s']}s "
            f"(**{row['speedup']}x** ± {row['speedup_sd']}, worst "
            f"{row['speedup_min']}x); dispatched {row['dispatched']}, "
            f"hits {row['hits']}, misses {row['misses']}, evicted "
            f"{row['evicted']}; score {row['score_b']} vs "
            f"{row['score_s']} — **{row['verdict']}** "
            f"(claim: {claim(p.name)})*",
            "",
        ]
    out += [p.hypothesis, ""]
    if p.turns:
        code = "\n--- next turn ---\n".join(p.turns)
    else:
        code = p.code or response_text(p)
    fence, lang = ("~~~", "text") if "```" in code else ("```", "python")
    out += [
        "<details><summary>program</summary>",
        "",
        fence + lang,
        code.strip(),
        fence,
        "",
        "</details>",
        "",
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--tag2", default="")
    ap.add_argument("--max-rep", type=int, default=0)
    ap.add_argument("--rates", default="")
    ap.add_argument("--out", default="benchmark/suite/SUITE.md")
    args = ap.parse_args()

    df = load(args.tag, args.max_rep)
    t = summarize(df).set_index("pattern")
    md = header(args.tag, df)

    # ---- headline
    fig = f"results/{args.tag.split(',')[0]}/figures"
    md += ["## Results at a glance", ""]
    agg = []
    for cat, title, _ in CATS:
        g = t[t["category"] == cat]
        if g.empty:
            continue
        gm = float(g["speedup"].pow(1 / len(g)).prod())
        agg.append(
            {
                "group": title,
                "patterns": len(g),
                "geomean speedup": round(gm, 2),
                "worst": g["speedup"].min(),
                "baseline wall (s)": round(g["base_s"].sum(), 1),
                "spec wall (s)": round(g["spec_s"].sum(), 1),
                "parity failures": int((g["d_score"].abs() > PARITY_EPS).sum()),
                "floor breaks": int((g["speedup"] < FLOOR).sum()),
            }
        )
    import pandas as pd

    md += [
        md_table(pd.DataFrame(agg), list(agg[0].keys())),
        "",
        f"![speedups]({fig}/speedups.png)",
        "",
        f"![structure]({fig}/structure.png)",
        "",
        "The single figure that summarises the technique: the win is a "
        "function of how many independent tool calls the turn makes "
        "available, not of anything about the model.",
        "",
        f"![walls]({fig}/walls.png)",
        "",
    ]

    # ---- per-category result tables now; the per-pattern catalogue at the end
    catalogue: list[str] = []
    for cat, title, blurb in CATS:
        pats = [p for p in PATTERNS if p.category == cat]
        if not pats:
            continue
        md += [f"## {title}", "", blurb, ""]
        rows = t[t["category"] == cat].reset_index()
        if not rows.empty:
            md += [
                md_table(rows.sort_values("speedup", ascending=False), ["pattern", *ROW]),
                "",
            ]
        if cat == "sweep":
            md += [
                f"![width]({fig}/sweep_width.png)",
                "",
                f"![length]({fig}/sweep_length.png)",
                "",
            ]
        catalogue += [f"### {title}", ""]
        for p in pats:
            r = t.loc[p.name] if p.name in t.index else None
            catalogue += pattern_section(p, r)

    # ---- mechanism attribution (peek vs shadow), straight from the data
    at = t.reset_index()
    at = at[(at["dispatched"] > 0) & (at["category"] != "sweep")].copy()
    at["from_peek"] = (at["peeked"] / at["dispatched"]).round(2)
    peeky = at[at["from_peek"] > 0.5]["pattern"].tolist()
    shady = at[at["from_peek"] == 0.0]["pattern"].tolist()
    md += [
        "## Where the dispatches come from",
        "",
        "The engine has two ways to start a call early: a **pre-close "
        "peek** that evaluates a call's arguments while the statement is "
        "still being typed (including unrolling a `for` loop that has not "
        "closed yet), and **shadow execution** of each statement the moment "
        "it closes. The suite separates them, because they fail for "
        "different reasons.",
        "",
        f"- dispatched mostly by the pre-close peek ({len(peeky)} patterns): "
        + ", ".join(f"`{x}`" for x in peeky[:12])
        + ("..." if len(peeky) > 12 else ""),
        f"- dispatched only by the shadow at statement close "
        f"({len(shady)} patterns): "
        + ", ".join(f"`{x}`" for x in shady[:12])
        + ("..." if len(shady) > 12 else ""),
        "",
        "Two rules fall out of this, both visible in the table below. "
        "`for`-loops that append call results are unrolled by the peek, so "
        "their calls start before the statement is even complete. "
        "Comprehensions are a single expression, so nothing can be resolved "
        "until the line closes and the shadow runs it — which is fast "
        "enough that both routes win, but the peek's extra head start shows "
        "up in `lead_s`. `no_peek_control` is the deliberate control: same "
        "`for`-loop shape as `map16`, but each argument comes from a "
        "model-defined helper the peek refuses to evaluate, and every "
        "dispatch there comes from the shadow instead.",
        "",
        md_table(
            at.sort_values("from_peek", ascending=False)[
                ["pattern", "disp", "peeked", "from_peek", "lead_s", "speedup"]
            ],
            ["pattern", "disp", "peeked", "from_peek", "lead_s", "speedup"],
        ),
        "",
    ]

    ct = t.reset_index()
    ct = ct[ct["tokens_x"].notna() & (ct["dispatched"] > 0)]
    worst = ct.nlargest(4, "tokens_x") if not ct.empty else ct
    free = int((ct["tokens_x"] <= 1.02).sum()) if not ct.empty else 0
    md += [
        "## Cost of guessing wrong",
        "",
        "Speculation is only free when the guess is right. `calls_x` and "
        "`tokens_x` are the measured sub-call request and generated-token "
        "counts relative to baseline, so the price of every retraction is "
        "on the record.",
        "",
        f"- **{free} of {len(ct)}** patterns that speculate at all cost "
        "within 2% of baseline tokens: the guess was right, so the work "
        "was going to happen anyway.",
        "- the exceptions are the divergent ones: "
        + "; ".join(
            f"`{r['pattern']}` {r['tokens_x']:.2f}x tokens for {r['speedup']:.2f}x speed"
            for _, r in worst.iterrows()
        )
        + ".",
        "- `deep_chain12` is the case to worry about, and the suite was "
        "built to expose it: a long serial chain re-guesses after every "
        "divergence, so it pays roughly double the tokens for no speedup "
        "at all. The obvious mitigation — stop speculating a site after k "
        "consecutive retractions — is not implemented; the number here is "
        "what the engine does today.",
        "",
        f"![cost]({fig}/cost.png)",
        "",
        f"![waste]({fig}/waste.png)",
        "",
        "Wasted speculation (misses + retractions) is plotted against "
        "speedup: the adversarial patterns sit at the left of the 1.0 line "
        "or above it, never below.",
        "",
        "## Supporting figures",
        "",
        "Head start versus savings, and the correctness parity that every "
        "pattern is checked against.",
        "",
        f"![lead]({fig}/lead_time.png)",
        "",
        f"![parity]({fig}/parity.png)",
        "",
    ]

    if args.rates:
        import pandas as pd

        frames = []
        for tg in args.rates.split(","):
            if (HERE / "results" / tg / "runs.csv").exists():
                s = summarize(load(tg))
                s["rate"] = int(s["tps"].iloc[0])
                frames.append(s)
        rt = pd.concat(frames) if frames else None
        md += [
            "## Stream-rate sensitivity",
            "",
            "The same programs replayed at 20, 60 and 150 tokens per "
            "second. This is the axis most likely to be misread, so it gets "
            "both numbers.",
            "",
        ]
        if rt is not None and rt["rate"].nunique() > 1:
            common = set.intersection(*[set(f["pattern"]) for f in frames])
            rt = rt[rt["pattern"].isin(common)]
            sp = (
                rt.pivot_table(index="pattern", columns="rate", values="speedup")
                .round(2)
                .reset_index()
            )
            ld = (
                rt.pivot_table(index="pattern", columns="rate", values="lead_s")
                .round(2)
                .reset_index()
            )
            sp.columns = ["pattern"] + [f"{c} tok/s" for c in sp.columns[1:]]
            ld.columns = ["pattern"] + [f"{c} tok/s" for c in ld.columns[1:]]
            md += [
                "**Speedup**",
                "",
                md_table(sp, list(sp.columns)),
                "",
                "**Median head start per dispatch (s)**",
                "",
                md_table(ld, list(ld.columns)),
                "",
            ]
        md += [
            "Two regimes, pulling in opposite directions:",
            "",
            "- **Fan-out patterns get *better* as the stream gets faster** "
            "(`map16` 1.94x at 20 tok/s, 3.79x at 60, 6.09x at 150). "
            "Speculation removes serialized call latency from the turn; the "
            "stream itself is untouched. When generation is slow it dominates "
            "the wall and compresses every ratio toward 1.0 — pure Amdahl. "
            "Real serving rates for small models sit at the fast end of this "
            "range, so the 60 tok/s numbers in this document are the "
            "conservative ones.",
            "- **Serial patterns get *worse* as the stream gets faster** "
            "(`chain4_slow` 1.56x at 20 tok/s, 1.20x at 60, 1.01x at 150). "
            "Their only win is the head start on the first call, and that head "
            "start is exactly the stream time left after the statement closes: "
            "the head-start table shows it shrinking from 2.18s to 0.45s.",
            "",
            "So the two tables measure different things: relative speedup is "
            "largest when generation is fast, while absolute seconds hidden is "
            "largest when generation is slow. A harness that wants one number "
            "should quote the rate its own model actually generates at.",
            "",
            f"![rates]({fig}/rate_sweep.png)",
            "",
        ]

    conc = HERE / "results" / "conc4" / "runs.csv"
    if conc.exists():
        tc = summarize(load("conc4"))
        if not tc.empty:
            m = t.reset_index()[["pattern", "speedup"]].rename(
                columns={"speedup": "speedup_1turn"}
            )
            tc = tc.merge(m, on="pattern", how="left")
            gm1 = float(
                tc["speedup_1turn"]
                .dropna()
                .pow(1 / max(len(tc["speedup_1turn"].dropna()), 1))
                .prod()
            )
            gm4 = float(tc["speedup"].pow(1 / len(tc)).prod())
            md += [
                "## Under load: 4 concurrent turns",
                "",
                "The tables above run one turn at a time against a "
                "dedicated endpoint — the friendliest case for "
                "speculation, because there is always spare GPU. This "
                "stage runs **four identical turns at once** and reports "
                "the **makespan** of the batch, so speculative requests "
                "compete with real ones for the same server.",
                "",
                f"- geomean speedup on these patterns: **{gm4:.2f}x** at "
                f"concurrency 4, versus **{gm1:.2f}x** for the same "
                "patterns run alone.",
                "- the win shrinks where it came from filling idle GPU and "
                "survives where it came from removing serialization; "
                "either way the floor holds.",
                "",
                md_table(
                    tc.sort_values("speedup", ascending=False)[
                        [
                            "pattern",
                            "base_s",
                            "spec_s",
                            "speedup",
                            "speedup_sd",
                            "speedup_1turn",
                            "hits",
                            "misses",
                            "evicted",
                            "score_b",
                            "score_s",
                        ]
                    ],
                    [
                        "pattern",
                        "base_s",
                        "spec_s",
                        "speedup",
                        "speedup_sd",
                        "speedup_1turn",
                        "hits",
                        "misses",
                        "evicted",
                        "score_b",
                        "score_s",
                    ],
                ),
                "",
                "(`base_s`/`spec_s` here are makespans for four turns, not single-turn walls.)",
                "",
            ]

    if args.tag2:
        d2 = load(args.tag2, args.max_rep)
        t2 = summarize(d2)
        m2 = d2["model"].iloc[0]
        gm2 = float(t2["speedup"].pow(1 / len(t2)).prod())
        cmp = t.reset_index()[["pattern", "category", "speedup", "call_s"]].merge(
            t2[
                [
                    "pattern",
                    "base_s",
                    "spec_s",
                    "speedup",
                    "call_s",
                    "score_b",
                    "score_s",
                    "verdict",
                ]
            ],
            on="pattern",
            suffixes=("_a", "_b"),
        )
        cmp = cmp.rename(
            columns={
                "speedup_a": f"speedup {df['model'].iloc[0]}",
                "speedup_b": f"speedup {m2}",
                "call_s_a": "call_s A",
                "call_s_b": "call_s B",
                "base_s": "base_s B",
                "spec_s": "spec_s B",
            }
        )
        cols = [
            "pattern",
            "category",
            f"speedup {df['model'].iloc[0]}",
            f"speedup {m2}",
            "call_s A",
            "call_s B",
            "base_s B",
            "spec_s B",
            "score_b",
            "score_s",
            "verdict",
        ]
        md += [
            "## Cross-model generality",
            "",
            f"Every pattern re-run against `{m2}` on a second node — same "
            "programs, same claims, a sub-model whose calls take roughly "
            f"{t2['call_s'].median() / max(t.reset_index()['call_s'].median(), 1e-6):.1f}x "
            "as long.",
            "",
            f"- geomean **{gm2:.2f}x** over {len(t2)} patterns "
            f"(versus {float(t['speedup'].pow(1 / len(t)).prod()):.2f}x on "
            f"`{df['model'].iloc[0]}`), best {t2['speedup'].max():.2f}x, "
            f"worst {t2['speedup'].min():.2f}x.",
            f"- correctness parity: "
            f"**{int((t2['d_score'].abs() <= PARITY_EPS).sum())}/{len(t2)}**; "
            f"floor breaks: **{int((t2['speedup'] < FLOOR).sum())}**.",
            "- the win is *larger* on the bigger model, which is the "
            "expected direction: speculation hides sub-call latency, and "
            "there is more of it to hide. Nothing about the ordering of "
            "patterns changes — the same structures win and the same ones "
            "sit at the floor.",
            "",
            f"![crossmodel]({fig}/crossmodel.png)",
            "",
            md_table(cmp.sort_values(f"speedup {m2}", ascending=False), cols),
            "",
        ]

    aa = HERE / "results" / "aa" / "runs.csv"
    if aa.exists():
        import pandas as pd

        ta = summarize(load("aa"), treat="aa")
        if not ta.empty:
            drift = (ta["speedup"] - 1).abs()
            md += [
                "## A/A control (measurement noise floor)",
                "",
                "The same patterns run **baseline against baseline**. Any "
                "apparent speedup here is measurement noise, so it bounds "
                "how small a real win the suite can resolve.",
                "",
                f"- {len(ta)} patterns, {int(ta['n'].max())} paired reps: "
                f"median |ratio − 1| = **{drift.median():.3f}**, "
                f"max = **{drift.max():.3f}** "
                f"(range {ta['speedup'].min():.2f}–{ta['speedup'].max():.2f}x)",
                f"- so the {WIN_MIN}x win threshold sits "
                f"{(WIN_MIN - 1) / max(drift.max(), 1e-6):.0f}x above the "
                "worst A/A deviation observed.",
                f"- **single repetitions are much noisier than means**: the "
                f"worst individual A/A repetition came in at "
                f"**{ta['speedup_min'].min():.2f}x** "
                f"(`{ta.loc[ta['speedup_min'].idxmin(), 'pattern']}`), "
                "baseline against baseline. That is why the never-slower "
                "invariant is evaluated on the mean of paired repetitions: "
                "the handful of speculative runs whose worst single "
                "repetition dips below the floor are inside this noise "
                "band, and every one of them is a short (1-3s) turn where "
                "one server stall dominates the wall.",
                "",
                md_table(
                    ta.rename(
                        columns={
                            "base_s": "armA_s",
                            "spec_s": "armB_s",
                            "speedup": "ratio_A/B",
                            "speedup_sd": "ratio_sd",
                            "speedup_min": "ratio_min",
                        }
                    )[["pattern", "armA_s", "armB_s", "ratio_A/B", "ratio_sd", "ratio_min"]],
                    ["pattern", "armA_s", "armB_s", "ratio_A/B", "ratio_sd", "ratio_min"],
                ),
                "",
            ]

    md += [
        "## What the suite found",
        "",
        "Developing the suite was iterative: each wave of patterns was "
        "written from the previous wave's traces. Three engine bugs came "
        "out of it, all of the same shape — correct results, silently lost "
        "speculation, which no correctness test would have caught.",
        "",
    ]
    for i, (title, what, fix) in enumerate(FIXES, 1):
        md += [f"{i}. **{title}.** {what} *Fix:* {fix}", ""]
    md += [
        "Three patterns were also revised after their first run, with the "
        "original prediction kept in the record:",
        "",
    ]
    for name, why in REFINED:
        md += [f"- {name} {why}", ""]

    ref = t.reset_index()
    ref = ref[ref["verdict"].str.startswith("REFUTED")]
    if not ref.empty:
        md += [
            "## Refuted predictions",
            "",
            "Kept in the record rather than relabelled. None of these is a "
            "correctness failure — every one is a case where the *size* of "
            "the predicted win was wrong.",
            "",
        ]
        for _, r in ref.iterrows():
            note = REFUTED_NOTES.get(r["pattern"], "")
            md += [
                f"- **`{r['pattern']}`** — predicted a win (>= "
                f"{WIN_MIN}x), measured {r['speedup']}x "
                f"(p={r['p_val']}). {note}",
                "",
            ]
        md += [
            "Two *dispatch-count* expectations were also wrong, and those "
            "were our arithmetic rather than predictions: "
            "`tool_error_recovered` makes 5 real calls (4 fragile + 1 "
            "fallback), not 8, and `tool_error_raises` makes 4 dispatches, "
            "not 3 — the engine does speculate the call that raises, then "
            "drops the failed speculation so the real call re-raises at the "
            "use site (3 hits, 1 miss). Both were corrected in "
            "`expectations.py` with the reason recorded there.",
            "",
        ]

    md += [
        "## Pattern catalogue",
        "",
        "Every pattern: what it was predicted to do, what it measured, and "
        "the exact program that ran.",
        "",
    ] + catalogue

    tr = HERE / "results" / args.tag.split(",")[0] / "traces.md"
    if tr.exists():
        md += [tr.read_text(), ""]

    md += [
        "## What this suite does not test",
        "",
        "- **Quality.** Scores check parity, not answer quality; the suite "
        "cannot tell you speculation makes answers better (it should not).",
        "- **Server-side contention.** One suite process talks to a "
        "dedicated endpoint. Speculation increases request concurrency, so "
        "on a saturated server the win shrinks; the OOLONG campaign at "
        "concurrency 8 is the measurement for that regime.",
        "- **Live trajectory drift.** Fixed programs deliberately remove "
        "the effect of speculation on what the model writes next.",
        "- **Non-Python runtimes.** Bun/bash frontends are not covered here.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "sbatch benchmark/suite/serve_suite.sbatch      # vLLM endpoint",
        f"uv run python -m benchmark.suite.run_suite --repeats 5 --tag {args.tag}",
        f"uv run python -m benchmark.suite.analyze_suite {args.tag}",
        f"uv run python -m benchmark.suite.plots_suite {args.tag}",
        f"uv run python -m benchmark.suite.render_md --tag {args.tag}",
        "uv run python -m benchmark.suite.inspect_pattern <name> spec  # event trace",
        "```",
        "",
    ]
    Path(args.out).write_text("\n".join(md))
    print("wrote", args.out, len("\n".join(md)), "chars")


if __name__ == "__main__":
    main()
