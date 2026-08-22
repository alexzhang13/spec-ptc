"""Scenario catalog: 30 scripted demos across 6 categories."""

from __future__ import annotations

from dataclasses import dataclass, field

from demo.oolong import make_log_context, make_report_context
from spec_ptc.runtime.engines import MockTiming


@dataclass
class Scenario:
    name: str
    category: str
    description: str
    context: str
    turns: list[str]
    timing: MockTiming = field(
        default_factory=lambda: MockTiming(
            main_tok_per_s=140, sub_base_s=0.6, sub_jitter_s=0.5, sub_tokens=10
        )
    )
    tags: list[str] = field(default_factory=list)


def _blk(code: str, pre: str = "Working on it.", post: str = "Executing.") -> str:
    return f"{pre}\n```repl\n{code}\n```\n{post}"


def _map_code(n_chunks: int, per: str = "summarize the following entries: ") -> str:
    return (
        f"size = max(1, len(context) // {n_chunks})\n"
        f"chunks = [context[i:i+size] for i in range(0, len(context), size)][: {n_chunks}]\n"
        "results = []\n"
        "for c in chunks:\n"
        f"    results.append(llm_query({per!r} + c))\n"
        "\n"
        "combined = '\\n'.join(str(r) for r in results)\n"
        "answer['content'] = llm_query('Combine these into one answer:\\n' + combined)\n"
        "answer['ready'] = True"
    )


CATALOG: list[Scenario] = []


def _add(name, category, description, context, turns, tags=(), **tkw):
    CATALOG.append(
        Scenario(
            name=name,
            category=category,
            description=description,
            context=context,
            turns=turns,
            tags=list(tags),
            timing=MockTiming(**tkw)
            if tkw
            else Scenario.__dataclass_fields__["timing"].default_factory(),
        )
    )


LOG = make_log_context()
LOG_BIG = make_log_context(n_entries=480, seed=13)
REPORT = make_report_context()

# ------------------------------------------------------------------ oolong (6)
_add(
    "oolong-mood-agg",
    "oolong",
    "Aggregate dominant mood per chunk over a 240-entry log, then reduce.",
    LOG,
    [
        _blk(
            _map_code(12, "What is the dominant mood in these log entries? "),
            pre="I'll chunk the log and aggregate moods per chunk.",
        )
    ],
    tags=["flagship"],
)

_add(
    "oolong-user-profile",
    "oolong",
    "Per-user filtered sub-queries (8 users) + cross-user reduce.",
    LOG,
    [
        _blk(
            "users = ['alice','bob','carol','dave','erin','frank','grace','heidi']\n"
            "profiles = {}\n"
            "for u in users:\n"
            "    entries = '\\n'.join(l for l in context.split('\\n') if 'user=' + u in l)\n"
            "    profiles[u] = llm_query('Profile this user from their entries: ' + entries[:800])\n"
            "\n"
            "summary = '\\n'.join(u + ': ' + str(p) for u, p in profiles.items())\n"
            "answer['content'] = llm_query('Which user is most negative overall?\\n' + summary)\n"
            "answer['ready'] = True",
            pre="Group entries by user, profile each, then compare.",
        )
    ],
)

_add(
    "oolong-needle-count",
    "oolong",
    "Count occurrences of a topic per chunk; numeric reduce in code.",
    LOG,
    [
        _blk(
            "size = len(context) // 10\n"
            "chunks = [context[i:i+size] for i in range(0, len(context), size)][:10]\n"
            "counts = []\n"
            "for c in chunks:\n"
            "    counts.append(llm_query('Reply with ONLY a number: how many entries "
            "mention the gpu cluster? ' + c))\n"
            "\n"
            "nums = [len(str(c)) % 7 for c in counts]  # mock-safe numeric fold\n"
            "answer['content'] = 'total-ish: ' + str(sum(nums))\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "oolong-batched",
    "oolong",
    "Same aggregation via llm_query_batched (model already parallelized).",
    LOG,
    [
        _blk(
            "size = len(context) // 12\n"
            "chunks = [context[i:i+size] for i in range(0, len(context), size)][:12]\n"
            "prompts = ['Dominant mood in: ' + c for c in chunks]\n"
            "results = llm_query_batched(prompts)\n"
            "answer['content'] = llm_query('Combine: ' + '|'.join(str(r) for r in results))\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "oolong-multiturn",
    "oolong",
    "Iterative REPL over 3 turns: peek, map, reduce — the RLM loop shape.",
    LOG_BIG,
    [
        _blk(
            "print('context chars:', len(context))\nprint(context[:200])",
            pre="First, let me look at the context shape.",
        ),
        _blk(
            "size = len(context) // 16\n"
            "chunks = [context[i:i+size] for i in range(0, len(context), size)][:16]\n"
            "notes = []\n"
            "for c in chunks:\n"
            "    notes.append(llm_query('Summarize moods and topics: ' + c))\n"
            "\n"
            "print('collected', len(notes))",
            pre="Now map over 16 chunks.",
        ),
        _blk(
            "answer['content'] = llm_query('Final answer from notes:\\n' + "
            "'\\n'.join(str(n) for n in notes))\n"
            "answer['ready'] = True",
            pre="Reduce to the final answer.",
        ),
    ],
    tags=["flagship", "multiturn"],
)

_add(
    "oolong-two-stage",
    "oolong",
    "Two-stage map: extract then judge, chained per chunk (pipelined).",
    REPORT,
    [
        _blk(
            "secs = context.split('## ')[1:9]\n"
            "findings = []\n"
            "for s in secs:\n"
            "    f = llm_query('Extract the key finding: ' + s[:600])\n"
            "    findings.append(llm_query('Is this finding positive or negative? ' + str(f)))\n"
            "\n"
            "answer['content'] = llm_query('Overall verdict:\\n' + '\\n'.join(str(x) for x in findings))\n"
            "answer['ready'] = True"
        )
    ],
)

# ------------------------------------------------------------------ map (7)
for n in (4, 8, 16, 32):
    _add(
        f"map-{n}",
        "map",
        f"Independent map over {n} chunks + reduce.",
        LOG if n <= 16 else LOG_BIG,
        [_blk(_map_code(n))],
        tags=["scaling"],
        main_tok_per_s=140,
        sub_base_s=0.6,
        sub_jitter_s=0.5,
        sub_tokens=10,
    )

_add(
    "map-listcomp",
    "map",
    "Map via list comprehension (single statement).",
    LOG,
    [
        _blk(
            "size = len(context) // 8\n"
            "results = [llm_query('gist: ' + context[i:i+size]) for i in range(0, len(context), size)][:8]\n"
            "answer['content'] = llm_query('merge: ' + '|'.join(str(r) for r in results))\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "map-enumerate-filter",
    "map",
    "Loop with an index condition (every other chunk).",
    LOG,
    [
        _blk(
            "size = len(context) // 12\n"
            "chunks = [context[i:i+size] for i in range(0, len(context), size)][:12]\n"
            "picked = []\n"
            "for i, c in enumerate(chunks):\n"
            "    if i % 2 == 0:\n"
            "        picked.append(llm_query('summarize: ' + c))\n"
            "\n"
            "answer['content'] = '|'.join(str(p) for p in picked)\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "map-fn-helper",
    "map",
    "Hooked call inside a model-defined helper function.",
    LOG,
    [
        _blk(
            "def gist(text):\n"
            "    return llm_query('one-line gist: ' + text)\n"
            "\n"
            "size = len(context) // 6\n"
            "parts = [gist(context[i:i+size]) for i in range(0, len(context), size)][:6]\n"
            "answer['content'] = '|'.join(str(p) for p in parts)\n"
            "answer['ready'] = True"
        )
    ],
)

# ------------------------------------------------------------------ chain (4)
_add(
    "chain-2",
    "chain",
    "Two dependent calls (draft -> polish).",
    REPORT,
    [
        _blk(
            "draft = llm_query('Draft a 2-line summary of: ' + context[:800])\n"
            "answer['content'] = llm_query('Polish this: ' + str(draft))\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "chain-4",
    "chain",
    "Four-deep dependent chain — the honest floor.",
    REPORT,
    [
        _blk(
            "a = llm_query('step1: ' + context[:400])\n"
            "b = llm_query('step2 refine: ' + str(a))\n"
            "c = llm_query('step3 refine: ' + str(b))\n"
            "answer['content'] = llm_query('step4 finalize: ' + str(c))\n"
            "answer['ready'] = True"
        )
    ],
    tags=["floor"],
)

_add(
    "tree-reduce",
    "chain",
    "Pairwise tree reduction over 8 leaves.",
    LOG,
    [
        _blk(
            "size = len(context) // 8\n"
            "layer = [llm_query('leaf: ' + context[i:i+size]) for i in range(0, len(context), size)][:8]\n"
            "while len(layer) > 1:\n"
            "    layer = [llm_query('merge: ' + str(layer[i]) + ' || ' + str(layer[i+1]))\n"
            "             for i in range(0, len(layer) - 1, 2)] + ([layer[-1]] if len(layer) % 2 else [])\n"
            "\n"
            "answer['content'] = str(layer[0])\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "refine-loop",
    "chain",
    "Bounded iterative refinement (3 rounds).",
    REPORT,
    [
        _blk(
            "text = context[:500]\n"
            "for _ in range(3):\n"
            "    text = str(llm_query('improve: ' + str(text)[:400]))\n"
            "\n"
            "answer['content'] = text\n"
            "answer['ready'] = True"
        )
    ],
)

# ------------------------------------------------------------------ vote (2)
_add(
    "majority-5",
    "vote",
    "5 identical prompts — independent samples (FIFO).",
    LOG,
    [
        _blk(
            "votes = [llm_query('Is the overall mood positive? Answer yes/no. ' + context[:300]) for _ in range(5)]\n"
            "tally = sum(1 for v in votes if 'yes' in str(v).lower())\n"
            "answer['content'] = 'yes' if tally >= 3 else 'no (' + str(tally) + '/5)'\n"
            "answer['ready'] = True"
        )
    ],
    tags=["multiplicity"],
)

_add(
    "self-consistency-8",
    "vote",
    "8 samples of the same reasoning prompt.",
    REPORT,
    [
        _blk(
            "samples = [llm_query('Estimate risk level (low/med/high): ' + context[:400]) for _ in range(8)]\n"
            "answer['content'] = max(set(str(s)[-6:] for s in samples), key=lambda x: sum(1 for s in samples if str(s).endswith(x)))\n"
            "answer['ready'] = True"
        )
    ],
)

# ------------------------------------------------------------------ branchy (4)
_add(
    "branch-known",
    "branchy",
    "Condition on known state — shadow takes the branch.",
    LOG,
    [
        _blk(
            "task = 'moods'\n"
            "if task == 'moods':\n"
            "    r = llm_query('mood analysis: ' + context[:400])\n"
            "else:\n"
            "    r = llm_query('topic analysis: ' + context[:400])\n"
            "\n"
            "answer['content'] = str(r)\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "branch-on-result",
    "branchy",
    "Condition on a speculated value — shadow waits, then proceeds.",
    LOG,
    [
        _blk(
            "cls = llm_query('Answer with one word, happy or sad: ' + context[:300])\n"
            "if 'happy' in str(cls).lower():\n"
            "    detail = llm_query('Explain the happiness: ' + context[:300])\n"
            "else:\n"
            "    detail = llm_query('Explain the sadness: ' + context[:300])\n"
            "\n"
            "answer['content'] = str(detail)\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "try-except",
    "branchy",
    "Call inside try/except.",
    LOG,
    [
        _blk(
            "try:\n"
            "    r = llm_query('summarize: ' + context[:300])\n"
            "    x = int('not a number')\n"
            "except ValueError:\n"
            "    r2 = llm_query('fallback path: ' + str(r)[:100])\n"
            "\n"
            "answer['content'] = str(r2)\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "classify-rows",
    "branchy",
    "Per-row conditional calls over parsed records.",
    LOG,
    [
        _blk(
            "rows = context.split('\\n')[:20]\n"
            "flagged = []\n"
            "for row in rows:\n"
            "    if 'angry' in row or 'anxious' in row:\n"
            "        flagged.append(llm_query('Why might this entry be negative? ' + row))\n"
            "\n"
            "answer['content'] = str(len(flagged)) + ' flagged'\n"
            "answer['ready'] = True"
        )
    ],
)

# ------------------------------------------------------------------ adversarial (5)
_add(
    "no-calls",
    "adversarial",
    "Zero hooked calls, heavy pure compute — overhead floor.",
    LOG,
    [
        _blk(
            "acc = 0\n"
            "for i in range(200000):\n"
            "    acc = (acc + i * i) % 1000003\n"
            "\n"
            "answer['content'] = str(acc)\n"
            "answer['ready'] = True"
        )
    ],
    tags=["guard"],
)

_add(
    "rebind-hook",
    "adversarial",
    "Model rebinds llm_query mid-block.",
    LOG,
    [
        _blk(
            "a = llm_query('before rebind')\n"
            "llm_query = str\n"
            "b = llm_query('after rebind')\n"
            "answer['content'] = str(a) + '|' + str(b)\n"
            "answer['ready'] = True"
        )
    ],
    tags=["guard"],
)

_add(
    "runaway-loop",
    "adversarial",
    "Hot loop in a closed statement — shadow line budget.",
    LOG,
    [
        _blk(
            "n = 0\n"
            "while n < 3 * 10**6:\n"
            "    n += 1\n"
            "\n"
            "r = llm_query('after the hot loop: ' + str(n))\n"
            "answer['content'] = str(r)\n"
            "answer['ready'] = True"
        )
    ],
    tags=["guard"],
)

_add(
    "shadow-crash",
    "adversarial",
    "Statement crashes after dispatching — real run fails identically.",
    LOG,
    [
        _blk(
            "xs = [llm_query('p1'), llm_query('p2')]\n"
            "boom = xs[99]\n"
            "answer['content'] = 'unreachable'\n"
            "answer['ready'] = True"
        )
    ],
    tags=["guard"],
)

_add(
    "giant-literal",
    "adversarial",
    "Multi-KB string literal — parser stress.",
    LOG,
    [
        _blk(
            "blob = '" + ("lorem ipsum " * 400) + "'\n"
            "r = llm_query('length check: ' + str(len(blob)))\n"
            "answer['content'] = str(r)\n"
            "answer['ready'] = True"
        )
    ],
    tags=["guard"],
)

# ------------------------------------------------------------------ mixed (2)
_add(
    "doc-pipeline",
    "mixed",
    "Extract -> per-section summary -> verdict -> final.",
    REPORT,
    [
        _blk(
            "secs = context.split('## ')[1:7]\n"
            "sums = [llm_query('Summarize section: ' + s[:500]) for s in secs]\n"
            "verdict = llm_query('Any regressions? ' + '|'.join(str(s) for s in sums))\n"
            "if 'regress' in str(verdict).lower() or True:\n"
            "    final = llm_query('Write the final report: ' + str(verdict))\n"
            "\n"
            "answer['content'] = str(final)\n"
            "answer['ready'] = True"
        )
    ],
)

_add(
    "explore-then-map",
    "mixed",
    "Turn 1 explores; turn 2 maps + answers (multi-turn).",
    LOG,
    [
        _blk(
            "lines = context.split('\\n')\nprint(len(lines), 'entries')\nprint(lines[0])",
            pre="Let me explore the data first.",
        ),
        _blk(
            "size = len(context) // 8\n"
            "parts = [llm_query('key facts: ' + context[i:i+size]) for i in range(0, len(context), size)][:8]\n"
            "answer['content'] = llm_query('Answer the question from: ' + '|'.join(str(p) for p in parts))\n"
            "answer['ready'] = True",
            pre="Now I'll map over the entries.",
        ),
    ],
    tags=["multiturn"],
)


# ------------------------------------------------------------------ geometry (4)
_add(
    "late-calls",
    "geometry",
    "Calls at the END of a long block — worst-case lead time for streaming.",
    LOG,
    [
        _blk(
            "\n".join(f"w{i} = ({i} * {i} + 13) % 89" for i in range(60))
            + "\nsize = len(context) // 8\n"
            "rs = [llm_query('late: ' + context[i:i+size]) for i in range(0, len(context), size)][:8]\n"
            "answer['content'] = '|'.join(str(r) for r in rs)\n"
            "answer['ready'] = True",
            pre="A lot of setup code first, calls at the very end.",
        )
    ],
)

_add(
    "prose-tail",
    "geometry",
    "Model explains at length AFTER the code block — free overlap window.",
    LOG,
    [
        _blk(
            "size = len(context) // 8\n"
            "rs = [llm_query('pt: ' + context[i:i+size]) for i in range(0, len(context), size)][:8]\n"
            "answer['content'] = '|'.join(str(r) for r in rs)\n"
            "answer['ready'] = True",
            post="Now, let me explain the approach in detail. "
            + "The mapping strategy chunks the context evenly and asks the sub-model for each part, which balances recall and cost. "
            * 12,
        )
    ],
)

_add(
    "two-block-turn",
    "geometry",
    "Two dependent ```repl blocks in ONE response (compute then map).",
    LOG,
    [
        "First I will prepare the chunks.\n```repl\n"
        "size = len(context) // 6\n"
        "chunks = [context[i:i+size] for i in range(0, len(context), size)][:6]\n"
        "print(len(chunks), 'chunks ready')\n"
        "```\nNow map over them in a second block.\n```repl\n"
        "rs = [llm_query('tb: ' + c) for c in chunks]\n"
        "answer['content'] = '|'.join(str(r) for r in rs)\n"
        "answer['ready'] = True\n"
        "```\nDone."
    ],
)

_add(
    "slow-stream",
    "geometry",
    "Frontier-API-speed stream (30 tok/s): maximal generation-overlap regime.",
    LOG,
    [_blk(_map_code(8), pre="Mapping at a slow, realistic API token rate.")],
    main_tok_per_s=30,
    sub_base_s=0.6,
    sub_jitter_s=0.5,
    sub_tokens=10,
)


def get_scenario(name: str) -> Scenario:
    for s in CATALOG:
        if s.name == name:
            return s
    raise KeyError(name)
