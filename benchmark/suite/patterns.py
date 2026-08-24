"""Pre-defined REPL patterns for the speculation suite. Each pattern is FIXED
code (the 'generation' is replayed at a realistic token rate), so run-to-run
variance comes only from real sub-call latencies — this isolates execution
strategy. `hypothesis` states what speculation should do and why; `check`
scores the outcome 0..1 where verifiable."""

from collections.abc import Callable
from dataclasses import dataclass

# A small, meaningful corpus the 4B sub-model can actually work with.
CONTEXT = "\n".join(
    f"[item {i:02d}] The {adj} {noun} report cites a {n}% change in {topic}."
    for i, (adj, noun, n, topic) in enumerate(
        [
            (a, b, (7 * i + 13) % 40 + 1, t)
            for i, (a, b, t) in enumerate(
                [
                    (a, b, t)
                    for a in ("quarterly", "annual", "interim", "audited")
                    for b in ("revenue", "safety", "latency", "quality")
                    for t in ("cloud spend", "energy use")
                ][:32]
            )
        ]
    )
)


@dataclass
class Pattern:
    name: str
    category: str  # easy | hard | edge
    hypothesis: str  # what speculation should do here, and why
    code: str  # the fixed ```repl program (real llm_query calls)
    prose_tail: str = ""  # text 'generated' after the block (overlap window)
    check: Callable[[dict], float] | None = None  # ns -> score in [0,1]
    n_expected_calls: int | None = None
    sub_max_tokens: int = 96
    prose_head: str = ""  # text 'generated' before the block
    raw: str = ""  # full scripted response (multi-block patterns)
    turns: tuple = ()  # multi-TURN patterns: one scripted response each


def _ans_lines(ns, var, n):
    v = ns.get(var)
    if not isinstance(v, list) or len(v) != n:
        return 0.0
    return sum(1 for x in v if isinstance(x, str) and len(x) > 0) / n


PATTERNS: list[Pattern] = []


def _p(**kw):
    PATTERNS.append(Pattern(**kw))


# ============================ EASY: expected wins ============================
_p(
    name="map16",
    category="easy",
    n_expected_calls=16,
    hypothesis=(
        "16 independent calls in a plain for-loop. The peek unrolls the "
        "loop before it closes; expected: near-max fan-out, wall ≈ "
        "stream + max(call) instead of Σ(calls)."
    ),
    code=(
        "lines = context.split('\\n')\n"
        "outs = []\n"
        "for i in range(16):\n"
        "    outs.append(llm_query('In six words, summarize: ' + lines[i]))\n"
        "\n"
        "answer['content'] = ' | '.join(str(o) for o in outs)\n"
        "answer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 16),
)

_p(
    name="map_reduce",
    category="easy",
    n_expected_calls=9,
    hypothesis=(
        "8-way map + dependent reduce. Map fans out during streaming; "
        "the reduce forces the map (correct sync) and overlaps the "
        "post-block prose. Expected: strong win, reduce pipelined."
    ),
    code=(
        "lines = context.split('\\n')[:8]\n"
        "notes = [llm_query('Extract the percentage from: ' + l) for l in lines]\n"
        "answer['content'] = llm_query('Which is largest? ' + ' ; '.join(str(n) for n in notes))\n"
        "answer['ready'] = True"
    ),
    prose_tail="Now I will explain the aggregation approach in some detail. " * 8,
    check=lambda ns: 1.0 if ns.get("answer", {}).get("content") else 0.0,
)

_p(
    name="batched8",
    category="easy",
    n_expected_calls=8,
    hypothesis=(
        "llm_query_batched: the model already parallelized, so baseline "
        "is fast too. Expected: small win only (head start from the "
        "pre-close peek is excluded for batched tools; win = pre-pass timing)."
    ),
    code=(
        "lines = context.split('\\n')[:8]\n"
        "prompts = ['One-word topic of: ' + l for l in lines]\n"
        "outs = llm_query_batched(prompts)\n"
        "answer['content'] = ','.join(str(o)[:20] for o in outs)\n"
        "answer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 8),
)

_p(
    name="two_stage",
    category="easy",
    n_expected_calls=12,
    hypothesis=(
        "Per-item chain (extract → judge) over 6 items. Stage-1 unrolls "
        "at peek time; stage-2 depends on stage-1 results and pipelines. "
        "Expected: Σ(2·call) → ≈ max(stage1)+max(stage2)."
    ),
    code=(
        "lines = context.split('\\n')[:6]\n"
        "verdicts = []\n"
        "for l in lines:\n"
        "    fact = llm_query('State the single number in: ' + l)\n"
        "    verdicts.append(llm_query('Is this above 20? Answer yes/no: ' + str(fact)))\n"
        "\n"
        "answer['content'] = ','.join(str(v)[:12] for v in verdicts)\n"
        "answer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "verdicts", 6),
)

_p(
    name="late_loop",
    category="easy",
    n_expected_calls=8,
    hypothesis=(
        "40 lines of pure compute BEFORE an 8-way map: dispatch can only "
        "start late in the stream. Expected: win shrinks toward the "
        "lazy/fan-out component; measures how head start decays."
    ),
    code=(
        "\n".join(f"v{i} = ({i} * 31 + 7) % 97" for i in range(40)) + "\n"
        "lines = context.split('\\n')[:8]\n"
        "outs = [llm_query('Rewrite formally: ' + l) for l in lines]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 8),
)

# ============================ HARD: designed to hurt ============================
_p(
    name="serial_chain6",
    category="hard",
    n_expected_calls=6,
    hypothesis=(
        "6 calls, each consuming the previous output — zero parallel "
        "width. v1 REFINED this hypothesis: the shadow starts the whole "
        "chain during streaming, so the head start applies once to the "
        "chain (measured 1.5× at short calls); it amortizes toward "
        "1.0× as calls lengthen (see chain4_slow). Must never be slower."
    ),
    code=(
        "s = context.split('\\n')[0]\n"
        + "\n".join("s = llm_query('Shorten by one word: ' + str(s))" for _ in range(6))
        + "\n"
        "answer['content'] = str(s)\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if ns.get("s") else 0.0,
)

_p(
    name="branch_on_result",
    category="hard",
    n_expected_calls=3,
    hypothesis=(
        "Control flow forced by a speculated value: the shadow must "
        "block on call 1 before knowing which branch's call to issue. "
        "Expected: only call-1's head start is winnable; no wrong-branch "
        "waste (peek skips tail conditionals)."
    ),
    code=(
        "cls = llm_query('Answer exactly yes or no: does the first line mention revenue? ' + context.split('\\n')[0])\n"
        "if 'yes' in str(cls).lower():\n"
        "    detail = llm_query('Quote the revenue figure: ' + context.split('\\n')[0])\n"
        "else:\n"
        "    detail = llm_query('Summarize: ' + context.split('\\n')[0])\n"
        "\n"
        "followup = llm_query('One-word reaction to: ' + str(detail))\n"
        "answer['content'] = str(followup)\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if ns.get("followup") else 0.0,
)

_p(
    name="mutating_loop",
    category="hard",
    n_expected_calls=6,
    hypothesis=(
        "The loop mutates its own iterable through a subscript — the "
        "peek's mutation rail + bet retraction case. Expected: iteration-0 "
        "peek only, retraction of any stale bets mid-stream, remaining "
        "calls dispatched by the shadow at loop close; waste ≈ 0-5 "
        "retracted calls, wall ≥ baseline-parity."
    ),
    code=(
        "data = context.split('\\n')[:6]\n"
        "outs = []\n"
        "for i in range(6):\n"
        "    outs.append(llm_query('Topic word of: ' + data[i]))\n"
        "    data[(i + 1) % 6] = data[(i + 1) % 6] + ' (seen)'\n"
        "\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 6),
)

_p(
    name="early_break",
    category="hard",
    n_expected_calls=3,
    hypothesis=(
        "Unroll overshoot: loop over 10 items breaks at i==2 on a "
        "call-free condition. Peek dispatches all 10; 7 are retracted/"
        "evicted. Measures the waste bound and that wall stays ≥ parity."
    ),
    code=(
        "lines = context.split('\\n')[:10]\n"
        "outs = []\n"
        "for i in range(10):\n"
        "    outs.append(llm_query('Five-word gist: ' + lines[i]))\n"
        "    if i == 2:\n"
        "        break\n"
        "\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 3),
)

_p(
    name="jail_break_mid",
    category="hard",
    n_expected_calls=6,
    hypothesis=(
        "A non-whitelisted import (socket) lands after 3 calls: the "
        "shadow opaque-aborts mid-block, calls 4-6 become misses. "
        "Measures graceful degradation: partial speculation must still "
        "beat zero and never corrupt."
    ),
    code=(
        "lines = context.split('\\n')[:6]\n"
        "a = [llm_query('Noun in: ' + lines[0]), llm_query('Noun in: ' + lines[1]), llm_query('Noun in: ' + lines[2])]\n"
        "import socket\n"
        "b = [llm_query('Verb in: ' + lines[3]), llm_query('Verb in: ' + lines[4]), llm_query('Verb in: ' + lines[5])]\n"
        "answer['content'] = str(len(a) + len(b)) + '|' + socket.__name__\n"
        "answer['ready'] = True"
    ),
    check=lambda ns: (
        1.0 if str(ns.get("answer", {}).get("content", "")).startswith("6|socket") else 0.0
    ),
)

_p(
    name="identical10",
    category="hard",
    n_expected_calls=10,
    hypothesis=(
        "10 IDENTICAL prompts (self-consistency vote). Multiplicity "
        "rail: each must be an independent sample (FIFO), never one "
        "shared result. Independence is verified MECHANICALLY (10 "
        "dispatches, 10 distinct claims in the event log); output "
        "diversity is a property of the model, not the method."
    ),
    code=(
        "votes = [llm_query('Pick a random-feeling word for change, one word only.') for _ in range(10)]\n"
        "answer['content'] = '|'.join(str(v).strip()[:16] for v in votes)\n"
        "answer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "votes", 10),
)

_p(
    name="crash_after_dispatch",
    category="hard",
    n_expected_calls=2,
    hypothesis=(
        "IndexError right after 2 dispatches: real run fails identically "
        "to baseline; in-flight speculations must be evicted and the "
        "engine must not be left loaded. Measures failure-path parity."
    ),
    code=(
        "lines = context.split('\\n')[:2]\n"
        "outs = [llm_query('Echo briefly: ' + lines[0]), llm_query('Echo briefly: ' + lines[1])]\n"
        "boom = outs[99]\n"
        "answer['content'] = 'unreachable'\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if not ns.get("answer", {}).get("ready") else 0.0,
)

# ============================ EDGE: curriculum ============================
_p(
    name="syntax_gauntlet",
    category="edge",
    n_expected_calls=4,
    hypothesis=(
        "Calls hidden in a decorator'd helper, a dict comprehension, a "
        "try/except and an f-string arg — segmenter/peek must neither "
        "misparse nor miss. Expected: all 4 speculated, full parity."
    ),
    code=(
        "def twice(fn):\n"
        "    return fn\n"
        "\n"
        "@twice\n"
        "def ask(q):\n"
        "    return llm_query(f'Answer tersely: {q}')\n"
        "\n"
        "first = ask('What color is the sky at noon?')\n"
        "pair = {k: llm_query('Opposite of ' + k) for k in ['hot', 'wet']}\n"
        "try:\n"
        "    x = int('nope')\n"
        "except ValueError:\n"
        "    fallback = llm_query('Say OK.')\n"
        "\n"
        "answer['content'] = str(first) + str(pair) + str(fallback)\n"
        "answer['ready'] = True"
    ),
    check=lambda ns: 1.0 if ns.get("fallback") and len(ns.get("pair", {})) == 2 else 0.0,
)

_p(
    name="while_computed",
    category="edge",
    n_expected_calls=5,
    hypothesis=(
        "While-loop with a computed bound (not peek-unrollable): the "
        "shadow handles it by simply running the loop. Expected: full "
        "hits via shadow-exec dispatch, no peek contribution."
    ),
    code=(
        "i = 0\n"
        "n = len(context.split('\\n')[:5])\n"
        "outs = []\n"
        "while i < n:\n"
        "    outs.append(llm_query('Length in words: ' + context.split('\\n')[i]))\n"
        "    i += 1\n"
        "\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 5),
)

_p(
    name="multiblock",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "Two dependent blocks in one response: block 2 maps over "
        "variables computed in block 1. Shadow state must carry across "
        "block boundaries within the turn. Expected: normal win."
    ),
    code="",  # scripted below as a full multi-block response
    check=lambda ns: _ans_lines(ns, "outs", 6),
)
MULTIBLOCK_RESPONSE = (
    "First, prepare the slices.\n```repl\n"
    "lines = context.split('\\n')\n"
    "picks = [lines[i] for i in range(0, 12, 2)]\n"
    "print(len(picks), 'picked')\n"
    "```\nNow query each.\n```repl\n"
    "outs = [llm_query('Key noun: ' + p) for p in picks]\n"
    "answer['content'] = ','.join(str(o)[:14] for o in outs)\n"
    "answer['ready'] = True\n```\nDone."
)

PATTERNS[-1].raw = MULTIBLOCK_RESPONSE

_p(
    name="prose_sandwich",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "Long prose BEFORE the block delays parsing; long prose AFTER "
        "extends the overlap window. Net expected: win preserved, "
        "showing prose position only shifts when dispatch happens."
    ),
    code=(
        "lines = context.split('\\n')[:6]\n"
        "outs = [llm_query('Formal tone: ' + l) for l in lines]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    prose_tail="Let me now walk through why this mapping is appropriate. " * 12,
    check=lambda ns: _ans_lines(ns, "outs", 6),
)
PROSE_HEAD = "I will start by considering the structure of the data carefully. " * 10
PATTERNS[-1].prose_head = PROSE_HEAD


def response_text(p: Pattern) -> str:
    if p.raw:
        return p.raw
    head = p.prose_head or "Working on it.\n"
    return f"{head}\n```repl\n{p.code}\n```\n{p.prose_tail or 'Done.'}"
