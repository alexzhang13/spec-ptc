"""V4: adversarial cases and Python-surface coverage. Written after reading the
v1/v2 results — these target the places where a speculator can be WRONG (tool
errors, stale guesses, unforkable state) or simply blind (comprehension forms,
callbacks, methods, recursion). Correctness parity is the primary metric here;
speed is secondary."""

from benchmark.suite.patterns import PATTERNS, Pattern, _ans_lines


def _p(**kw):
    PATTERNS.append(Pattern(**kw))


# --------------------------- Python-surface coverage ---------------------------

_p(
    name="dict_comp",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "Calls inside a DICT comprehension. The peek's loop unroll is "
        "written for list comprehensions and for-loops; a dict comp "
        "must still speculate via shadow execution at statement close. "
        "Expected: normal fan-out win, all 6 keys present."
    ),
    code=(
        "lines = context.split('\\n')[:6]\n"
        "outs = {i: llm_query('Three-word gist: ' + l) for i, l in enumerate(lines)}\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: (
        sum(1 for v in (ns.get("outs") or {}).values() if isinstance(v, str) and v) / 6
        if isinstance(ns.get("outs"), dict)
        else 0.0
    ),
)

_p(
    name="nested_comp",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "Nested comprehension (2 outer x 3 inner). Tests that the "
        "unroller does not double-dispatch or mis-key nested loop "
        "variables. Expected: exactly 6 distinct claims, full win."
    ),
    code=(
        "lines = context.split('\\n')[:3]\n"
        "outs = [[llm_query(t + l) for l in lines] for t in ['Noun in: ', 'Verb in: ']]\n"
        "flat = [x for row in outs for x in row]\n"
        "answer['content'] = str(len(flat))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "flat", 6),
)

_p(
    name="sorted_key",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "Calls made from inside a C-level callback (sorted key=). The "
        "call sites are invisible to any AST scan of the block and the "
        "results are forced immediately by the comparison. Expected: "
        "correctness preserved; win only from the pre-close peek."
    ),
    code=(
        "lines = context.split('\\n')[:6]\n"
        "ranked = sorted(lines, key=lambda l: len(str(llm_query('Rate 1-9, digit only: ' + l))))\n"
        "answer['content'] = str(len(ranked))\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if len(ns.get("ranked") or []) == 6 else 0.0,
)

_p(
    name="class_method",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "Calls inside a method of a class defined in the same block. "
        "The shadow must deepcopy an instance of a class that only "
        "exists in REPL globals. Expected: fork succeeds, normal win."
    ),
    code=(
        "class Tagger:\n"
        "    def __init__(self, ls):\n"
        "        self.ls = ls\n"
        "    def run(self):\n"
        "        return [llm_query('Two-word tag: ' + l) for l in self.ls]\n"
        "\n"
        "outs = Tagger(context.split('\\n')[:6]).run()\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 6),
)

_p(
    name="recursion",
    category="hard",
    n_expected_calls=4,
    hypothesis=(
        "A recursive function that calls the tool at each level, with "
        "each level's result concatenated. Serial by construction, so "
        "the floor applies; the test exists to prove the shadow does "
        "not blow the stack or duplicate dispatches under recursion."
    ),
    code=(
        "lines = context.split('\\n')\n"
        "def deep(i):\n"
        "    if i == 0:\n"
        "        return str(llm_query('One word: ' + lines[0]))\n"
        "    return deep(i - 1) + '|' + str(llm_query('One word: ' + lines[i]))\n"
        "\n"
        "r = deep(3)\n"
        "answer['content'] = r[:60]\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if str(ns.get("r", "")).count("|") == 3 else 0.0,
)

_p(
    name="fstring_force",
    category="edge",
    n_expected_calls=4,
    hypothesis=(
        "A lazy result interpolated into an f-string. __format__ must "
        "force the proxy (not print '<SpecValue ...>'), and the three "
        "later independent calls must still fan out. Expected: win, "
        "and no proxy repr leaking into the answer."
    ),
    code=(
        "lines = context.split('\\n')[:4]\n"
        "head = f\"topic={llm_query('One word topic: ' + lines[0])}\"\n"
        "rest = [llm_query('Two-word tag: ' + l) for l in lines[1:]]\n"
        "answer['content'] = head + '/' + str(len(rest))\nanswer['ready'] = True"
    ),
    check=lambda ns: (
        0.0 if "SpecValue" in str(ns.get("head", "")) else _ans_lines(ns, "rest", 3)
    ),
)

_p(
    name="dedup_mixed",
    category="edge",
    n_expected_calls=8,
    hypothesis=(
        "8 calls over 4 distinct prompts (each repeated twice). Claims "
        "are keyed by (tool, args), so duplicates must queue as "
        "separate FIFO claims — 8 dispatches, 8 hits, no cross-wiring. "
        "Expected: full win and 8 non-empty results."
    ),
    code=(
        "lines = context.split('\\n')[:4]\n"
        "outs = [llm_query('Three-word gist: ' + l) for l in lines + lines]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 8),
)

# ------------------------------- tool failures -------------------------------

_p(
    name="tool_error_raises",
    category="edge",
    n_expected_calls=4,
    hypothesis=(
        "The 4th call hits a tool that raises. A speculative dispatch "
        "must NOT surface its exception early: the error has to appear "
        "at the use site so control flow matches baseline exactly. "
        "Expected: identical failure in both modes, 3 results kept."
    ),
    code=(
        "lines = context.split('\\n')[:3]\n"
        "outs = [str(llm_query_fragile('Three-word gist: ' + l)) for l in lines]\n"
        "outs.append(str(llm_query_fragile('BOOM')))\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if len(ns.get("outs") or []) == 3 else 0.0,
)

_p(
    name="tool_error_recovered",
    category="edge",
    n_expected_calls=8,
    hypothesis=(
        "Every call is wrapped in try/except and one of them always "
        "raises, so the except branch runs a fallback call. Tests that "
        "a failed speculation is retracted, the fallback is dispatched, "
        "and the loop completes. Expected: 4 results, win preserved."
    ),
    code=(
        "lines = context.split('\\n')[:4]\n"
        "outs = []\n"
        "for i, l in enumerate(lines):\n"
        "    try:\n"
        "        outs.append(str(llm_query_fragile(('BOOM' if i == 2 else 'Two-word tag: ') + l)))\n"
        "    except Exception:\n"
        "        outs.append(str(llm_query('Two-word tag: ' + l)))\n"
        "\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 4),
)

# ----------------------------- unforkable state ------------------------------

_p(
    name="generator_partial",
    category="edge",
    n_expected_calls=3,
    hypothesis=(
        "A generator of calls, only 3 of 8 consumed. Generators cannot "
        "be deepcopied, so the shadow fork should FAIL and degrade to "
        "no speculation rather than guessing. Expected: ~1.0x (no win, "
        "no loss), correct results, and no dispatch for the 5 items "
        "that were never consumed."
    ),
    code=(
        "lines = context.split('\\n')[:8]\n"
        "gen = (llm_query('Two-word tag: ' + l) for l in lines)\n"
        "outs = [str(next(gen)) for _ in range(3)]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 3),
)

_p(
    name="dependent_args",
    category="hard",
    n_expected_calls=5,
    hypothesis=(
        "One call's OUTPUT becomes the next four calls' input. Nothing "
        "downstream can be speculated until the first result lands, so "
        "the only overlap is call 1 versus the rest of the stream. "
        "Expected: modest win (one call hidden), never slower."
    ),
    code=(
        "lines = context.split('\\n')[:5]\n"
        "topic = str(llm_query('Name one topic in: ' + lines[0]))[:60]\n"
        "outs = [llm_query('Relate ' + topic + ' to: ' + l) for l in lines[1:]]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 4),
)

# --------------------------- streaming geometry ------------------------------

_p(
    name="long_prose_burst",
    category="easy",
    n_expected_calls=12,
    hypothesis=(
        "~1.5k characters of reasoning prose, then a 12-way map. The "
        "head start is bounded by how much stream remains AFTER the "
        "block, not before it, so a long PREFIX should NOT help — this "
        "is the control for prose_sandwich's long suffix."
    ),
    prose_head=(
        "Let me think carefully about the structure of this data before "
        "writing any code at all. " * 14
    ),
    code=(
        "lines = (context.split('\\n') * 2)[:12]\n"
        "outs = [llm_query('Three-word gist: ' + l) for l in lines]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 12),
)

_MB3 = (
    "Step one: pick the slices.\n```repl\n"
    "lines = context.split('\\n')\n"
    "picks = [lines[i] for i in range(0, 10, 2)]\n"
    "print(len(picks), 'picked')\n"
    "```\nStep two: tag each slice.\n```repl\n"
    "tags = [llm_query('Key noun: ' + p) for p in picks]\n"
    "print('tagged', len(tags))\n"
    "```\nStep three: reduce.\n```repl\n"
    "summ = llm_query('Which noun repeats most? ' + ' ; '.join(str(t) for t in tags))\n"
    "answer['content'] = str(summ)[:80]\n"
    "answer['ready'] = True\n```\nThat completes the analysis."
)

_p(
    name="multiblock3",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "THREE dependent blocks in one response: slice, map, reduce. "
        "Shadow state must survive two block boundaries and the reduce "
        "must force the map's claims. Expected: the map fans out while "
        "block 3 is still streaming, so a win on top of multiblock."
    ),
    code="",
    check=lambda ns: _ans_lines(ns, "tags", 5) * 0.5 + (0.5 if ns.get("summ") else 0.0),
)
PATTERNS[-1].raw = _MB3


_GEN_BLOCKS = (
    "First build the lazy stream.\n```repl\n"
    "lines = context.split('\\n')[:8]\n"
    "gen = (llm_query('Two-word tag: ' + l) for l in lines)\n"
    "print('stream ready')\n"
    "```\nNow consume three of them.\n```repl\n"
    "outs = [str(next(gen)) for _ in range(3)]\n"
    "answer['content'] = str(len(outs))\n"
    "answer['ready'] = True\n```\nDone."
)

_p(
    name="generator_across_blocks",
    category="edge",
    n_expected_calls=3,
    hypothesis=(
        "A generator created in block 1 and consumed in block 2. The "
        "block-2 fork must deepcopy live state that INCLUDES a "
        "generator, which Python refuses. Expected: the shadow marks it "
        "unforkable and aborts on first use — no speculation, ~1.0x, "
        "and correct results. (generator_partial is the control where "
        "the generator never crosses a fork boundary.)"
    ),
    code="",
    check=lambda ns: _ans_lines(ns, "outs", 3),
)
PATTERNS[-1].raw = _GEN_BLOCKS

_p(
    name="taint_split",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "A NON-speculatable tool (env_probe) returns a value that feeds "
        "the first three call args; the last three are independent. "
        "Taint must flow BY VALUE: the 3 tainted calls cannot be "
        "speculated (misses) while the 3 clean ones still fan out "
        "(hits). Expected: hits=3, misses=3, correct results."
    ),
    code=(
        "tag = env_probe('topic')\n"
        "lines = context.split('\\n')[:6]\n"
        "tainted = [llm_query('About ' + str(tag) + ' -- ' + l) for l in lines[:3]]\n"
        "clean = [llm_query('Two-word tag: ' + l) for l in lines[3:]]\n"
        "answer['content'] = str(len(tainted) + len(clean))\nanswer['ready'] = True"
    ),
    check=lambda ns: 0.5 * _ans_lines(ns, "tainted", 3) + 0.5 * _ans_lines(ns, "clean", 3),
)
