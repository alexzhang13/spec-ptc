"""V6 (second wave, written from the wave-1 traces): overhead controls, waste
at realistic call lengths, and one composite that looks like an actual agent
turn. Wave 1 established that speculation wins where there is width; this wave
asks what it COSTS when there is nothing to win and how bad the waste gets when
the guess is wrong and the calls are expensive."""

from benchmark.suite.patterns import PATTERNS, Pattern, _ans_lines


def _p(**kw):
    PATTERNS.append(Pattern(**kw))


# ------------------------------ overhead controls ------------------------------

_p(
    name="nonspec_only",
    category="hard",
    n_expected_calls=0,
    hypothesis=(
        "A block whose only tool is NON-speculatable (log_note) plus "
        "one plain string call. There is nothing to speculate, so this "
        "measures pure harness overhead: streaming, segmenting, forking "
        "the shadow and running it for no benefit. Expected: 1.0x "
        "within noise, zero dispatches."
    ),
    code=(
        "lines = context.split('\\n')[:12]\n"
        "for i, l in enumerate(lines):\n"
        "    log_note(str(i) + ':' + l[:20])\n"
        "\n"
        "answer['content'] = str(len(lines))\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if len(ns.get("lines") or []) == 12 else 0.0,
)

_p(
    name="pure_compute",
    category="hard",
    n_expected_calls=0,
    hypothesis=(
        "No tool calls at all, but real CPU work (nested loops over the "
        "context). The shadow re-executes all of it for nothing and the "
        "runaway watchdog may cut it short. Either way the real turn "
        "must not slow down. Expected: 1.0x, zero dispatches, and any "
        "shadow abort must be recorded rather than silent."
    ),
    code=(
        "words = context.replace('\\n', ' ').split()\n"
        "acc = 0\n"
        "for i in range(400):\n"
        "    for w in words:\n"
        "        acc += len(w) * (i % 7)\n"
        "\n"
        "answer['content'] = str(acc)\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if int(ns.get("acc", 0)) > 0 else 0.0,
)

# ------------------------- waste at realistic latency -------------------------

_p(
    name="mutating_loop_slow",
    category="edge",
    n_expected_calls=6,
    sub_max_tokens=320,
    hypothesis=(
        "mutating_loop's twin with EXPENSIVE calls: the loop mutates "
        "the list it iterates, so the shadow's guesses for later "
        "iterations are wrong and get retracted. With 320-token calls "
        "the retracted work is real GPU work, so `calls_aborted` and "
        "`tokens_x` should both rise. Expected: still >= 1.0x wall, "
        "with the extra token cost visible and bounded."
    ),
    code=(
        "lines = context.split('\\n')[:6]\n"
        "outs = []\n"
        "for i in range(6):\n"
        "    outs.append(llm_query('Write four sentences about: ' + lines[i]))\n"
        "    lines[min(i + 1, 5)] = 'MUTATED ' + lines[min(i + 1, 5)]\n"
        "\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 6),
)

_p(
    name="wide_then_discard",
    category="edge",
    n_expected_calls=8,
    sub_max_tokens=320,
    hypothesis=(
        "The block speculates an 8-wide map with expensive calls, then "
        "the real path only ever reads two of the results (the rest are "
        "discarded by a slice). The worst case for wasted GPU work with "
        "no correctness risk. Expected: never slower, and the token "
        "overhead is the price of the win elsewhere — reported, not "
        "hidden."
    ),
    code=(
        "lines = context.split('\\n')[:8]\n"
        "outs = [llm_query('Write four sentences about: ' + l) for l in lines]\n"
        "kept = [str(o)[:40] for o in outs[:2]]\n"
        "answer['content'] = ' | '.join(kept)\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "kept", 2),
)

_p(
    name="retry_loop",
    category="edge",
    n_expected_calls=4,
    hypothesis=(
        "A while-loop that retries until the model returns something "
        "digit-like (capped at 4 tries). The number of iterations "
        "depends on real sampled text, so the shadow and the real run "
        "can disagree about how many calls happen — genuine model "
        "nondeterminism, not a scripted trick. Expected: correct result "
        "either way; misses or evictions when the counts differ."
    ),
    code=(
        "line = context.split('\\n')[0]\n"
        "tries = 0\n"
        "got = ''\n"
        "while tries < 4 and not got.strip()[:1].isdigit():\n"
        "    got = str(llm_query('Reply with ONLY one digit 1-9 about: ' + line))\n"
        "    tries += 1\n"
        "\n"
        "answer['content'] = got[:20] + '/' + str(tries)\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if 1 <= int(ns.get("tries", 0)) <= 4 else 0.0,
)

_p(
    name="deep_chain12",
    category="hard",
    n_expected_calls=12,
    hypothesis=(
        "A 12-step serial chain — the floor at scale. Each call needs "
        "the previous result, so at most the first step can overlap the "
        "stream. Expected: speedup close to 1.0x and clearly below the "
        "fan-out patterns; this is the honest 'no width, no win' case."
    ),
    code=(
        "s = context.split('\\n')[0][:80]\n"
        "for _ in range(12):\n"
        "    s = str(llm_query('Rewrite in five words: ' + s))[:80]\n"
        "\n"
        "answer['content'] = s\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if len(str(ns.get("s", ""))) > 0 else 0.0,
)

# ------------------------------ realistic composite ---------------------------

_COMPOSITE = (
    "I'll extract the numbers first, then rank them, then double-check the "
    "top hit.\n```repl\n"
    "lines = context.split('\\n')[:10]\n"
    "log_note('extracting ' + str(len(lines)))\n"
    "nums = [llm_query('Reply with only the percentage in: ' + l) for l in lines]\n"
    "print('extracted', len(nums))\n"
    "```\n"
    "Good. Now I will rank them and verify the top one, which requires a "
    "second pass over the extracted values.\n```repl\n"
    "pairs = [(str(n)[:8], l[:30]) for n, l in zip(nums, lines)]\n"
    "rank = llm_query('Which is largest? ' + '; '.join(p[0] for p in pairs))\n"
    "checks = [llm_query('Is ' + p[0] + ' a percentage? yes/no') for p in pairs[:4]]\n"
    "answer['content'] = str(rank)[:60] + ' | ' + str(len(checks))\n"
    "answer['ready'] = True\n```\n"
    "That gives the ranked answer with a verification pass."
)

_p(
    name="agent_turn",
    category="easy",
    n_expected_calls=15,
    hypothesis=(
        "A composite that looks like a real agent turn: reasoning "
        "prose, a non-speculatable log call, a 10-wide extraction, a "
        "second block that reduces and then verifies 4 items. Mixes "
        "every mechanism the suite tests separately. Expected: a strong "
        "win driven by the two fan-outs, with the serial reduce in the "
        "middle limiting it below a pure map."
    ),
    code="",
    check=lambda ns: 0.5 * _ans_lines(ns, "nums", 10) + 0.5 * _ans_lines(ns, "checks", 4),
)
PATTERNS[-1].raw = _COMPOSITE

_p(
    name="batched_duplicates",
    category="edge",
    n_expected_calls=8,
    hypothesis=(
        "A batched call whose prompt list contains each prompt twice. "
        "Batched tools are claimed as a unit, so duplicate members must "
        "not collapse or cross-wire. Expected: 8 results in list order, "
        "matching baseline exactly."
    ),
    code=(
        "lines = context.split('\\n')[:4]\n"
        "outs = llm_query_batched([('Three-word gist: ' + l) for l in lines + lines])\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 8),
)
