"""V5: multi-TURN patterns. Everything up to V4 lives inside one model response,
so the shadow forks once from a clean namespace. Real agents run many turns, and
turn N's fork must deepcopy whatever turn N-1 left behind — including values that
cannot be copied at all. These patterns exercise that boundary."""

from benchmark.suite.patterns import PATTERNS, Pattern, _ans_lines


def _p(**kw):
    PATTERNS.append(Pattern(**kw))


def _turn(body: str, head: str = "Next step.\n", tail: str = "Continuing.") -> str:
    return f"{head}\n```repl\n{body}\n```\n{tail}"


_p(
    name="turn2_map",
    category="edge",
    n_expected_calls=8,
    hypothesis=(
        "Turn 1 builds plain data; turn 2 maps 8 calls over it. Turn 2's "
        "fork must deepcopy turn 1's namespace before it can speculate. "
        "Expected: the same fan-out win as a single-turn map, proving "
        "cross-turn state carries correctly."
    ),
    code="",
    turns=(
        _turn(
            "picks = [l for l in context.split('\\n')[:8]]\nprint(len(picks), 'picked')",
            head="First I gather the lines.\n",
        ),
        _turn(
            "outs = [llm_query('Three-word gist: ' + p) for p in picks]\n"
            "answer['content'] = str(len(outs))\n"
            "answer['ready'] = True",
            head="Now I query each one.\n",
        ),
    ),
    check=lambda ns: _ans_lines(ns, "outs", 8),
)

_p(
    name="turn2_generator",
    category="edge",
    n_expected_calls=3,
    hypothesis=(
        "Turn 1 leaves a GENERATOR in the namespace; turn 2 consumes "
        "three items. Generators refuse deepcopy, so turn 2's fork must "
        "mark the value unforkable and abort on first use instead of "
        "guessing. Expected: no speculation in turn 2 (~1.0x), correct "
        "results, and a recorded abort reason."
    ),
    code="",
    turns=(
        _turn(
            "lines = context.split('\\n')[:8]\n"
            "gen = (llm_query('Two-word tag: ' + l) for l in lines)\n"
            "print('stream ready')",
            head="Set up a lazy stream.\n",
        ),
        _turn(
            "outs = [str(next(gen)) for _ in range(3)]\n"
            "answer['content'] = str(len(outs))\n"
            "answer['ready'] = True",
            head="Consume three.\n",
        ),
    ),
    check=lambda ns: _ans_lines(ns, "outs", 3),
)

_p(
    name="turn2_lazy_carry",
    category="hard",
    n_expected_calls=6,
    hypothesis=(
        "Turn 1 dispatches 6 calls and stores the results WITHOUT using "
        "them, so they may still be lazy proxies when the turn ends; "
        "turn 2 forces them. Either the engine settles them at turn end "
        "or the proxies survive the next fork — both are correct, "
        "neither may lose or duplicate a result. Expected: 6 hits in "
        "turn 1, no re-dispatch in turn 2, correct strings."
    ),
    code="",
    turns=(
        _turn(
            "lines = context.split('\\n')[:6]\n"
            "outs = [llm_query('Three-word gist: ' + l) for l in lines]\n"
            "print('queued')",
            head="Queue the queries.\n",
        ),
        _turn(
            "joined = ' / '.join(str(o) for o in outs)\n"
            "answer['content'] = joined[:120]\n"
            "answer['ready'] = True",
            head="Now combine them.\n",
        ),
    ),
    check=lambda ns: (
        0.0 if "SpecValue" in str(ns.get("joined", "")) else _ans_lines(ns, "outs", 6)
    ),
)

_p(
    name="turn3_chain",
    category="hard",
    n_expected_calls=9,
    hypothesis=(
        "Three turns, each mapping 3 calls over the previous turn's "
        "results — the shape of a real agent loop. Each turn's map "
        "should overlap that turn's stream, so the win should compound "
        "across turns rather than decay."
    ),
    code="",
    turns=(
        _turn(
            "lines = context.split('\\n')[:3]\n"
            "a = [llm_query('Key noun: ' + l) for l in lines]\n"
            "print('a done')",
            head="Round one.\n",
        ),
        _turn(
            "b = [llm_query('Antonym of the main noun in: ' + str(x)[:60]) for x in a]\n"
            "print('b done')",
            head="Round two.\n",
        ),
        _turn(
            "c = [llm_query('One adjective for: ' + str(x)[:60]) for x in b]\n"
            "answer['content'] = str(len(a) + len(b) + len(c))\n"
            "answer['ready'] = True",
            head="Round three.\n",
        ),
    ),
    check=lambda ns: (
        (_ans_lines(ns, "a", 3) + _ans_lines(ns, "b", 3) + _ans_lines(ns, "c", 3)) / 3
    ),
)
