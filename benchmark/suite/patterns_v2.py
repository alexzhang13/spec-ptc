"""V2 patterns (added after v1 review): realistic latencies, width scaling,
prefill pressure, peek-attribution control, mixed serial/parallel."""

from benchmark.suite.patterns import PATTERNS, Pattern, _ans_lines


def _p(**kw):
    PATTERNS.append(Pattern(**kw))


_p(
    name="map8_slow",
    category="easy",
    n_expected_calls=8,
    sub_max_tokens=320,
    hypothesis=(
        "map16's twin at REALISTIC sub-call latencies (~5-sentence "
        "outputs, 320-token cap): the latency regime of the OOLONG "
        "campaign. Expected: absolute savings grow with call length; "
        "wall bounded by stream + max(call)."
    ),
    code=(
        "lines = context.split('\\n')[:8]\n"
        "outs = [llm_query('Write five distinct sentences elaborating on: ' + l) for l in lines]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 8),
)

_p(
    name="chain4_slow",
    category="hard",
    n_expected_calls=4,
    sub_max_tokens=320,
    hypothesis=(
        "Serial chain at realistic latencies — the honest floor where "
        "each call is expensive. Expected ≈1.0×; never slower."
    ),
    code=(
        "s = context.split('\\n')[0]\n"
        "s = llm_query('Expand this by two sentences: ' + str(s)[:400])\n"
        "s = llm_query('Expand this by two sentences: ' + str(s)[:400])\n"
        "s = llm_query('Expand this by two sentences: ' + str(s)[:400])\n"
        "s = llm_query('Expand this by two sentences: ' + str(s)[:400])\n"
        "answer['content'] = str(s)[:100]\nanswer['ready'] = True"
    ),
    check=lambda ns: 1.0 if ns.get("s") else 0.0,
)

_p(
    name="map32",
    category="easy",
    n_expected_calls=32,
    hypothesis=(
        "Width scaling: 32-way map probes the launcher's max_inflight "
        "cap (16) — expected two dispatch waves, speedup ≈ "
        "min(width, cap) modulo stream. Compare against map16."
    ),
    code=(
        "lines = (context.split('\\n') * 2)[:32]\n"
        "outs = [llm_query('Three-word gist: ' + l) for l in lines]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 32),
)

_p(
    name="prefill_heavy",
    category="hard",
    n_expected_calls=6,
    sub_max_tokens=64,
    hypothesis=(
        "6 calls each carrying the WHOLE context (~2.5k chars) — "
        "prefill-bound speculation, exercising the prefill char budget "
        "path (6×2.5k fits the 120k default, so expect no throttling; "
        "the test documents the regime and pins engine behavior)."
    ),
    code=(
        "qs = ['revenue', 'safety', 'latency', 'quality', 'cloud', 'energy']\n"
        "outs = [llm_query('In the report lines below, how many mention ' + q + '? Answer with a number.\\n' + context) for q in qs]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 6),
)

_p(
    name="no_peek_control",
    category="edge",
    n_expected_calls=8,
    hypothesis=(
        "CONTROL for peek attribution: same for-loop shape as map16, "
        "but every argument flows through a model-defined helper "
        "function. The pre-close peek refuses to evaluate that (it only "
        "evaluates expressions it can prove safe), so 100% of the "
        "dispatches must come from shadow execution at statement close. "
        "Confirmed in the traces: 8/8 dispatches tagged `shadow`, "
        "versus 16/16 tagged `peek` for map16. The remaining speedup "
        "isolates what the shadow alone buys."
    ),
    code=(
        "def pick(i):\n"
        "    return context.split('\\n')[i]\n"
        "\n"
        "outs = []\n"
        "for i in range(8):\n"
        "    outs.append(llm_query('Four-word gist: ' + pick(i)))\n"
        "\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 8),
)

_p(
    name="side_branch_chain",
    category="hard",
    n_expected_calls=8,
    hypothesis=(
        "Mixed serial/parallel: a 4-step chain where each step ALSO "
        "fires an independent audit call. Expected: the chain stays "
        "serial (floor) but the 4 audits hide behind it entirely — "
        "spec wall ≈ chain alone, baseline pays chain + audits."
    ),
    code=(
        "s = context.split('\\n')[0]\n"
        "audits = []\n"
        "audits.append(llm_query('Rate clarity 1-5, digit only: ' + context.split('\\n')[0]))\n"
        "s = llm_query('Trim one word: ' + str(s)[:300])\n"
        "audits.append(llm_query('Rate clarity 1-5, digit only: ' + context.split('\\n')[1]))\n"
        "s = llm_query('Trim one word: ' + str(s)[:300])\n"
        "audits.append(llm_query('Rate clarity 1-5, digit only: ' + context.split('\\n')[2]))\n"
        "s = llm_query('Trim one word: ' + str(s)[:300])\n"
        "audits.append(llm_query('Rate clarity 1-5, digit only: ' + context.split('\\n')[3]))\n"
        "s = llm_query('Trim one word: ' + str(s)[:300])\n"
        "answer['content'] = str(len(audits))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "audits", 4),
)


_p(
    name="nonspec_interleave",
    category="edge",
    n_expected_calls=6,
    hypothesis=(
        "A side-effecting, NON-speculatable tool (log_note) fires "
        "between llm_query calls. The explicit tool separation must "
        "keep speculation alive around it (inert marker), the notes "
        "must run exactly once each (real run only), and the calls "
        "must still all hit."
    ),
    code=(
        "lines = context.split('\\n')[:6]\n"
        "outs = []\n"
        "for i in range(6):\n"
        "    log_note('starting ' + str(i))\n"
        "    outs.append(llm_query('Two-word tag: ' + lines[i]))\n"
        "\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    ),
    check=lambda ns: _ans_lines(ns, "outs", 6),
)
