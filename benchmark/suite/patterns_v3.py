"""V3: parametric sweeps. Two axes control whether speculation pays — how many
independent calls a block has (width) and how long each call takes (length).
These families vary one axis at a time so the suite yields a phase diagram
instead of anecdotes."""

from benchmark.suite.patterns import PATTERNS, Pattern, _ans_lines

WIDTHS = (1, 2, 4, 8, 16, 32, 64)
LENGTHS = (32, 96, 320, 640)
CAP = 16  # engine max_inflight


def _p(**kw):
    PATTERNS.append(Pattern(**kw))


def _map_code(w: int, prompt: str) -> str:
    return (
        f"lines = (context.split('\\n') * 3)[:{w}]\n"
        f"outs = [llm_query({prompt!r} + l) for l in lines]\n"
        "answer['content'] = str(len(outs))\nanswer['ready'] = True"
    )


for _w in WIDTHS:
    _p(
        name=f"sweep_w{_w:02d}",
        category="sweep",
        n_expected_calls=_w,
        sub_max_tokens=96,
        hypothesis=(
            f"Width sweep, call length fixed (96-token cap): {_w} independent "
            f"calls. Baseline wall grows linearly in width; speculative wall "
            f"should grow only in ceil(width/{CAP}) dispatch waves, so speedup "
            f"should rise roughly linearly to the inflight cap ({CAP}) and then "
            "flatten. w01 is the degenerate case and must not regress."
        ),
        code=_map_code(_w, "Three-word gist: "),
        check=lambda ns, w=_w: _ans_lines(ns, "outs", w),
    )

for _t in LENGTHS:
    _p(
        name=f"sweep_len{_t:03d}",
        category="sweep",
        n_expected_calls=8,
        sub_max_tokens=_t,
        hypothesis=(
            f"Length sweep, width fixed at 8: each call capped at {_t} tokens "
            "with a prompt that actually fills the cap. Speedup should stay "
            "roughly flat in call length (both modes pay one call's latency; "
            "spec pays it once, baseline eight times) while ABSOLUTE savings "
            "grow linearly — the regime that matters for real agent workloads."
        ),
        code=_map_code(8, "Write a detailed multi-sentence analysis of: "),
        check=lambda ns: _ans_lines(ns, "outs", 8),
    )
