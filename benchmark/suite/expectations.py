"""Pre-registered, machine-checkable claims — one per pattern, derived from its
`hypothesis` text and fixed BEFORE the r=5 run that the tables report.

Two claims per pattern:
  win  = speculation should give a real speedup (>= WIN_MIN)
  floor= speculation must never be materially slower (>= FLOOR)
Plus one global claim: scores must match between modes (|d| <= PARITY_EPS).
"""

WIN_MIN = 1.30
FLOOR = 0.85
PARITY_EPS = 0.05

# patterns where a real speedup is claimed
WIN = {
    "map16",
    "map_reduce",
    "two_stage",
    "late_loop",
    "mutating_loop",
    "early_break",
    "identical10",
    "while_computed",
    "multiblock",
    "prose_sandwich",
    "map8_slow",
    "map32",
    "prefill_heavy",
    "no_peek_control",
    "side_branch_chain",
    "nonspec_interleave",
    "sweep_w04",
    "sweep_w08",
    "sweep_w16",
    "sweep_w32",
    "sweep_w64",
    "sweep_len032",
    "sweep_len096",
    "sweep_len320",
    "sweep_len640",
    "dict_comp",
    "nested_comp",
    "dedup_mixed",
    "fstring_force",
    "tool_error_recovered",
    "dependent_args",
    "long_prose_burst",
    "multiblock3",
    "generator_across_blocks",
    "taint_split",
    "turn2_map",
    "turn2_lazy_carry",
    "turn3_chain",
    "class_method",
    # wave 2
    "mutating_loop_slow",
    "wide_then_discard",
    "agent_turn",
}

# patterns whose claim is only the floor (serial, degenerate, or degraded)
FLOOR_ONLY = {
    "serial_chain6",
    "chain4_slow",
    "branch_on_result",
    "jail_break_mid",
    "crash_after_dispatch",
    "syntax_gauntlet",
    "batched8",
    "recursion",
    "sorted_key",
    "generator_partial",
    "tool_error_raises",
    "sweep_w01",
    "sweep_w02",
    "turn2_generator",
    # wave 2
    "nonspec_only",
    "pure_compute",
    "retry_loop",
    "deep_chain12",
    "batched_duplicates",
}


# Mechanism check: how many speculative dispatches the engine SHOULD make in
# spec mode. Defaults to the pattern's call count; these are the cases where a
# lower number is the correct answer (taint, unforkable state, no tools, a call
# that raises before the rest run, or a model-dependent loop count).
DISPATCH = {
    "taint_split": 3,  # 3 tainted args cannot be guessed
    # CORRECTED after reading the traces: the engine does speculate the call
    # that raises (4th dispatch); the failed speculation is dropped and the real
    # call re-raises at the use site, giving 3 hits + 1 miss. The first value
    # here (3) was our miscount, not an engine fault.
    "tool_error_raises": 4,
    "turn2_generator": 0,  # unforkable state -> no speculation
    "nonspec_only": 0,  # nothing speculatable in the block
    "pure_compute": 0,  # no tool calls at all
    "generator_partial": 3,  # only the consumed items exist
    # 4 fragile calls + 1 except-branch fallback = 5 real calls, not 8
    "tool_error_recovered": 5,
    "jail_break_mid": 3,  # shadow aborts at the illegal import
    "crash_after_dispatch": None,
    "syntax_gauntlet": None,
    "retry_loop": None,  # iteration count depends on sampled text
    "mutating_loop": None,  # divergence -> re-dispatch, count varies
    "mutating_loop_slow": None,
    "early_break": None,
    "branch_on_result": None,
    "while_computed": None,
    "deep_chain12": None,  # each step re-guesses after divergence
    "batched_duplicates": None,  # batched tools claim as one unit
    "batched8": None,
}


def expected_dispatch(name: str, n_calls):
    """Expected speculative dispatch count, or None if not claimed."""
    return DISPATCH.get(name, n_calls) if name in DISPATCH else n_calls


def claim(name: str) -> str:
    if name in WIN:
        return "win"
    if name in FLOOR_ONLY:
        return "floor"
    return "unclaimed"


def verdict(name: str, speedup: float, score_delta: float) -> str:
    """SUPPORTED / REFUTED / (parity failures are fatal regardless of speed)."""
    if abs(score_delta) > PARITY_EPS:
        return "REFUTED(parity)"
    c = claim(name)
    if speedup < FLOOR:
        return "REFUTED(slower)"
    if c == "win":
        return "SUPPORTED" if speedup >= WIN_MIN else "REFUTED(no win)"
    if c == "floor":
        return "SUPPORTED"
    return "n/a"
