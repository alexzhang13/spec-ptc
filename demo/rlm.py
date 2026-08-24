"""SpeculativeLocalREPL: RLM's LocalREPL + spec-ptc speculation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlm.environments.local_repl import _SAFE_BUILTINS, LocalREPL

from spec_ptc.contracts.events import EventBus
from spec_ptc.contracts.tools import ToolRegistry
from spec_ptc.speculator import SpecSession, StreamTurn


class SpeculativeLocalREPL(LocalREPL):
    def __init__(
        self,
        *args: Any,
        spec_bus: EventBus | None = None,
        spec_max_inflight: int = 16,
        subcall_override: Callable[..., str] | None = None,
        auto_speculate: bool = True,
        **kwargs: Any,
    ) -> None:
        self._auto_speculate = auto_speculate
        super().__init__(*args, **kwargs)
        self.bus = spec_bus or EventBus()
        self.reg = ToolRegistry()
        base_single = subcall_override or LocalREPL._llm_query.__get__(self)
        base_batched = (
            (lambda prompts, model=None: [subcall_override(p) for p in prompts])
            if subcall_override
            else LocalREPL._llm_query_batched.__get__(self)
        )
        self.reg.register(
            "llm_query", base_single, speculatable=True, pure=True, latency_hint_ms=2000
        )
        self.reg.register("llm_query_batched", base_batched, speculatable=True, pure=True,
                          batched=True, latency_hint_ms=2000)
        # unmarked (never speculated): the shadow returns an inert marker
        # instead of NameError-aborting or recursing into a real sub-RLM
        self.reg.register("rlm_query", LocalREPL._rlm_query.__get__(self))
        self.reg.register("rlm_query_batched", LocalREPL._rlm_query_batched.__get__(self))
        self.session = SpecSession(self.reg, self.bus, max_inflight=spec_max_inflight,
                                   max_dispatches_per_turn=4096)
        self.spec_store = self.session.store
        self.spec_launcher = self.session.launcher
        self._real_hooks = self.session.real_hooks()
        self._turn: StreamTurn | None = None
        # rebind the REPL-visible hooks to the claiming versions
        self.globals["llm_query"] = self._spec_llm_query
        self.globals["llm_query_batched"] = self._spec_llm_query_batched

    # ---- the REPL-visible hooks (waiting mode) -------------------------------
    def _spec_llm_query(self, prompt: str, model: str | None = None) -> str:
        return self._real_hooks["llm_query"](prompt)

    def _spec_llm_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        return self._real_hooks["llm_query_batched"](prompts)

    # keep scaffold restoration pointing at the claiming hooks
    def _restore_scaffold(self) -> None:  # type: ignore[override]
        super()._restore_scaffold()
        if hasattr(self, "_real_hooks"):
            self.globals["llm_query"] = self._spec_llm_query
            self.globals["llm_query_batched"] = self._spec_llm_query_batched

    # ---- streaming side (speculative mode) ------------------------------------
    def begin_stream_turn(self) -> None:
        safe = {k: v for k, v in _SAFE_BUILTINS.items() if v is not None}
        self._turn = self.session.begin_stream_turn({**self.locals}, safe)

    def feed(self, delta: str) -> None:
        assert self._turn is not None, "begin_stream_turn() first"
        self._turn.feed(delta)

    def end_stream_turn(self, timeout: float = 600) -> None:
        if self._turn is not None:
            self._turn.end(timeout)
            self._turn = None

    def execute_code(self, code: str):  # type: ignore[override]
        # Zero-integration win: a caller that never used the streaming API
        # (stock RLM loop) still gets the lazy fan-out — run a shadow pre-pass
        # over the complete block before real execution.
        if (
            hasattr(self, "session")
            and getattr(self, "_auto_speculate", False)
            and self._turn is None
        ):
            self.begin_stream_turn()
            self._turn.feed(f"```repl\n{code}\n```\n")
            self.end_stream_turn()
        result = super().execute_code(code)
        if hasattr(self, "session"):
            self.session.end_turn()
        return result


def patch_rlm():
    """Make the installed rlms package use SpeculativeLocalREPL everywhere."""
    import rlm.environments as envs

    envs.local_repl.LocalREPL = SpeculativeLocalREPL  # type: ignore[misc]
    envs.LocalREPL = SpeculativeLocalREPL  # type: ignore[misc]
    return SpeculativeLocalREPL
