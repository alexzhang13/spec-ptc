"""Main entrypoint for the REPL PTC speculator.

  Speculator   — ``@tool`` / ``hooks()`` / ``turn()`` over a session
  SpecSession  — store + launcher + hook factories (what daemon.py uses)
  StreamTurn   — ``feed(delta)`` → StreamSegmenter → ShadowRunner (+ tail peek)

Tools are ID'd by their inputs, and a unique ID for identical inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from spec_ptc.contracts.events import EventBus
from spec_ptc.contracts.tools import Tool, ToolRegistry
from spec_ptc.engine.shadow import ShadowRunner
from spec_ptc.engine.speculation import (
    Budget,
    Launcher,
    SpecStore,
    make_baseline_hooks,
    make_real_hooks,
    make_shadow_hooks,
)
from spec_ptc.engine.streaming import StreamSegmenter


def _real_builtins() -> dict:
    from spec_ptc.runtime.harness import REAL_BUILTINS

    return REAL_BUILTINS


class SpeculativeExecutor(Protocol):
    """What a language runtime implements to get shadow semantics (L4)."""

    def begin_stream_turn(self) -> None: ...
    def feed(self, delta: str) -> None: ...
    def end_stream_turn(self, timeout: float = 600) -> None: ...
    def execute_code(self, code: str): ...


@dataclass
class StreamTurn:
    """One streaming turn's segmenter + shadow, bound to a host namespace."""

    segmenter: StreamSegmenter
    shadow: ShadowRunner
    peek: bool = True
    _last_tail: str = ""

    def feed(self, delta: str) -> None:
        for seg in self.segmenter.feed(delta):
            self.shadow.feed(seg)
        if self.peek and "\n" in delta:
            tail = self.segmenter.pending_tail()
            if tail.strip() and tail != self._last_tail and self._peek_worthwhile(tail):
                self._last_tail = tail
                self.shadow.feed_peek(tail)

    def _peek_worthwhile(self, tail: str) -> bool:
        """Re-planning costs O(tail); do it only when the content that changed
        since the last plan could introduce a new dispatchable call — i.e. it
        mentions a hooked tool. Keeps pure-code compounds at zero peek cost."""
        if len(tail) > 12_000:
            return False
        changed = tail[len(self._last_tail) :] if tail.startswith(self._last_tail) else tail
        if any(name in changed for name in self.shadow.hooks):
            return True
        # outstanding bets: ANY new line may invalidate a prior plan (e.g. a
        # body line mutating the loop's iterable) — re-plan so bet retraction
        # (shadow._peek reconcile) can fire mid-stream instead of at turn end
        return bool(self.shadow._last_peek_tally)

    def end(self, timeout: float = 600) -> None:
        for seg in self.segmenter.finish():
            self.shadow.feed(seg)
        self.shadow.finish()
        self.shadow.join(timeout)
        self.shadow.abort("turn_end")


class SpecSession:
    """Store + launcher + hook factories around a host's tool registry (L2)."""

    def __init__(
        self, registry: ToolRegistry, bus: EventBus | None = None,
        max_inflight: int = 16, max_dispatches_per_turn: int = 2048,
    ) -> None:
        self.reg = registry
        self.bus = bus or EventBus()
        self.store = SpecStore(self.bus)
        self.launcher = Launcher(
            self.store, self.bus,
            Budget(max_inflight=max_inflight,
                   max_dispatches_per_turn=max_dispatches_per_turn))

    def real_hooks(self) -> dict:
        return make_real_hooks(self.reg, self.store, self.launcher, self.bus)

    def baseline_hooks(self) -> dict:
        return make_baseline_hooks(self.reg)

    def begin_stream_turn(
        self, host_locals: dict, safe_builtins: dict, peek: bool = True
    ) -> StreamTurn:
        shadow = ShadowRunner(
            host_locals,
            make_shadow_hooks(self.reg, self.launcher),
            self.store,
            safe_builtins,
            self.bus,
            launcher=self.launcher,
            registry=self.reg,
        )
        return StreamTurn(StreamSegmenter(), shadow, peek=peek)

    def end_turn(self) -> None:
        self.store.evict_unclaimed("turn_end")
        self.launcher.budget.reset()

    def close(self) -> None:
        self.launcher.shutdown()


class Speculator:
    def __init__(self, max_inflight: int = 16, bus: EventBus | None = None) -> None:
        self.registry = ToolRegistry()
        self.bus = bus or EventBus()
        self._session: SpecSession | None = None
        self.max_inflight = max_inflight

    # ---------------------------------------------------------- registration
    def tool(
        self,
        *,
        speculatable: bool = False,
        pure: bool = False,
        deterministic: bool = False,
        latency_hint_ms: float = 1000.0,
        batched: bool = False,
        name: str | None = None,
        cancel: Callable | None = None,
        claim_key: Callable | None = None,
        gate: Callable | None = None,
    ) -> Callable:
        """Decorator: register a plain function as a tool. Unmarked functions
        are never speculated."""

        def deco(fn: Callable) -> Callable:
            self.registry.register(
                name or fn.__name__,
                fn,
                speculatable=speculatable,
                pure=pure,
                deterministic=deterministic,
                latency_hint_ms=latency_hint_ms,
                batched=batched,
                cancel_fn=cancel,
                key_fn=claim_key,
                gate_fn=gate,
            )
            return fn

        return deco

    def add(self, tool: Tool) -> None:
        """Register a tool_api.Tool instance (the modular class interface)."""
        self.registry.register_tool(tool)

    # ---------------------------------------------------------- runtime
    @property
    def session(self) -> SpecSession:
        if self._session is None:
            self._session = SpecSession(self.registry, self.bus, max_inflight=self.max_inflight)
        return self._session

    def hooks(self) -> dict[str, Callable]:
        """Claiming hooks to install in the host REPL's namespace. Real-run
        semantics are byte-identical to calling the tools directly."""
        return self.session.real_hooks()

    @contextmanager
    def turn(
        self,
        repl_locals: dict | None = None,
        safe_builtins: dict | None = None,
        peek: bool = True,
    ):
        """One streaming turn: feed model deltas inside the block; on exit the
        shadow drains and, after your real execution, call end_turn() —
        or just let the NEXT turn() do the cleanup."""
        self.session.end_turn()  # evict any leftovers from a prior turn
        t = self.session.begin_stream_turn(
            dict(repl_locals or {}), safe_builtins or _real_builtins(), peek=peek
        )
        try:
            yield t
        finally:
            t.end()

    def end_turn(self) -> None:
        self.session.end_turn()

    def stats(self) -> dict:
        store = self.session.store
        return {
            "speculated": len(store.all),
            "claimed": sum(1 for s in store.all if s.state == "claimed"),
            "evicted": sum(1 for s in store.all if s.state == "evicted"),
        }

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
