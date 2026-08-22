"""EventBus + SpecEvent: every state change is observable."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class SpecEvent:
    kind: str  # token | stmt_closed | dispatch | adopt | ready | claim_* | evict | ...
    t: float = field(default_factory=time.perf_counter)
    data: dict = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Callable[[SpecEvent], None]] = []
        self._lock = threading.Lock()
        self.history: list[SpecEvent] = []
        self.record = True

    def subscribe(self, fn: Callable[[SpecEvent], None]) -> None:
        with self._lock:
            self._subs.append(fn)

    def emit(self, kind: str, **data) -> SpecEvent:
        ev = SpecEvent(kind=kind, data=data)
        with self._lock:
            if self.record:
                self.history.append(ev)
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(ev)
            except Exception:
                pass
        return ev


NULL_BUS = EventBus()
NULL_BUS.record = False
