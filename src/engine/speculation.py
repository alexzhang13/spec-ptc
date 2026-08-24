"""Speculation lifecycle: Speculation, lazy SpecValue, SpecStore (FIFO claim/adopt/evict),
Launcher + Budget, and the three hook modes (baseline / real-claim / shadow-dispatch)."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from spec_ptc.contracts.events import NULL_BUS, EventBus
from spec_ptc.contracts.tools import (
    NonSpeculated,
    SpecKey,
    ToolRegistry,
    ToolSpec,
    contains_nonspec,
    spec_key,
)

SpecState = Literal["pending", "running", "ready", "claimed", "evicted", "failed"]


@dataclass
class Speculation:
    key: SpecKey
    seq: int  # global dispatch order (== claim order)
    args: tuple
    kwargs: dict
    source: Literal["shadow", "lazy-real"]
    state: SpecState = "pending"
    result: Any = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)
    adopted: bool = False  # a shadow hook took ownership of this peek
    cancel: Callable[[], None] | None = None
    # timing (perf_counter seconds)
    t_dispatch: float = 0.0
    t_ready: float = 0.0
    t_claim: float = 0.0
    args_preview: str = ""

    def wait(self, timeout: float | None = None) -> Any:
        if not self.done.wait(timeout):
            raise TimeoutError(f"speculation {self.key} not ready after {timeout}s")
        if self.error is not None:
            raise self.error
        return self.result


_UNSET = object()

# The shadow thread registers a ForceTracker here so its runaway watchdog can
# distinguish "blocked waiting for a speculation" (legitimate, unbounded) from
# "spinning in model-written compute" (budgeted). See shadow.py.
force_ctx = threading.local()


class ForceTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.forced_total = 0.0  # seconds spent inside spec waits
        self.force_start: float | None = None

    def begin(self) -> None:
        with self.lock:
            self.force_start = time.perf_counter()

    def end(self) -> None:
        with self.lock:
            if self.force_start is not None:
                self.forced_total += time.perf_counter() - self.force_start
                self.force_start = None

    def forced_seconds(self) -> float:
        with self.lock:
            cur = (
                time.perf_counter() - self.force_start if self.force_start is not None else 0.0
            )
            return self.forced_total + cur


async def _identity(v: Any) -> Any:
    """Await-protocol shim: completes immediately with `v`, never suspends."""
    return v


class SpecValue:
    __slots__ = ("_spec", "_val")

    def __init__(self, spec: Speculation) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_val", _UNSET)

    def _force(self) -> Any:
        val = object.__getattribute__(self, "_val")
        if val is _UNSET:
            spec: Speculation = object.__getattribute__(self, "_spec")
            tracker = getattr(force_ctx, "tracker", None)
            if tracker is not None:
                tracker.begin()
            try:
                val = spec.wait(timeout=600)
            finally:
                if tracker is not None:
                    tracker.end()
            object.__setattr__(self, "_val", val)
        return val

    def __await__(self):
        """`x = await llm(...)` binds the LAZY proxy, exactly like the sync
        form — awaiting must not be the force point, or every async tool
        call would block at its own await instead of at first use."""
        return _identity(self).__await__()

    # ---- everything below forces ----
    def __str__(self):
        return str(self._force())

    def __repr__(self):
        return repr(self._force())

    def __format__(self, s):
        return format(self._force(), s)

    def __bool__(self):
        return bool(self._force())

    def __len__(self):
        return len(self._force())

    def __iter__(self):
        return iter(self._force())

    def __contains__(self, x):
        return x in self._force()

    def __getitem__(self, i):
        return self._force()[i]

    def __eq__(self, o):
        return self._force() == deep_force(o)

    def __ne__(self, o):
        return self._force() != deep_force(o)

    def __lt__(self, o):
        return self._force() < deep_force(o)

    def __le__(self, o):
        return self._force() <= deep_force(o)

    def __gt__(self, o):
        return self._force() > deep_force(o)

    def __ge__(self, o):
        return self._force() >= deep_force(o)

    def __hash__(self):
        return hash(self._force())

    def __add__(self, o):
        return self._force() + deep_force(o)

    def __radd__(self, o):
        return deep_force(o) + self._force()

    def __mul__(self, o):
        return self._force() * deep_force(o)

    def __int__(self):
        return int(self._force())

    def __float__(self):
        return float(self._force())

    def __getattr__(self, name):
        return getattr(self._force(), name)


def deep_force(obj: Any, depth: int = 3) -> Any:
    """Replace SpecValues with concrete values through common containers."""
    if isinstance(obj, SpecValue):
        return obj._force()
    if depth <= 0:
        return obj
    if isinstance(obj, list):
        return [deep_force(x, depth - 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(deep_force(x, depth - 1) for x in obj)
    if isinstance(obj, dict):
        return {k: deep_force(v, depth - 1) for k, v in obj.items()}
    return obj


def force_in_place(obj: Any, depth: int = 3) -> None:
    """Force SpecValues inside mutable containers, mutating in place (used by the
    shadow's per-statement read-set sync so the namespace object identity survives)."""
    if depth <= 0:
        return
    if isinstance(obj, list):
        for i, x in enumerate(obj):
            if isinstance(x, SpecValue):
                obj[i] = x._force()
            else:
                force_in_place(x, depth - 1)
    elif isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, SpecValue):
                obj[k] = v._force()
            else:
                force_in_place(v, depth - 1)


class SpecStore:
    def __init__(self, bus: EventBus = NULL_BUS) -> None:
        self._q: dict[SpecKey, deque[Speculation]] = {}
        self._lock = threading.Lock()
        self.bus = bus
        self.all: list[Speculation] = []  # every speculation ever, for metrics

    def put(self, spec: Speculation) -> None:
        with self._lock:
            self._q.setdefault(spec.key, deque()).append(spec)
            self.all.append(spec)

    def claim(self, key: SpecKey, reuse: bool = False) -> Speculation | None:
        """Pop the oldest unclaimed speculation for this key (hit), else None (miss).

        Multiplicity: N identical dispatches queue N independent speculations;
        the k-th claim pops the k-th. Never collapses nondeterministic results.
        """
        with self._lock:
            q = self._q.get(key)
            # deterministic tools reuse the one cached run; sub-LMs need a fresh sample per call → pop
            if reuse:
                for spec in q or ():
                    if spec.state in ("pending", "running", "ready", "claimed"):
                        spec.state = "claimed" if spec.state == "ready" else spec.state
                        spec.t_claim = spec.t_claim or time.perf_counter()
                        return spec
                return None
            while q:
                spec = q.popleft()
                if spec.state in ("pending", "running", "ready"):
                    spec.state = "claimed" if spec.state == "ready" else spec.state
                    spec.t_claim = time.perf_counter()
                    return spec
            return None

    def existing(self, key: SpecKey) -> Speculation | None:
        with self._lock:
            for spec in self._q.get(key, ()):
                if spec.state in ("pending", "running", "ready", "claimed"):
                    return spec
            return None

    def adopt(self, key: SpecKey):
        """Shadow-side twin of claim(): take ownership of the oldest un-adopted
        PEEK speculation for this key without removing it from the claim FIFO
        (the real run must still claim it later, in the same order)."""
        with self._lock:
            for spec in self._q.get(key, ()):
                if (
                    spec.source == "peek"
                    and not spec.adopted
                    and spec.state in ("pending", "running", "ready")
                ):
                    spec.adopted = True
                    return spec
            return None

    def unadopted_peeks(self, key: SpecKey) -> int:
        with self._lock:
            return sum(
                1
                for s in self._q.get(key, ())
                if s.source == "peek"
                and not s.adopted
                and s.state in ("pending", "running", "ready")
            )

    def evict_unadopted_peeks(self, key: SpecKey, keep: int, reason: str) -> int:
        """Retract peek bets: evict un-adopted peek speculations for this key
        beyond the first `keep` (oldest stay — adoption takes oldest first).
        Used when a newly streamed line invalidates a previous tail plan."""
        n = 0
        with self._lock:
            q = self._q.get(key)
            if not q:
                return 0
            seen = 0
            for spec in list(q):
                if (
                    spec.source == "peek"
                    and not spec.adopted
                    and spec.state in ("pending", "running", "ready")
                ):
                    seen += 1
                    if seen > keep:
                        spec.state = "evicted"
                        if spec.cancel:
                            try:
                                spec.cancel()
                            except Exception:
                                pass
                        q.remove(spec)
                        n += 1
                        self.bus.emit("evict", key=key, seq=spec.seq, reason=reason)
        return n

    def evict_tool(self, tool: str, reason: str) -> int:
        return self._evict(lambda s: s.key[0] == tool, reason)

    def evict_unclaimed(self, reason: str) -> int:
        return self._evict(lambda s: True, reason)

    def _evict(self, pred, reason: str) -> int:
        n = 0
        with self._lock:
            for q in self._q.values():
                for spec in list(q):
                    if pred(spec) and spec.state in ("pending", "running", "ready"):
                        spec.state = "evicted"
                        if spec.cancel:
                            try:
                                spec.cancel()
                            except Exception:
                                pass
                        q.remove(spec)
                        n += 1
                        self.bus.emit("evict", key=spec.key, seq=spec.seq, reason=reason)
        return n


class Budget:
    """Caps on speculative spend. max_inflight bounds concurrent executions;
    max_dispatches_per_turn bounds total dispatches (hard deny);
    max_inflight_chars bounds total argument characters being executed at
    once — the prefill-pressure control. Char-budget
    excess WAITS in its worker (delayed dispatch) rather than denying, so it
    can never abort the shadow or cause a miss."""

    def __init__(
        self,
        max_inflight: int = 16,
        max_dispatches_per_turn: int = 64,
        max_inflight_chars: int = 120_000,
    ):
        self.max_inflight = max_inflight
        self.max_dispatches_per_turn = max_dispatches_per_turn
        self.max_inflight_chars = max_inflight_chars
        self._dispatched = 0
        self._chars = 0
        self._lock = threading.Lock()
        self._chars_freed = threading.Condition(self._lock)

    def acquire_chars(self, n: int, timeout: float = 600.0) -> None:
        n = min(n, self.max_inflight_chars)  # one huge call may still run
        with self._chars_freed:
            deadline = time.monotonic() + timeout
            while self._chars + n > self.max_inflight_chars:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._chars_freed.wait(remaining):
                    break  # degrade: run anyway
            self._chars += n

    def release_chars(self, n: int) -> None:
        n = min(n, self.max_inflight_chars)
        with self._chars_freed:
            self._chars -= n
            self._chars_freed.notify_all()

    def admit(self) -> bool:
        with self._lock:
            if self._dispatched >= self.max_dispatches_per_turn:
                return False
            self._dispatched += 1
            return True

    def reset(self) -> None:
        with self._lock:
            self._dispatched = 0


class Launcher:
    def __init__(
        self, store: SpecStore, bus: EventBus = NULL_BUS, budget: Budget | None = None
    ) -> None:
        self.store = store
        self.bus = bus
        self.budget = budget or Budget()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=self.budget.max_inflight, thread_name_prefix="spec"
        )
        self._aloop: asyncio.AbstractEventLoop | None = None
        self._aloop_lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        """One shared background event loop for every async tool. Coroutines
        from different speculations run concurrently ON it, and tool clients
        that bind themselves to a loop (aiohttp, httpx.AsyncClient) stay
        valid across calls — which a per-call asyncio.run() would break."""
        with self._aloop_lock:
            if self._aloop is None:
                self._aloop = asyncio.new_event_loop()
                threading.Thread(
                    target=_serve_loop, args=(self._aloop,), daemon=True, name="spec-aio"
                ).start()
            return self._aloop

    def run_awaitable(self, aw: Any, timeout: float = 600.0) -> Any:
        """Drive an awaitable to completion from a worker thread."""
        fut = asyncio.run_coroutine_threadsafe(_as_coro(aw), self.loop())
        return fut.result(timeout)

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def dispatch_or_adopt(
        self, tool: ToolSpec, args: tuple, kwargs: dict, source: str
    ) -> Speculation | None:
        """Shadow hooks come through here: if a peek already fired this exact
        call, adopt the in-flight speculation instead of duplicating it."""
        key = spec_key(tool, args, kwargs)
        if tool.deterministic:
            spec = self.store.existing(key)
            if spec is not None:
                spec.adopted = True  # shield from peek retraction
                return spec
        spec = self.store.adopt(key)
        if spec is not None:
            self.bus.emit("adopt", key=key, seq=spec.seq, tool=tool.name)
            return spec
        return self.dispatch(tool, args, kwargs, source)

    def ensure_peeked(self, tool: ToolSpec, args: tuple, needed: int) -> int:
        """Top the store up to `needed` un-adopted peek speculations for this
        exact call (multiplicity-safe dedup across repeated peeks of a growing
        tail). Returns how many new dispatches were made."""
        key = spec_key(tool, args, {})
        if tool.deterministic:
            if self.store.existing(key):
                return 0
            needed = 1
        new = 0
        while self.store.unadopted_peeks(key) < needed:
            if self.dispatch(tool, args, {}, "peek") is None:
                break
            new += 1
        return new

    def dispatch(
        self, tool: ToolSpec, args: tuple, kwargs: dict, source: str
    ) -> Speculation | None:
        """Fire the tool now; returns the Speculation, or None if budget-denied."""
        if not self.budget.admit():
            self.bus.emit("note", msg=f"budget denied dispatch of {tool.name}")
            return None
        key = spec_key(tool, args, kwargs)
        spec = Speculation(
            key=key,
            seq=self.next_seq(),
            args=args,
            kwargs=kwargs,
            source=source,  # type: ignore[arg-type]
            args_preview=_preview(args, kwargs),
        )
        spec.t_dispatch = time.perf_counter()
        if tool.cancel_fn is not None:
            spec.cancel = lambda _t=tool, _s=spec: _t.cancel_fn(_s)
        self.store.put(spec)
        self.bus.emit(
            "dispatch",
            key=key,
            seq=spec.seq,
            tool=tool.name,
            source=source,
            preview=spec.args_preview,
        )

        def run() -> None:
            if spec.state == "evicted":
                spec.done.set()
                return
            spec.state = "running"
            nchars = sum(len(a) for a in args if isinstance(a, str))
            self.budget.acquire_chars(nchars)  # prefill-pressure gate (EXP-2)
            run_fn = tool.spec_fn or tool.fn  # speculative-mode variant
            try:
                if getattr(run_fn, "wants_spec", False):
                    out = run_fn(*args, _spec=spec, **kwargs)
                else:
                    out = run_fn(*args, **kwargs)
                # an async tool hands back a coroutine: drive it here, so the
                # speculation stores the VALUE. Storing the un-awaited
                # coroutine would make every claim hit return garbage while
                # the tool never ran at all.
                if inspect.isawaitable(out):
                    out = self.run_awaitable(out)
                spec.result = out
                if spec.state != "evicted":
                    spec.state = "ready"
            except BaseException as e:  # surfaced at claim/force point
                spec.error = e
                spec.state = "failed" if spec.state != "evicted" else "evicted"
            self.budget.release_chars(nchars)
            spec.t_ready = time.perf_counter()
            spec.done.set()
            if spec.state == "ready":
                self.bus.emit(
                    "ready", key=key, seq=spec.seq, ms=(spec.t_ready - spec.t_dispatch) * 1000
                )

        self._pool.submit(run)
        return spec

    def shutdown(self) -> None:
        with self._aloop_lock:
            loop, self._aloop = self._aloop, None
        if loop is not None:
            # cancel in-flight coroutines FIRST: a worker parked in
            # run_awaitable() unblocks only when its future resolves, and a
            # merely-stopped loop never resolves it — the pool's atexit join
            # would then hang for the whole timeout.
            asyncio.run_coroutine_threadsafe(_cancel_all_and_stop(), loop)
        self._pool.shutdown(wait=False, cancel_futures=True)


async def _as_coro(aw: Any) -> Any:
    return await aw


async def _cancel_all_and_stop() -> None:
    loop = asyncio.get_running_loop()
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


def _serve_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Own the loop end-to-end so it is CLOSED on its own thread — leaving it
    to __del__ raises 'Invalid file descriptor' noise at interpreter exit."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _preview(args: tuple, kwargs: dict, n: int = 60) -> str:
    s = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
    return s[:n] + ("…" if len(s) > n else "")


def make_shadow_hooks(reg: ToolRegistry, launcher: Launcher) -> dict[str, Callable]:
    hooks: dict[str, Callable] = {}
    for name in reg.names():
        tool = reg.get(name)
        assert tool is not None

        def hook(*args: Any, _tool=tool, **kwargs: Any):
            args = tuple(deep_force(a) for a in args)  # chained calls force here
            kwargs = {k: deep_force(v) for k, v in kwargs.items()}
            # the explicit separation: non-speculatable (or per-call gated-off)
            # tools do NOT run early — return an inert marker; the shadow
            # aborts only if a later statement actually uses it (tool_api).
            if not _tool.speculatable or (_tool.gate_fn and not _tool.gate_fn(args, kwargs)):
                return NonSpeculated(_tool.name)
            if contains_nonspec(args) or contains_nonspec(kwargs):
                raise RuntimeError(f"{_tool.name} args depend on a non-speculated result")
            if _tool.batched and args and isinstance(args[0], (list, tuple)):
                # llm_query_batched: dispatch per element so the real run can claim
                # elementwise; return list of SpecValues.
                prompts = list(args[0])
                specs = [
                    launcher.dispatch_or_adopt(
                        _single_of(_tool, reg), (p,) + args[1:], kwargs, "shadow"
                    )
                    for p in prompts
                ]
                return [SpecValue(s) if s else None for s in specs]
            spec = launcher.dispatch_or_adopt(_tool, args, kwargs, "shadow")
            if spec is None:  # budget denied
                raise ShadowBudgetDenied(_tool.name)
            return SpecValue(spec)

        hooks[name] = hook
    return hooks


def make_real_hooks(
    reg: ToolRegistry, store: SpecStore, launcher: Launcher, bus: EventBus = NULL_BUS
) -> dict[str, Callable]:
    hooks: dict[str, Callable] = {}
    for name in reg.names():
        tool = reg.get(name)
        assert tool is not None

        if tool.is_async:
            hooks[name] = _async_real_hook(tool, reg, store, bus)
            continue

        def hook(*args: Any, _tool=tool, **kwargs: Any):
            if not _tool.speculatable:
                return _tool.fn(*args, **kwargs)  # never speculated: passthrough
            if _tool.batched and args and isinstance(args[0], (list, tuple)):
                single = _single_of(_tool, reg)
                prompts = list(args[0])
                rest = tuple(args[1:])
                out: list[Any] = [None] * len(prompts)
                misses: list[int] = []
                claimed: list[tuple[int, Any]] = []
                for i, p in enumerate(prompts):
                    key = spec_key(single, (p,) + rest, kwargs)
                    spec = store.claim(key, reuse=single.deterministic)
                    if spec is None:
                        bus.emit("claim_miss", key=key, tool=single.name)
                        misses.append(i)
                    else:
                        bus.emit(
                            "claim_hit",
                            key=key,
                            seq=spec.seq,
                            tool=single.name,
                            already_ready=spec.done.is_set(),
                        )
                        claimed.append((i, spec))
                if misses:
                    batch_res = _tool.fn([prompts[i] for i in misses], *rest, **kwargs)
                    for j, i in enumerate(misses):
                        out[i] = batch_res[j]
                for i, spec in claimed:
                    out[i] = spec.wait(timeout=600)
                    spec.state = "claimed"
                    spec.t_claim = time.perf_counter()
                return out
            return _claim_or_run(_tool, tuple(args), kwargs, store, bus)

        hooks[name] = hook
    return hooks


def _async_real_hook(tool: ToolSpec, reg: ToolRegistry, store: SpecStore, bus: EventBus):
    """Claim-or-run for an `async def` tool. The hook is itself a coroutine
    function, so model code keeps its natural shape (`await llm(x)`,
    `asyncio.gather(...)`) — and a claim never blocks the caller's event loop,
    so sibling gather branches that MISSED still run concurrently."""

    async def hook(*args: Any, _tool=tool, **kwargs: Any):
        if not _tool.speculatable:
            return await _tool.fn(*args, **kwargs)
        if _tool.batched and args and isinstance(args[0], (list, tuple)):
            single = _single_of(_tool, reg)
            prompts, rest = list(args[0]), tuple(args[1:])
            out: list[Any] = [None] * len(prompts)
            misses, claimed = [], []
            for i, p in enumerate(prompts):
                key = spec_key(single, (p,) + rest, kwargs)
                spec = store.claim(key, reuse=single.deterministic)
                if spec is None:
                    bus.emit("claim_miss", key=key, tool=single.name)
                    misses.append(i)
                else:
                    bus.emit(
                        "claim_hit",
                        key=key,
                        seq=spec.seq,
                        tool=single.name,
                        already_ready=spec.done.is_set(),
                    )
                    claimed.append((i, spec))
            # the misses' batch call and every claim wait overlap
            async def _fill_misses():
                if not misses:
                    return
                res = await _tool.fn([prompts[i] for i in misses], *rest, **kwargs)
                for j, i in enumerate(misses):
                    out[i] = res[j]

            async def _fill_claim(i, spec):
                out[i] = await _await_spec(spec)
                spec.state = "claimed"
                spec.t_claim = time.perf_counter()

            await asyncio.gather(
                _fill_misses(), *[_fill_claim(i, s) for i, s in claimed]
            )
            return out
        key = spec_key(_tool, tuple(args), kwargs)
        t0 = time.perf_counter()
        spec = store.claim(key, reuse=_tool.deterministic)
        if spec is None:
            bus.emit("claim_miss", key=key, tool=_tool.name)
            return await _tool.fn(*args, **kwargs)  # miss: the baseline path
        bus.emit(
            "claim_hit", key=key, seq=spec.seq, tool=_tool.name, already_ready=spec.done.is_set()
        )
        result = await _await_spec(spec)
        spec.state = "claimed"
        spec.t_claim = time.perf_counter()
        bus.emit(
            "claim_done",
            key=key,
            seq=spec.seq,
            waited_ms=(time.perf_counter() - t0) * 1000,
            saved_ms=max(0.0, (t0 - spec.t_dispatch)) * 1000,
        )
        return result

    return hook


async def _await_spec(spec: Speculation, timeout: float = 600.0) -> Any:
    """Wait for an in-flight speculation without stalling the event loop."""
    if spec.done.is_set():
        return spec.wait(0)
    return await asyncio.to_thread(spec.wait, timeout)


def make_baseline_hooks(reg: ToolRegistry) -> dict[str, Callable]:
    hooks: dict[str, Callable] = {}
    for name in reg.names():
        tool = reg.get(name)
        assert tool is not None

        def hook(*args: Any, _tool=tool, **kwargs: Any):
            return _tool.fn(*args, **kwargs)  # async: returns the coroutine

        hooks[name] = hook
    return hooks


def _claim_or_run(tool, args: tuple, kwargs: dict, store: SpecStore, bus: EventBus):
    key = spec_key(tool, args, kwargs)
    t0 = time.perf_counter()
    spec = store.claim(key, reuse=tool.deterministic)
    if spec is not None:
        bus.emit(
            "claim_hit", key=key, seq=spec.seq, tool=tool.name, already_ready=spec.done.is_set()
        )
        result = spec.wait(timeout=600)  # waiting mode: block on in-flight call
        spec.state = "claimed"
        spec.t_claim = time.perf_counter()
        bus.emit(
            "claim_done",
            key=key,
            seq=spec.seq,
            waited_ms=(time.perf_counter() - t0) * 1000,
            saved_ms=max(0.0, (t0 - spec.t_dispatch)) * 1000,
        )
        return result
    bus.emit("claim_miss", key=key, tool=tool.name)
    return tool.fn(*args, **kwargs)  # miss: exactly the baseline path


def _single_of(batched_tool, reg: ToolRegistry):
    """The single-prompt ToolSpec that a batched tool decomposes into."""
    base = batched_tool.name.removesuffix("_batched")
    single = reg.get(base)
    if single is not None:
        return single
    return batched_tool


class ShadowBudgetDenied(Exception):
    pass
