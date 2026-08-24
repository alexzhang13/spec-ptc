"""ShadowRunner: jailed fork of the REPL namespace, executing ahead of the model."""

from __future__ import annotations

import ast
import copy
import ctypes
import queue
import threading
import time
from collections import Counter

from spec_ptc.contracts.events import NULL_BUS, EventBus
from spec_ptc.contracts.tools import NonSpeculated, contains_nonspec
from spec_ptc.engine.speculation import (
    ForceTracker,
    SpecStore,
    SpecValue,
    force_ctx,
    force_in_place,
)
from spec_ptc.engine.streaming import Segment, plan_peeks

STMT_WALL_BUDGET_S = 2.0  # runaway guard: max wall time per shadow statement


class Opaque:
    """Marker for values that refused deepcopy; any use raises -> opaque-abort."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, k):
        raise RuntimeError(f"opaque value {self._name!r} touched in shadow")

    def __getitem__(self, k):
        raise RuntimeError(f"opaque value {self._name!r} touched in shadow")

    def __iter__(self):
        raise RuntimeError(f"opaque value {self._name!r} touched in shadow")


def snapshot_ns(ns: dict) -> dict:
    out = {}
    for k, v in ns.items():
        if k.startswith("__"):
            continue
        try:
            out[k] = copy.deepcopy(v)
        except Exception:
            # dict/list subclasses with un-copyable attrs (e.g. RLM's
            # _AnswerDict callback): a plain cast keeps the DATA and drops the
            # behavior — mutations stay inside the fork, callbacks never fire.
            if isinstance(v, dict):
                try:
                    out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items()}
                    continue
                except Exception:
                    pass
            if isinstance(v, (list, tuple)):
                try:
                    out[k] = type(v)(copy.deepcopy(x) for x in v)
                    continue
                except Exception:
                    pass
            out[k] = Opaque(k)
    return out


# Pure stdlib modules with no import-time side effects: model code imports
# these constantly (re, json, collections...) and blocking them silenced whole
# turns. random/os/time stay blocked (nondeterminism / effects).
_SHADOW_IMPORT_WHITELIST = {
    "re",
    "json",
    "math",
    "itertools",
    "collections",
    "functools",
    "operator",
    "statistics",
    "string",
    "textwrap",
    "heapq",
    "bisect",
    "difflib",
    "ast",
    "unicodedata",
    "fractions",
    "decimal",
    "copy",
    "typing",
    "dataclasses",
}


def _shadow_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _SHADOW_IMPORT_WHITELIST:
        return __import__(name, *args, **kwargs)
    raise RuntimeError(f"import {name!r} blocked in shadow (not in pure whitelist)")


_SHADOW_BLOCKED = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "exit",
    "quit",
    "help",
    "breakpoint",
}


def shadow_builtins(real_builtins: dict) -> dict:
    b = dict(real_builtins)
    for name in _SHADOW_BLOCKED:
        b[name] = _blocked(name)
    b["__import__"] = _shadow_import
    b["print"] = _shadow_print  # captured-and-discarded
    return b


def _shadow_print(*a, **k):
    pass


def _blocked(name: str):
    def fn(*a, **k):
        raise RuntimeError(f"{name}() blocked in shadow")

    return fn


class ShadowAborted(Exception):
    pass


class ShadowRunner:
    def __init__(
        self,
        real_locals: dict,
        shadow_hooks: dict,
        store: SpecStore,
        real_builtins: dict,
        bus: EventBus = NULL_BUS,
        extra_globals: dict | None = None,
        launcher=None,
        registry=None,
        taint_skip: bool = True,
    ) -> None:
        self.launcher = launcher  # needed only for peeks
        self.registry = registry
        self.taint_skip = taint_skip
        self.bus = bus
        self.store = store
        self.hooks = shadow_hooks
        self.ns: dict = dict(extra_globals or {})
        self.ns.update(snapshot_ns(real_locals))
        self.ns.update(shadow_hooks)
        self.ns["__builtins__"] = shadow_builtins(real_builtins)
        self._q: queue.Queue[Segment | None] = queue.Queue()
        self.aborted: str | None = None
        self.executed = 0
        self._last_peek_tally: dict = {}  # spec_key -> count from the last plan
        self._thread = threading.Thread(target=self._run, daemon=True, name="shadow")
        self._done = threading.Event()
        self._thread.start()

    # -- producer side --------------------------------------------------------
    def feed(self, seg: Segment) -> None:
        self._q.put(seg)

    def feed_peek(self, tail: str) -> None:
        """Queue a peek over the current unclosed tail. Runs on the shadow
        thread AFTER all fed statements, so the namespace it evaluates
        against is exactly the state those statements produced."""
        if self.launcher is not None:
            self._q.put(("peek", tail))

    def finish(self) -> None:
        self._q.put(None)

    def join(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    # -- worker ----------------------------------------------------------------
    def _run(self) -> None:
        self._tracker = ForceTracker()
        force_ctx.tracker = self._tracker
        try:
            while True:
                seg = self._q.get()
                if seg is None:
                    break
                if self.aborted:
                    continue
                if isinstance(seg, tuple) and seg[0] == "peek":
                    self._peek(seg[1])
                else:
                    self._exec_one(seg)
        finally:
            self._done.set()

    def _exec_one(self, seg: Segment) -> None:
        try:
            tree = ast.parse(seg.source)
        except SyntaxError:
            self._abort("syntax", seg)
            return
        # guard: rebinding a hooked name evicts + aborts
        for name in _bound_names(tree):
            if name in self.hooks:
                self.store.evict_tool(name, reason=f"rebound {name!r}")
                self._abort(f"rebind:{name}", seg)
                return
        reads = _read_names(tree)
        # a statement reading a NonSpeculated marker is skipped and its
        # targets poisoned — taint flows by value instead of killing the turn
        if self.taint_skip:
            tainted = sorted(n for n in reads if contains_nonspec(self.ns.get(n)))
            if tainted:
                for name in _bound_names(tree):
                    self.ns[name] = NonSpeculated("tainted:" + "+".join(tainted))
                self.bus.emit(
                    "shadow_skip",
                    block=seg.block_id,
                    index=seg.index,
                    tainted=tainted,
                    src=seg.source[:80],
                )
                return
        # force-sync: names this statement reads that hold SpecValues
        for name in reads:
            v = self.ns.get(name)
            try:
                if isinstance(v, SpecValue):
                    self.ns[name] = v._force()
                else:
                    force_in_place(v)
            except BaseException:
                self._abort("force-failed", seg)
                return
        self.bus.emit("shadow_exec", block=seg.block_id, index=seg.index, src=seg.source[:80])
        # Runaway guard: a watchdog raises ShadowAborted *in this thread* when
        # a statement's PURE COMPUTE exceeds its budget. Time spent blocked on
        # speculation waits (SpecValue forcing — e.g. a long per-item chain)
        # is metered by ForceTracker and does NOT count: only model-written
        # spinning (`while True:`) trips the guard. Async-exc lands at a
        # bytecode boundary, exactly right for pure-Python hot loops.
        armed = {"on": True}
        tid = threading.get_ident()
        t_stmt = time.perf_counter()
        forced_at_start = self._tracker.forced_seconds()

        def watchdog():
            if not armed["on"]:
                return
            pure = (time.perf_counter() - t_stmt) - (
                self._tracker.forced_seconds() - forced_at_start
            )
            if pure < STMT_WALL_BUDGET_S:  # mostly waiting: re-arm
                t2 = threading.Timer(max(0.1, STMT_WALL_BUDGET_S - pure), watchdog)
                t2.daemon = True
                timers.append(t2)
                t2.start()
                return
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(tid), ctypes.py_object(ShadowAborted)
            )

        timers = [threading.Timer(STMT_WALL_BUDGET_S, watchdog)]
        timers[0].daemon = True
        timers[0].start()
        try:
            exec(compile(tree, "<shadow>", "exec"), self.ns, self.ns)
            self.executed += 1
        except BaseException as e:
            self._abort(f"{type(e).__name__}: {e}", seg)
        finally:
            armed["on"] = False
            for tm in timers:
                tm.cancel()

    def abort(self, why: str = "external") -> None:
        self.aborted = why

    def _peek(self, tail: str) -> None:
        spec_names = self.registry.speculative_names() if self.registry else set(self.hooks)
        try:
            plans = plan_peeks(tail, spec_names, self.ns)
        except Exception:
            return
        # multiplicity-safe dedup: top each distinct call up to its tally
        from spec_ptc.contracts.tools import spec_key as _key

        tally = Counter((p.tool, p.args) for p in plans)
        new_tally: dict = {}
        for (tool_name, args), needed in tally.items():
            tool = self.registry.get(tool_name) if self.registry else None
            if tool is None or tool.batched or not tool.speculatable:
                continue
            if tool.gate_fn and not tool.gate_fn(args, {}):
                continue
            self.launcher.ensure_peeked(tool, args, needed)
            new_tally[_key(tool, args, {})] = needed
        # BET RETRACTION: a key the previous plan justified but this one
        # doesn't means new tokens invalidated the bet (e.g. a just-streamed
        # body line mutates the loop's iterable) — evict the stale peeks now,
        # while their sub-calls are barely started, instead of at turn end.
        for key, old_n in self._last_peek_tally.items():
            keep = new_tally.get(key, 0)
            if keep < old_n:
                self.store.evict_unadopted_peeks(key, keep, "peek-retracted")
        self._last_peek_tally = new_tally

    def _abort(self, why: str, seg: Segment) -> None:
        self.aborted = why
        self.bus.emit("shadow_stop", reason=why, block=seg.block_id, index=seg.index)


def _read_names(tree: ast.AST) -> set[str]:
    return {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _bound_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out
