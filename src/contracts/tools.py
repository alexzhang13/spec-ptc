"""Tools: explicit speculated/non-speculated split, registry, Tool interface."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spec_ptc.engine.speculation import Speculation

SpecKey = tuple[str, str]  # (tool_name, canonical args hash)


def canonical_hash(tool: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps([tool, args, kwargs], sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ToolSpec:
    """Registry record. `speculatable` is an explicit opt-in requiring pure=True;
    unmarked tools are invisible to the whole speculation machinery."""

    name: str
    fn: Callable[..., Any]  # the real implementation
    is_async: bool = False  # fn is a coroutine function: awaited, not called
    speculatable: bool = False
    pure: bool = False
    deterministic: bool = False
    latency_hint_ms: float = 1000.0
    batched: bool = False
    spec_fn: Callable[..., Any] | None = None  # how to RUN speculatively
    cancel_fn: Callable[..., None] | None = None  # abort in-flight on evict
    key_fn: Callable[..., Any] | None = None  # canonical claim identity
    gate_fn: Callable[..., bool] | None = None  # per-CALL speculatability


def spec_key(tool: ToolSpec, args: tuple, kwargs: dict) -> SpecKey:
    """Claim identity for one concrete call — BOTH dispatch and claim hash through here."""
    material = tool.key_fn(args, kwargs) if tool.key_fn else (args, kwargs)
    return (tool.name, canonical_hash(tool.name, (material,), {}))


class Tool:
    name: str = ""
    speculatable: bool = False
    pure: bool = False
    deterministic: bool = False
    latency_hint_ms: float = 1000.0
    batched: bool = False

    # -- required ---------------------------------------------------------
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # -- overridable speculation behavior ----------------------------------
    def speculative_execute(
        self, *args: Any, _spec: Speculation | None = None, **kwargs: Any
    ) -> Any:
        return self.execute(*args, **kwargs)

    def cancel(self, spec) -> None:
        pass

    def claim_key(self, args: tuple, kwargs: dict) -> Any:
        return (args, kwargs)

    def speculatable_call(self, args: tuple, kwargs: dict) -> bool:
        return True


async def _identity(v: Any) -> Any:
    """Await-protocol shim: completes immediately with `v`, never suspends."""
    return v


class NonSpeculated:
    """Inert marker a non-speculatable tool returns in the SHADOW. Storing it
    is fine; using it (any operation, or passing it into a speculatable
    call's arguments) raises, which opaque-aborts speculation at exactly the
    first statement that actually depended on the un-speculated result."""

    __slots__ = ("_tool",)

    def __init__(self, tool: str) -> None:
        object.__setattr__(self, "_tool", tool)

    def _boom(self):
        raise RuntimeError(
            f"result of non-speculatable tool {object.__getattribute__(self, '_tool')!r} "
            "used in shadow"
        )

    def __getattr__(self, k):
        self._boom()

    def __await__(self):
        # `x = await slow_tool()` in the shadow: stay a marker so taint flows
        # by value instead of aborting on an un-awaitable object.
        return _identity(self).__await__()

    def __str__(self):
        self._boom()

    def __format__(self, s):
        self._boom()

    def __bool__(self):
        self._boom()

    def __iter__(self):
        self._boom()

    def __getitem__(self, k):
        self._boom()

    def __add__(self, o):
        self._boom()

    def __radd__(self, o):
        self._boom()

    def __eq__(self, o):
        self._boom()

    def __hash__(self):
        self._boom()


def contains_nonspec(obj: Any, depth: int = 3) -> bool:
    """Deep check: is a NonSpeculated marker hiding in these args? (repr/hash
    of the marker raises, but json-default=repr canonicalization must never
    get that far — dispatching on marker args would be garbage.)"""
    if isinstance(obj, NonSpeculated):
        return True
    if depth <= 0:
        return False
    if isinstance(obj, (list, tuple, set)):
        return any(contains_nonspec(x, depth - 1) for x in obj)
    if isinstance(obj, dict):
        return any(contains_nonspec(v, depth - 1) for v in obj.values())
    return False


def _is_async_callable(fn: Any) -> bool:
    """True for `async def` tools (incl. partials, and objects whose
    __call__ is a coroutine function)."""
    if inspect.iscoroutinefunction(fn):
        return True
    # B004's callable() fix is wrong here: we test whether __call__ is async.
    call = getattr(type(fn), "__call__", None)  # noqa: B004
    return call is not None and inspect.iscoroutinefunction(call)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        speculatable: bool = False,
        pure: bool = False,
        deterministic: bool = False,
        latency_hint_ms: float = 1000.0,
        batched: bool = False,
        spec_fn: Callable[..., Any] | None = None,
        cancel_fn: Callable[..., None] | None = None,
        key_fn: Callable[..., Any] | None = None,
        gate_fn: Callable[..., bool] | None = None,
    ) -> ToolSpec:
        if speculatable and not pure:
            raise ValueError(
                f"tool {name!r}: speculatable=True requires pure=True — a tool "
                "with observable side effects must never execute early"
            )
        spec = ToolSpec(
            name=name,
            fn=fn,
            is_async=_is_async_callable(fn),
            speculatable=speculatable,
            pure=pure,
            deterministic=deterministic,
            latency_hint_ms=latency_hint_ms,
            batched=batched,
            spec_fn=spec_fn,
            cancel_fn=cancel_fn,
            key_fn=key_fn,
            gate_fn=gate_fn,
        )
        self._tools[name] = spec
        return spec

    def register_tool(self, tool: Any) -> ToolSpec:
        """Register a tool_api.Tool instance (the modular interface)."""

        def spec_fn(*a, _spec=None, _t=tool, **k):
            return _t.speculative_execute(*a, _spec=_spec, **k)

        spec_fn.wants_spec = True  # type: ignore[attr-defined]
        return self.register(
            tool.name,
            tool.execute,
            spec_fn=spec_fn,
            speculatable=tool.speculatable,
            pure=tool.pure,
            deterministic=tool.deterministic,
            latency_hint_ms=tool.latency_hint_ms,
            batched=tool.batched,
            cancel_fn=getattr(tool, "cancel", None),
            key_fn=getattr(tool, "claim_key", None),
            gate_fn=getattr(tool, "speculatable_call", None),
        )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def speculative_names(self) -> set[str]:
        return {n for n, t in self._tools.items() if t.speculatable}
