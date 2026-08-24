"""Async tools: coroutine tools are awaited (not stored un-awaited), the real
hook stays awaitable, and awaiting a SpecValue does not force it."""

import asyncio
import time

import pytest

from spec_ptc import Speculator
from spec_ptc.contracts.events import EventBus
from spec_ptc.engine.speculation import SpecValue

SUB_S = 0.25  # per-call latency: long enough that overlap is unambiguous


def block(code):
    return f"here goes\n```repl\n{code}\n```\ndone"


class Fixture:
    """One Speculator + async tools + a fed turn, then the real exec."""

    def __init__(self, **tool_kwargs):
        self.bus = EventBus()
        self.spec = Speculator(bus=self.bus)
        self.calls: list[str] = []
        self.effects: list[str] = []

        @self.spec.tool(speculatable=True, pure=True, **tool_kwargs)
        async def allm(p: str) -> str:
            self.calls.append(p)
            await asyncio.sleep(SUB_S)
            return f"A[{p}]"

        @self.spec.tool(speculatable=True, pure=True, batched=True)
        async def allm_batched(ps: list) -> list:
            self.calls.extend(ps)
            await asyncio.sleep(SUB_S)
            return [f"A[{p}]" for p in ps]

        @self.spec.tool()  # side effects: never speculated
        async def asend(p: str) -> str:
            self.effects.append(p)
            await asyncio.sleep(0.01)
            return f"sent:{p}"

        @self.spec.tool(speculatable=True, pure=True)
        def sllm(p: str) -> str:  # sync tool: regression guard
            self.calls.append(p)
            time.sleep(SUB_S)
            return f"S[{p}]"

        self.ns: dict = {}
        self.ns.update(self.spec.hooks())

    def run(self, code, peek=True):
        """Stream `code` through a turn, then really execute it. Returns
        (namespace, wall seconds for stream+exec)."""
        text = block(code)
        t0 = time.perf_counter()
        with self.spec.turn(repl_locals=self.ns, peek=peek) as t:
            for ch in text:
                t.feed(ch)
            self.shadow_stops = [e for e in self.bus.history if e.kind == "shadow_stop"]
        exec(compile(code, "<real>", "exec"), self.ns)
        return self.ns, time.perf_counter() - t0

    def counts(self):
        k = lambda kind: sum(1 for e in self.bus.history if e.kind == kind)
        return k("claim_hit"), k("claim_miss"), k("dispatch")

    def close(self):
        self.spec.close()


@pytest.fixture
def fx():
    f = Fixture()
    yield f
    f.close()


RUN_TWO = (
    "import asyncio\n"
    "async def main():\n"
    "    a = await allm('one')\n"
    "    b = await allm('two')\n"
    "    return str(a) + '|' + str(b)\n"
    "out = asyncio.run(main())"
)


# ------------------------------------------------------------------ regression
def test_speculated_async_tool_returns_value_not_coroutine(fx):
    """The bug this support fixes: the launcher used to store the un-awaited
    coroutine, so a claim hit handed the REPL a coroutine object while the
    tool body never ran at all."""
    text = block("x = allm('one')")
    with fx.spec.turn(repl_locals=fx.ns) as t:
        for ch in text:
            t.feed(ch)
    specs = fx.spec.session.store.all
    assert specs, "nothing was speculated"
    done = [s for s in specs if s.done.wait(5)]
    assert done and all(not asyncio.iscoroutine(s.result) for s in done)
    assert all(s.result == "A[one]" for s in done if s.error is None)
    assert fx.calls == ["one"], "the async tool body must actually execute"


def test_async_tool_end_to_end(fx):
    ns, _ = fx.run(RUN_TWO)
    hits, misses, _ = fx.counts()
    assert ns["out"] == "A[one]|A[two]"
    assert (hits, misses) == (2, 0)
    assert sorted(fx.calls) == ["one", "two"]


def test_serial_awaits_overlap(fx):
    """Two sequentially awaited calls in the real run are served by
    speculations that already ran concurrently."""
    _, wall = fx.run(RUN_TWO)
    assert wall < 2 * SUB_S, f"serial awaits did not overlap ({wall:.2f}s)"


def test_gather_claims_every_branch(fx):
    code = (
        "import asyncio\n"
        "async def main():\n"
        "    rs = await asyncio.gather(allm('x'), allm('y'), allm('z'))\n"
        "    return '|'.join(str(r) for r in rs)\n"
        "out = asyncio.run(main())"
    )
    ns, wall = fx.run(code)
    hits, misses, _ = fx.counts()
    assert ns["out"] == "A[x]|A[y]|A[z]"  # gather order preserved
    assert (hits, misses) == (3, 0)
    assert wall < 3 * SUB_S


# ------------------------------------------------------------------ semantics
def test_await_does_not_force_specvalue():
    """Awaiting must bind the lazy proxy, exactly like the sync form — if
    await were the force point, every async call would block at its own
    await instead of at first use."""
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    async def allm(p: str) -> str:
        await asyncio.sleep(5)  # never completes within the test
        return p

    hooks = spec.session.launcher
    sp = hooks.dispatch(spec.registry.get("allm"), ("slow",), {}, "shadow")
    sv = SpecValue(sp)

    async def go():
        return await sv

    got = asyncio.run(go())
    assert got is sv, "await forced the proxy instead of staying lazy"
    spec.close()


def test_non_speculatable_async_tool_never_runs_early(fx):
    code = (
        "import asyncio\n"
        "async def main():\n"
        "    a = await allm('one')\n"
        "    s = await asend('report')\n"
        "    return str(a) + '/' + str(s)\n"
        "out = asyncio.run(main())"
    )
    ns, _ = fx.run(code)
    assert ns["out"] == "A[one]/sent:report"
    assert fx.effects == ["report"], "side-effecting async tool ran more than once"


def test_gather_mixed_hit_and_miss_overlap(fx):
    """A claimed branch must not stall the event loop: the sibling branch
    that missed has to keep making progress while the claim resolves."""
    fx.run("out = None")  # warm nothing; separate turn below
    code = (
        "import asyncio\n"
        "async def main():\n"
        "    return await asyncio.gather(allm('hit'), allm('miss'))\n"
        "out = asyncio.run(main())"
    )
    # speculate only 'hit' by feeding a turn that mentions just that call
    text = block("x = allm('hit')")
    with fx.spec.turn(repl_locals=fx.ns) as t:
        for ch in text:
            t.feed(ch)
    t0 = time.perf_counter()
    exec(compile(code, "<real>", "exec"), fx.ns)
    wall = time.perf_counter() - t0
    assert [str(v) for v in fx.ns["out"]] == ["A[hit]", "A[miss]"]
    assert wall < 2 * SUB_S, f"claimed branch blocked the loop ({wall:.2f}s)"


def test_async_error_surfaces_at_claim():
    spec = Speculator()

    @spec.tool(speculatable=True, pure=True)
    async def afail(p: str) -> str:
        await asyncio.sleep(0.01)
        raise ValueError("boom " + p)

    ns: dict = {}
    ns.update(spec.hooks())
    code = (
        "import asyncio\n"
        "async def main():\n"
        "    return await afail('x')\n"
        "out = asyncio.run(main())"
    )
    with spec.turn(repl_locals=ns) as t:
        for ch in block(code):
            t.feed(ch)
    with pytest.raises(ValueError, match="boom x"):
        exec(compile(code, "<real>", "exec"), ns)
    spec.close()


def test_async_batched_claims_elementwise(fx):
    code = (
        "import asyncio\n"
        "async def main():\n"
        "    rs = await allm_batched(['p', 'q'])\n"
        "    return '|'.join(str(r) for r in rs)\n"
        "out = asyncio.run(main())"
    )
    ns, _ = fx.run(code)
    hits, misses, _ = fx.counts()
    assert ns["out"] == "A[p]|A[q]"
    assert hits == 2 and misses == 0


def test_deterministic_async_tool_runs_once():
    spec = Speculator()
    calls = []

    @spec.tool(speculatable=True, pure=True, deterministic=True)
    async def acache(p: str) -> str:
        calls.append(p)
        await asyncio.sleep(0.02)
        return "C[" + p + "]"

    ns: dict = {}
    ns.update(spec.hooks())
    code = (
        "import asyncio\n"
        "async def main():\n"
        "    a = await acache('k')\n"
        "    b = await acache('k')\n"
        "    return str(a) + str(b)\n"
        "out = asyncio.run(main())"
    )
    with spec.turn(repl_locals=ns) as t:
        for ch in block(code):
            t.feed(ch)
    exec(compile(code, "<real>", "exec"), ns)
    assert ns["out"] == "C[k]C[k]"
    assert calls == ["k"], "deterministic async tool ran more than once"
    spec.close()


def test_sync_tools_unaffected(fx):
    code = "a = sllm('one')\nb = sllm('two')\nout = str(a) + '|' + str(b)"
    ns, wall = fx.run(code)
    hits, misses, _ = fx.counts()
    assert ns["out"] == "S[one]|S[two]"
    assert (hits, misses) == (2, 0)
    assert wall < 2 * SUB_S


# ------------------------------------------------------------------ shadow
def test_shadow_can_import_asyncio(fx):
    """Model code needs asyncio.run/gather to reach the hooks at all; a
    blocked import silently killed the whole turn's speculation."""
    ns, _ = fx.run(RUN_TWO)
    assert not fx.shadow_stops, [e.data for e in fx.shadow_stops]


def test_top_level_await_degrades_safely(fx):
    """`exec` cannot run top-level await, so the shadow can't either. It must
    abort cleanly (no speculation) rather than crash the turn."""
    text = block("a = await allm('one')")
    with fx.spec.turn(repl_locals=fx.ns) as t:
        for ch in text:
            t.feed(ch)
    stops = [e for e in fx.bus.history if e.kind == "shadow_stop"]
    assert stops and "await" in str(stops[0].data.get("reason", ""))
    assert fx.counts()[0] == 0  # no bogus hits
