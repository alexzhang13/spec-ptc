"""Python driver for the Bun REPL: same store/launcher, JS execution."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from spec_ptc.contracts.events import NULL_BUS, EventBus
from spec_ptc.contracts.tools import ToolRegistry, spec_key
from spec_ptc.engine.speculation import Launcher, SpecStore

_REPL_JS = str(Path(__file__).parent / "repl.js")


class _BunProc:
    def __init__(self) -> None:
        self.p = subprocess.Popen(
            ["bun", "run", _REPL_JS],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()

    def send(self, obj: dict) -> None:
        with self._lock:
            assert self.p.stdin
            self.p.stdin.write(json.dumps(obj) + "\n")
            self.p.stdin.flush()

    def recv(self) -> dict:
        assert self.p.stdout
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("bun process died")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.p.terminate()
        except Exception:
            pass


class BunREPL:
    """Persistent-namespace JS REPL with speculation (real + shadow twins)."""

    def __init__(
        self, reg: ToolRegistry, store: SpecStore, launcher: Launcher, bus: EventBus = NULL_BUS
    ) -> None:
        self.reg, self.store, self.launcher, self.bus = reg, store, launcher, bus
        self.real = _BunProc()
        self.shadow = _BunProc()
        self.committed: list[str] = []  # statement log for twin rebuilds
        self._exec_id = 0
        self._dispatch_by_id: dict[int, object] = {}

    # ---------------- real side (waiting mode) ------------------------------
    def execute_code(self, code: str) -> dict:
        self._exec_id += 1
        self.real.send({"op": "exec", "id": self._exec_id, "code": code, "mode": "real"})
        while True:
            m = self.real.recv()
            if m["op"] == "exec_done":
                if not m.get("error"):
                    self.committed.append(code)
                return m
            if m["op"] == "tool_call":
                tool = self.reg.get(m["tool"])
                args = tuple(m["args"])
                if not tool.speculatable:
                    result = tool.fn(*args)
                    self.real.send({"op": "reply", "id": m["id"], "data": {"result": result}})
                    continue
                key = spec_key(tool, args, {})
                spec = self.store.claim(key)
                if spec is not None:
                    self.bus.emit("claim_hit", key=key, seq=spec.seq, tool=tool.name)
                    result = spec.wait(600)
                    spec.state = "claimed"
                else:
                    self.bus.emit("claim_miss", key=key, tool=tool.name)
                    result = tool.fn(*args)
                self.real.send({"op": "reply", "id": m["id"], "data": {"result": result}})

    # ---------------- shadow side (speculative mode) -------------------------
    def shadow_feed(self, statement: str) -> None:
        """Execute one closed statement in the shadow twin (dispatches tools)."""
        self._exec_id += 1
        self.shadow.send(
            {"op": "exec", "id": self._exec_id, "code": statement, "mode": "shadow"}
        )
        while True:
            m = self.shadow.recv()
            if m["op"] == "exec_done":
                return
            if m["op"] == "tool_dispatch":
                tool = self.reg.get(m["tool"])
                spec = self.launcher.dispatch(tool, tuple(m["args"]), {}, "shadow")
                # the reply's id doubles as the lazyProxy's force handle
                self._dispatch_by_id[m["id"]] = spec
                self.shadow.send({"op": "reply", "id": m["id"], "data": {"id": m["id"]}})
            elif m["op"] == "tool_force":
                spec = self._dispatch_by_id.get(m["id"])
                result = spec.wait(600) if spec else None
                self.shadow.send({"op": "reply", "id": m["id"], "data": {"result": result}})

    def close(self) -> None:
        self.real.close()
        self.shadow.close()
