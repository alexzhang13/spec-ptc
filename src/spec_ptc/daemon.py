"""Out-of-process speculation server: 4 JSON-lines ops over a unix socket
(turn_begin / feed / resolve / turn_end). Client: plugins/client.py."""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading

from spec_ptc.contracts.tools import ToolRegistry, spec_key
from spec_ptc.runtime.harness import REAL_BUILTINS
from spec_ptc.speculator import SpecSession, StreamTurn


class DaemonState:
    def __init__(self, engine) -> None:
        self.reg = ToolRegistry()
        self.session_bus_lock = threading.Lock()
        self.engine = engine
        self.session = None
        self._reset_session()

    def _reset_session(self):
        from spec_ptc.contracts.events import EventBus

        bus = EventBus()
        self.reg = ToolRegistry()
        self.engine.make_tools(self.reg, bus)
        self.session = SpecSession(self.reg, bus)


class Handler(socketserver.StreamRequestHandler):
    state: DaemonState  # injected by serve()

    def handle(self) -> None:
        turn: StreamTurn | None = None
        for raw in self.rfile:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                self._send({"ok": False, "error": "bad json"})
                continue
            op = msg.get("op")
            st = self.state
            if op == "turn_begin":
                turn = st.session.begin_stream_turn(msg.get("vars") or {}, REAL_BUILTINS)
                self._send({"ok": True})
            elif op == "feed":
                if turn is not None:
                    turn.feed(msg.get("delta", ""))
                self._send({"ok": True})
            elif op == "resolve":
                tool = st.reg.get(msg["tool"])
                if tool is None:
                    self._send({"status": "miss"})
                    continue
                args = tuple(msg.get("args", []))
                if not tool.speculatable:
                    self._send({"status": "miss"})
                    continue
                key = spec_key(tool, args, {})
                spec = st.session.store.claim(key)
                if spec is None:
                    self._send({"status": "miss"})
                else:
                    import time

                    t0 = time.perf_counter()
                    result = spec.wait(600)
                    spec.state = "claimed"
                    self._send(
                        {
                            "status": "hit",
                            "result": result,
                            "waited_ms": (time.perf_counter() - t0) * 1000,
                        }
                    )
            elif op == "turn_end":
                if turn is not None:
                    turn.end(timeout=5)
                    turn = None
                store = st.session.store
                metrics = {
                    "speculated": len(store.all),
                    "claimed": sum(1 for s in store.all if s.state == "claimed"),
                    "evicted": sum(1 for s in store.all if s.state == "evicted"),
                }
                st.session.end_turn()
                self._send({"ok": True, "metrics": metrics})
            else:
                self._send({"ok": False, "error": f"unknown op {op!r}"})

    def _send(self, obj: dict) -> None:
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()


def serve(socket_path: str, engine) -> socketserver.ThreadingUnixStreamServer:
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    state = DaemonState(engine)

    class BoundHandler(Handler):
        pass

    BoundHandler.state = state
    srv = socketserver.ThreadingUnixStreamServer(socket_path, BoundHandler)
    srv.daemon_threads = True
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/spec-ptc.sock")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.mock:
        from spec_ptc.runtime.engines import MockLM

        engine = MockLM()
    else:
        from spec_ptc.runtime.engines import engine_from_env

        engine = engine_from_env()
    srv = serve(args.socket, engine)
    print(f"spec-ptc daemon on {args.socket}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
