"""Vendorable spec-ptc client — stdlib only, ~60 lines. Copy this file into a."""

from __future__ import annotations

import json
import socket


class SpecClient:
    def __init__(self, socket_path: str = "/tmp/spec-ptc.sock", timeout: float = 600.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(socket_path)
        self.sock.settimeout(timeout)
        self._buf = b""

    def _rpc(self, msg: dict) -> dict:
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("spec-ptc daemon closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def turn_begin(self, variables: dict | None = None) -> None:
        self._rpc({"op": "turn_begin", "vars": variables or {}})

    def feed(self, delta: str) -> None:
        self._rpc({"op": "feed", "delta": delta})

    def resolve(self, tool: str, args: list):
        """The claimed result, or None (miss -> run the tool yourself)."""
        r = self._rpc({"op": "resolve", "tool": tool, "args": args})
        return r["result"] if r.get("status") == "hit" else None

    def turn_end(self) -> dict:
        return self._rpc({"op": "turn_end"}).get("metrics", {})

    def close(self) -> None:
        self.sock.close()
