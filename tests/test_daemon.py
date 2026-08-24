"""A 'foreign harness' (the vendorable client) against the daemon."""

import threading
import time

from plugins.client import SpecClient
from spec_ptc.daemon import serve
from spec_ptc.runtime.engines import MockLM, MockTiming

SCRIPT = (
    "thinking...\n```repl\n"
    "parts = []\n"
    "for c in ['aa', 'bb', 'cc']:\n"
    "    parts.append(llm_query('x: ' + c))\n"
    "\n"
    "done = '|'.join(str(p) for p in parts)\n"
    "```\n"
)


def test_daemon_hit_and_miss(tmp_path):
    sock = str(tmp_path / "d.sock")
    eng = MockLM(MockTiming(sub_base_s=0.3, sub_jitter_s=0.1, sub_tokens=2))
    srv = serve(sock, eng)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = SpecClient(sock)
        c.turn_begin({})
        for i in range(0, len(SCRIPT), 8):
            c.feed(SCRIPT[i : i + 8])
        time.sleep(0.6)  # generation "continues"; speculations complete

        # host executes its code tool; its llm_query wrapper resolves first
        t0 = time.perf_counter()
        for chunk in ("aa", "bb", "cc"):
            r = c.resolve("llm_query", ["x: " + chunk])
            assert r is not None and "mock-answer" in r
        assert time.perf_counter() - t0 < 0.3, "hits should be near-instant"

        t1 = time.perf_counter()
        assert c.resolve("llm_query", ["never speculated"]) is None
        assert time.perf_counter() - t1 < 0.05, "miss must be near-free"

        m = c.turn_end()
        assert m["claimed"] == 3
        c.close()
    finally:
        srv.shutdown()
