"""JSONL trajectory + stats logging (all campaign logging lives here)."""

import json
import threading
import time
from pathlib import Path


class TrajectoryLogger:
    """One events.jsonl per unit run; every event carries epoch + relative time."""

    def __init__(self, outdir: Path):
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.outdir / "events.jsonl", "a", buffering=1)
        self._lock = threading.Lock()
        self.t0 = time.time()

    def log(self, kind: str, **data) -> None:
        rec = {"t": round(time.time(), 4), "rel_t": round(time.time() - self.t0, 4),
               "kind": kind, **data}
        with self._lock:
            self._f.write(json.dumps(rec, default=str) + "\n")

    def write_json(self, name: str, obj) -> None:
        (self.outdir / name).write_text(json.dumps(obj, indent=1, default=str))

    def close(self) -> None:
        self._f.close()
