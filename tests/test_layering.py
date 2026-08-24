"""Core (src/) never imports its consumers (demo/, plugins/, benchmark/)."""

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORE = ROOT / "src" / "spec_ptc"
FORBIDDEN = ("demo", "plugins", "benchmark")


def _imports(path):
    for f in path.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    yield f, a.name
            elif isinstance(node, ast.ImportFrom):
                yield f, ("." * node.level) + (node.module or "")


def test_core_is_standalone():
    bad = [
        f"{f.name}: {m}"
        for f, m in _imports(CORE)
        if any(m == b or m.startswith(b + ".") for b in FORBIDDEN)
    ]
    assert not bad, bad


def test_demo_and_plugins_never_import_benchmark():
    for layer in ("demo", "plugins"):
        bad = [
            f"{f}: {m}"
            for f, m in _imports(ROOT / layer)
            if m == "benchmark" or m.startswith("benchmark.")
        ]
        assert not bad, bad
