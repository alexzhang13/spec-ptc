"""OOLONG-style synthetic long-context tasks."""

from __future__ import annotations

import random

USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]
MOODS = ["happy", "sad", "angry", "calm", "anxious", "excited"]
TOPICS = [
    "training run",
    "gpu cluster",
    "paper draft",
    "code review",
    "benchmark",
    "dataset",
    "meeting",
    "deploy",
]


def make_log_context(n_entries: int = 240, seed: int = 7) -> str:
    rng = random.Random(seed)
    lines = []
    for i in range(n_entries):
        day = 1 + (i * 3) % 28
        month = 1 + (i // 40) % 12
        u = rng.choice(USERS)
        m = rng.choice(MOODS)
        t = rng.choice(TOPICS)
        lines.append(
            f"2025-{month:02d}-{day:02d} | user={u} | mood={m} | "
            f"note=worked on the {t}, felt {m} about progress (entry {i})."
        )
    return "\n".join(lines)


def make_report_context(n_sections: int = 12, seed: int = 11) -> str:
    rng = random.Random(seed)
    sections = []
    for i in range(n_sections):
        t = rng.choice(TOPICS)
        body = " ".join(
            rng.choice(
                [
                    "The",
                    "results",
                    "show",
                    "steady",
                    "gains",
                    "on",
                    t,
                    "with",
                    "regressions",
                    "in",
                    "latency",
                    "under",
                    "load",
                    "and",
                    "notable",
                    "improvements",
                    "in",
                    "throughput",
                    "measured",
                    "overnight",
                ]
            )
            for _ in range(80)
        )
        sections.append(f"## Section {i}: {t}\n{body}")
    return "\n\n".join(sections)
