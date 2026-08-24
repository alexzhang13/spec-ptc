"""Task loading + scoring for trec-coarse-132k and oolong-pairs-32k."""

import ast
import json
import random
import re
from dataclasses import dataclass

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from benchmark.oolong_campaign.hints import HINT_V8_PAIRS, HINT_V8_TREC

COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")


@dataclass
class Task:
    task_id: str
    dataset: str  # trec_coarse_131072 | oolong_pairs_32768
    context: str
    question: str
    hint: str
    answer: str
    answer_type: str


def load_tasks(n_trec: int = 4, n_pairs: int = 4, seed: int = 7) -> list[Task]:
    rng = random.Random(seed)
    val = load_dataset("oolongbench/oolong-synth", split="validation")
    trec = [r for r in val if r["dataset"] == "trec_coarse" and int(r["context_len"]) == 131072]
    rng.shuffle(trec)
    tasks = [
        Task(
            f"trec132k-{r['id']}",
            "trec_coarse_131072",
            r["context_window_text"],
            r["question"],
            HINT_V8_TREC,
            str(r["answer"]),
            str(r["answer_type"]),
        )
        for r in trec[:n_trec]
    ]

    ctx32 = next(
        r for r in val if r["dataset"] == "trec_coarse" and int(r["context_len"]) == 32768
    )["context_window_text"]
    p = hf_hub_download(
        "mit-oasys/oolong-pairs", "data/oolong-pairs-32768.json", repo_type="dataset"
    )
    pairs = json.load(open(p))
    rng.shuffle(pairs)
    tasks += [
        Task(
            f"pairs32k-{r['id']}",
            "oolong_pairs_32768",
            ctx32,
            r["question"],
            HINT_V8_PAIRS,
            str(r["answer"]),
            r["type"],
        )
        for r in pairs[:n_pairs]
    ]
    return tasks


# ---- scoring (trec scorer ported from rlm training env; pairs = set F1) ----
def _parse(answer: str) -> str:
    low = answer.lower()
    hits = [(low.rfind(ph), ph) for ph in COMPARISON_PHRASES if ph in low]
    if hits:
        return max(hits)[1]
    if ":" not in answer:
        return answer if len(answer) < 20 else answer.split()[-1]
    cand = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    return cand if len(cand) < 20 else answer


def score(task: Task, output: str) -> float:
    output = output or ""
    if task.dataset.startswith("oolong_pairs"):
        try:
            gold = set(ast.literal_eval(task.answer))
        except Exception:
            gold = set()
        got = set(f"({a}, {b})" for a, b in re.findall(r"\((\d+),\s*(\d+)\)", output))
        if not gold:
            return 0.0
        if not got:
            return 0.0
        tp = len(gold & got)
        prec, rec = tp / len(got), tp / len(gold)
        return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    try:
        gold = str(ast.literal_eval(task.answer)[0])
    except Exception:
        gold = task.answer
    trimmed = str(_parse(output))
    if trimmed.lower() == gold.lower():
        return 1.0
    if task.answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except Exception:
            return 0.0
    if gold and gold.lower() not in [p.lower() for p in COMPARISON_PHRASES]:
        if gold.lower() in output.lower():
            return 1.0
    return 0.0
