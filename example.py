"""Minimal RLM + spec-ptc example: one OOLONG-pairs 32k question."""

import json
import os

import rlm.environments as envs
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from rlm import RLM

from demo.rlm import SpeculativeLocalREPL

envs.local_repl.LocalREPL = SpeculativeLocalREPL
envs.LocalREPL = SpeculativeLocalREPL

pairs = json.load(open(hf_hub_download(
    "mit-oasys/oolong-pairs", "data/oolong-pairs-32768.json", repo_type="dataset",
)))
context = next(
    r["context_window_text"]
    for r in load_dataset("oolongbench/oolong-synth", split="validation")
    if r["dataset"] == "trec_coarse" and int(r["context_len"]) == 32768
)

r = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": os.environ.get("SPEC_MAIN_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        "base_url": os.environ.get("SPEC_MAIN_URL", "http://localhost:8100/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
    },
    environment="local",
)
print(r.completion(prompt=context, root_prompt=pairs[0]["question"]).response)
