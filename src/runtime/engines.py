"""Engines: MockLM (deterministic, timed) and VLLMEngine (OpenAI-compatible streaming)."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from spec_ptc.contracts.events import NULL_BUS, EventBus
from spec_ptc.contracts.tools import ToolRegistry


# --------------------------------------------------------------------------- mock
@dataclass
class MockTiming:
    main_tok_per_s: float = 120.0  # coder model stream speed
    sub_base_s: float = 0.8  # sub-LM latency: base + jitter by prompt hash
    sub_jitter_s: float = 0.7
    sub_tokens: int = 24  # sub-LM streams this many tokens over its latency


class MockLM:
    """Deterministic content; latency simulated with interruptible sleeps.
    Identical prompts get distinct samples (per-prompt counter) — multiplicity."""

    def __init__(self, timing: MockTiming | None = None, bus: EventBus = NULL_BUS):
        self.timing = timing or MockTiming()
        self.bus = bus
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _latency(self, prompt: str) -> float:
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        return self.timing.sub_base_s + (h % 1000) / 1000.0 * self.timing.sub_jitter_s

    def _sample_id(self, prompt: str) -> int:
        with self._lock:
            n = self._counts.get(prompt, 0)
            self._counts[prompt] = n + 1
            return n

    def stream_main(self, script: str) -> Iterator[str]:
        """'Generate' a scripted response by streaming it token-ish chunks."""
        delay = 1.0 / self.timing.main_tok_per_s
        words = script.split(" ")
        for i, w in enumerate(words):
            time.sleep(delay)
            yield w + (" " if i < len(words) - 1 else "")

    def sub_call(self, prompt: str, _spec=None) -> str:
        lat = self._latency(prompt)
        sid = self._sample_id(prompt)
        n = self.timing.sub_tokens
        step = lat / n
        text = []
        h = hashlib.sha256(f"{prompt}|{sid}".encode()).hexdigest()[:8]
        for k in range(n):
            if _spec is not None and _spec.state == "evicted":
                raise RuntimeError("evicted")
            time.sleep(step)
            tok = f"w{k}"
            text.append(tok)
            if _spec is not None:
                self.bus.emit("subtoken", seq=_spec.seq, text=tok + " ")
        return f"[{h}#{sid}] " + _digest(prompt)

    def make_tools(self, reg: ToolRegistry, bus: EventBus) -> None:
        self.bus = bus

        def single(prompt, _spec=None):
            return self.sub_call(prompt, _spec=_spec)

        single.wants_spec = True  # type: ignore[attr-defined]
        reg.register("llm_query", single, speculatable=True, pure=True, latency_hint_ms=1200)

        def batched(prompts, **kw):
            with ThreadPoolExecutor(max_workers=8) as pool:
                return list(pool.map(lambda p: self.sub_call(p), prompts))

        reg.register(
            "llm_query_batched",
            batched,
            speculatable=True,
            pure=True,
            batched=True,
            latency_hint_ms=1200,
        )


def _digest(prompt: str) -> str:
    head = prompt.strip().replace("\n", " ")[:48]
    return f"mock-answer({head}…)"


# --------------------------------------------------------------------------- vllm
class VLLMEngine:
    """Two OpenAI-compatible endpoints: main (coder) + sub (llm_query)."""

    def __init__(
        self,
        main_base_url: str,
        main_model: str,
        sub_base_url: str,
        sub_model: str,
        bus: EventBus = NULL_BUS,
        sub_max_tokens: int = 256,
        main_max_tokens: int = 1200,
    ):
        from openai import OpenAI

        self.main = OpenAI(base_url=main_base_url, api_key="EMPTY")
        self.sub = OpenAI(base_url=sub_base_url, api_key="EMPTY")
        self.main_model = main_model
        self.sub_model = sub_model
        self.bus = bus
        self.sub_max_tokens = sub_max_tokens
        self.main_max_tokens = main_max_tokens

    def stream_main(self, messages) -> Iterator[str]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = self.main.chat.completions.create(
            model=self.main_model,
            messages=messages,
            stream=True,
            max_tokens=self.main_max_tokens,
            temperature=0.2,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def sub_call(self, prompt: str, _spec=None) -> str:
        resp = self.sub.chat.completions.create(
            model=self.sub_model,
            messages=[{"role": "user", "content": str(prompt)}],
            stream=True,
            max_tokens=self.sub_max_tokens,
            temperature=0.7,
        )
        parts: list[str] = []
        for chunk in resp:
            if _spec is not None and _spec.state == "evicted":
                resp.close()
                raise RuntimeError("evicted")
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
                if _spec is not None:
                    self.bus.emit("subtoken", seq=_spec.seq, text=delta)
        return "".join(parts)

    def make_tools(self, reg: ToolRegistry, bus: EventBus) -> None:
        self.bus = bus

        def single(prompt, _spec=None):
            return self.sub_call(prompt, _spec=_spec)

        single.wants_spec = True  # type: ignore[attr-defined]
        reg.register("llm_query", single, speculatable=True, pure=True, latency_hint_ms=2500)

        def batched(prompts, **kw):
            with ThreadPoolExecutor(max_workers=8) as pool:
                return list(pool.map(lambda p: self.sub_call(p), prompts))

        reg.register(
            "llm_query_batched",
            batched,
            speculatable=True,
            pure=True,
            batched=True,
            latency_hint_ms=2500,
        )


# --------------------------------------------------------------- wiring
def engine_from_env(path: str = ".endpoints.env", bus=NULL_BUS) -> VLLMEngine:
    env: dict[str, str] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

    def get(k, d=None):
        return os.environ.get(k, env.get(k, d))

    return VLLMEngine(
        main_base_url=get("SPEC_MAIN_URL", "http://localhost:8100/v1"),
        main_model=get("SPEC_MAIN_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        sub_base_url=get("SPEC_SUB_URL", "http://localhost:8101/v1"),
        sub_model=get("SPEC_SUB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        bus=bus,
    )
