"""spec-ptc: speculative programmatic tool calling. Start with Speculator."""

from .contracts.events import EventBus
from .contracts.tools import Tool, ToolRegistry
from .runtime.harness import Harness
from .speculator import SpecSession, Speculator, StreamTurn

__all__ = [
    "EventBus",
    "Harness",
    "SpecSession",
    "Speculator",
    "StreamTurn",
    "Tool",
    "ToolRegistry",
]
