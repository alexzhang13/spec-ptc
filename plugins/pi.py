"""Pi-mono plugin: wrap a tool executor with daemon-backed speculation."""

from plugins.client import SpecClient


class SpeculativeTools:
    """Wrap your existing tool table; misses fall through to your executor.

    tools = SpeculativeTools({"llm_query": my_llm}, socket="/tmp/spec-ptc.sock")
    tools.turn_begin({"context": doc})
    ... tools.feed(delta) per streamed token ...
    result = tools.call("llm_query", prompt)   # hit -> speculated, miss -> my_llm
    tools.turn_end()
    """

    def __init__(self, executors: dict, socket: str = "/tmp/spec-ptc.sock"):
        self.executors = executors
        self.spec = SpecClient(socket)

    def turn_begin(self, variables=None):
        self.spec.turn_begin(variables)

    def feed(self, delta: str):
        self.spec.feed(delta)

    def call(self, tool: str, *args):
        hit = self.spec.resolve(tool, list(args))
        return hit if hit is not None else self.executors[tool](*args)

    def turn_end(self):
        return self.spec.turn_end()
