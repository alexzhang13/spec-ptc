"""Robustness over many AST shapes: (1) the segmenter must recombine any
streamed block into the exact original AST, at any chunking; (2) spec mode
must produce baseline-identical answers on tricky code shapes."""

import ast
import random
import re

import pytest

from spec_ptc.engine.streaming import StreamSegmenter
from spec_ptc.runtime.engines import MockLM, MockTiming
from spec_ptc.runtime.harness import Harness

SNIPPETS = [
    "x = (1 +\n     2) * 3",
    "s = '''tri\nple'''\nt = f\"nested {s!r:>10}\"",
    "@staticmethod\ndef deco():\n    return 1",
    "def f(a, b=2, *args, **kw):\n    def g():\n        return a\n    return g()",
    "class C:\n    x = 1\n    def m(self):\n        return self.x",
    "try:\n    v = 1 / 1\nexcept ZeroDivisionError:\n    v = 0\nfinally:\n    w = 2",
    "match [1, 2]:\n    case [a, b]:\n        m = a + b\n    case _:\n        m = 0",
    "n = 0\nwhile n < 3:\n    n += 1\nelse:\n    n = -n",
    "if (y := 10) > 5:\n    z = y\nelif y:\n    z = 0\nelse:\n    z = -1",
    "a, (b, c) = 1, (2, 3)\nd = [*range(3), a]",
    "async def afn():\n    return 1",
    "data = {'k': [i for i in range(5) if i % 2],\n        'j': (1,\n              2)}",
    "lam = lambda q: q * 2\nout = lam(21)",
    "for i in range(3):\n    if i == 1:\n        continue\n    for j in range(2):\n        pass\n",
    "x = 1;  y = 2",
    "very_long = 'aaa' 'bbb' \\\n    'ccc'",
    "def gen():\n    yield from range(3)\n\nvals = list(gen())",
    "with open('/dev/null') as fh:\n    _ = fh.read(0)",
]


@pytest.mark.parametrize("seed", range(12))
def test_segmenter_recombines_exactly(seed):
    rng = random.Random(seed)
    body = "\n".join(rng.sample(SNIPPETS, k=rng.randint(3, 8)))
    text = f"prose before\n```repl\n{body}\n```\nprose after"
    seg = StreamSegmenter()
    out = []
    i = 0
    while i < len(text):
        n = rng.randint(1, 17)
        out += seg.feed(text[i : i + n])
        i += n
    out += seg.finish()
    recombined = "\n".join(s.source for s in out)
    assert ast.dump(ast.parse(recombined)) == ast.dump(ast.parse(body))


TRICKY_EXEC = [
    # walrus condition + call
    "if (t := 'go') == 'go':\n    r = llm_query('walrus ' + t)\n\n"
    "answer['content'] = str(r)\nanswer['ready'] = True",
    # nested function defining + calling hook
    "def ask(q):\n    return llm_query('fn: ' + q)\n\n"
    "rs = [ask('one'), ask('two')]\n"
    "answer['content'] = '|'.join(str(x) for x in rs)\nanswer['ready'] = True",
    # dict/set comprehension args
    "qs = {k: llm_query('d: ' + k) for k in ['p', 'q']}\n"
    "answer['content'] = str(qs['p']) + str(qs['q'])\nanswer['ready'] = True",
    # while with break, call after
    "i = 0\nwhile True:\n    i += 1\n    if i > 2:\n        break\n\n"
    "r = llm_query('after while ' + str(i))\n"
    "answer['content'] = str(r)\nanswer['ready'] = True",
    # exception swallowed, call in except
    "try:\n    raise ValueError('x')\nexcept ValueError:\n    r = llm_query('in except')\n\n"
    "answer['content'] = str(r)\nanswer['ready'] = True",
    # starred call args + conditional expression
    "parts = ['a', 'b']\nr = llm_query(('long ' if len(parts) > 1 else 'short ') + parts[0])\n"
    "answer['content'] = str(r)\nanswer['ready'] = True",
]

T = MockTiming(main_tok_per_s=2000, sub_base_s=0.1, sub_jitter_s=0.05, sub_tokens=2)


def strip_ids(s):
    return re.sub(r"\[\w+#\d+\] ", "", s or "")


@pytest.mark.parametrize("idx", range(len(TRICKY_EXEC)))
def test_spec_equals_baseline_on_tricky_shapes(idx):
    code = TRICKY_EXEC[idx]
    script = f"go\n```repl\n{code}\n```\n"
    answers = {}
    for mode in ("baseline", "spec"):
        eng = MockLM(T)
        h = Harness(eng, mode, context="ctx " * 50)
        out = h.run_turn(eng.stream_main(script))
        h.launcher.shutdown()
        answers[mode] = strip_ids(out.final_answer)
    assert answers["baseline"] == answers["spec"], code
