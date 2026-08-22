set shell := ["bash", "-cu"]

test:
    PATH=$HOME/.bun/bin:$PATH uv run pytest tests/ -q

bench:
    uv run python -m benchmark.bench

serve:
    sbatch benchmark/experiments/serve.sbatch && watch -n 5 'squeue -u $USER; tail -2 .endpoints.env 2>/dev/null'

down:
    scancel -n spec-ptc-serve

demo scenario="oolong-mood-agg":
    uv run python -m demo.tui --scenario {{scenario}}

demo-live scenario="oolong-mood-agg":
    uv run python -m demo.tui --scenario {{scenario}} --vllm

play scenario="oolong-mood-agg" mode="spec":
    uv run python -m demo.play --scenario {{scenario}} --mode {{mode}}

list:
    uv run python -m demo.play --list
