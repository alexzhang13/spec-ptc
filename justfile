set shell := ["bash", "-cu"]

# one-time per clone: enable the ruff+pytest pre-commit hook
hooks:
    git config core.hooksPath .githooks

test:
    PATH=$HOME/.bun/bin:$PATH uv run pytest tests/ -q

bench:
    uv run python -m benchmark.bench

# live demo endpoints (Qwen3.5-27B for both main and sub) on a free guest node
serve:
    sbatch infra/serve_demo.sbatch
    @echo "watch with: just status   (ready when both curls return a model id)"

status:
    @squeue -u $USER
    @echo; cat .endpoints.env 2>/dev/null || echo "(no .endpoints.env yet)"
    @echo; source .endpoints.env 2>/dev/null && \
      for u in $SPEC_MAIN_URL $SPEC_SUB_URL; do \
        printf "%-32s " $u; m=$(curl -sf -m 3 $u/models | grep -o '"id":"[^"]*"' | head -1); echo "${m:-not up}"; \
      done

down:
    scancel -n spec-ptc-serve

# flagship: spec vs serial racing a REAL RLM on an OOLONG 32k task (needs `just serve`)
demo:
    uv run python -m demo.oolong_race

# interactive CodeAct session over the same live endpoints
codeact:
    uv run python -m demo.codeact
