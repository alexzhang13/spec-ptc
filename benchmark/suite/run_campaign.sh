#!/bin/bash
# Suite campaign stages (each stage is resumable on its own tag).
#   bash benchmark/suite/run_campaign.sh                    # all stages
#   STAGES="rates aa" bash benchmark/suite/run_campaign.sh   # pick stages
set -u
cd /shared/home/altzhang-de4f8c/spec-ptc
R=${REPEATS:-5}
STAGES=${STAGES:-"main wave2 aa rates conc model2"}
WAVE2=${WAVE2:-"nonspec_only,pure_compute,mutating_loop_slow,wide_then_discard,retry_loop,deep_chain12,agent_turn,batched_duplicates"}
# representative subset for the A/A control
SUBSET=${SUBSET:-"map16,map_reduce,two_stage,mutating_loop,serial_chain6,chain4_slow,sweep_w08,sweep_w16,sweep_len320,dependent_args,turn2_map,long_prose_burst,prose_sandwich,batched8"}
# smaller subset for the rate sweep (slow streams at 20 tok/s)
RATE_SET=${RATE_SET:-"map16,map_reduce,serial_chain6,chain4_slow,sweep_w08,dependent_args,turn2_map,prose_sandwich"}
# cross-model generality pass: wins, floors, adversarial, and both sweeps
M30B=${M30B:-"map16,map32,map_reduce,two_stage,map8_slow,agent_turn,long_prose_burst,mutating_loop,serial_chain6,chain4_slow,deep_chain12,dependent_args,recursion,batched8,taint_split,tool_error_recovered,class_method,turn2_map,turn2_generator,wide_then_discard,sweep_w01,sweep_w08,sweep_w16,sweep_len640"}
run() { echo "=== $(date +%H:%M:%S) $*"; uv run python -m benchmark.suite.run_suite "$@"; }

for s in $STAGES; do
  case $s in
    main)   run --repeats "$R" --tag main    --tps 60 ;;
    wave2)  run --repeats "$R" --tag wave2   --tps 60 --only "$WAVE2" ;;
    aa)     run --repeats "$R" --tag aa      --tps 60 --only "$SUBSET" --aa ;;
    rates)  run --repeats "$R" --tag rate20  --tps 20  --only "$RATE_SET"
            run --repeats "$R" --tag rate150 --tps 150 --only "$RATE_SET" ;;
    conc)   run --repeats "$R" --tag conc4   --tps 60 --conc 4 --only "$SUBSET" ;;
    model2) run --repeats "$R" --tag m30b    --tps 60 --endpoint 2 --only "$M30B" ;;
  esac
done
echo "CAMPAIGN COMPLETE $(date +%H:%M:%S)"
