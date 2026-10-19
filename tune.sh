#!/usr/bin/env bash
# Round 2: tune the speedrun winner (fgpt + Muon) at the same wall-clock budget. Lowest val@T256 wins.
# usage: bash tune.sh [budget_seconds]      (default 45; ~15 min total)
cd "$(dirname "$0")"
BUDGET=${1:-45}
mkdir -p runs
run() {
  name=$1; shift
  echo "=== $name: $* ==="
  env "$@" OPT=${OPT:-muon} RUN_NAME=t_$name TIME_BUDGET=$BUDGET EVAL_EVERY=200 python train.py ${MODEL:-fgpt} 1000000 2>&1 \
    | tee "runs/t_$name.log" | grep -E "params,|steady|Traceback|Error|failed"
}
run base
run seed2   SEED=2                       # same config, other seed: the noise level
run mlr02   MUON_LR=0.02
run mlr05   MUON_LR=0.05
run alr6    LR=6e-3                      # AdamW part (embeddings, head)
run alr25   LR=2.5e-3
run b64     BATCH=64                     # more, smaller steps
run b256    BATCH=256                    # fewer, bigger steps
run L5      CFG='{"L":5}'
run L3      CFG='{"L":3}'
run d320    CFG='{"d":320,"heads":5}'
run mlp3    CFG='{"mlp":3}'
run head2k  CFG='{"n_head":2048}'
run head512 CFG='{"n_head":512}'
python compare.py t_
