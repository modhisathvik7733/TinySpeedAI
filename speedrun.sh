#!/usr/bin/env bash
# Speedrun: every variant gets the same wall-clock training budget; lowest final val loss wins.
# usage: bash speedrun.sh [budget_seconds] [variant...]      (default 45 s, all variants)
#   variants: gpt gpt_muon sghm_muon fgpt_s_muon fgpt_muon fgpt_l_muon fgpt_w_muon hpt_muon
set -e
cd "$(dirname "$0")"
BUDGET=${1:-45}
shift || true
VARIANTS=${@:-gpt gpt_muon sghm_muon fgpt_s_muon fgpt_muon fgpt_l_muon fgpt_w_muon hpt_muon}
python test_causal.py > /dev/null
mkdir -p runs
for v in $VARIANTS; do
  model=${v%_muon}
  opt=adamw; [ "$v" != "$model" ] && opt=muon
  echo "=== $v: model $model, opt $opt, ${BUDGET}s ==="
  OPT=$opt RUN_NAME=sr_$v TIME_BUDGET=$BUDGET EVAL_EVERY=${EVAL_EVERY:-200} python train.py "$model" 1000000 2>&1 \
    | tee "runs/sr_$v.log" | grep -E "params,|compile|steady|Traceback|Error|failed|val="
done
python compare.py sr_
