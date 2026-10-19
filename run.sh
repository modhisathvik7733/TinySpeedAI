#!/usr/bin/env bash
# Prepare data, check the model, then train GPT and SGHM one after another and compare.
# usage: bash run.sh [steps] [models...]
#   bash run.sh                      -> 1000 steps, gpt + sghm_dense
#   bash run.sh 3000                 -> 3000 steps, gpt + sghm_dense
#   bash run.sh 1000 gpt sghm sghm_dense sghm_nogate sghm_nomem   -> full ablation set
set -e
cd "$(dirname "$0")"
STEPS=${1:-1000}
shift || true
MODELS=${@:-gpt sghm_dense}

[ -f data/train.bin ] || python prepare.py 40
python test_equiv.py
mkdir -p runs
for m in $MODELS; do
  echo "=== $m ($STEPS steps) ==="
  python train.py "$m" "$STEPS" 2>&1 | tee "runs/$m.log" | grep -E "params on|^step|Traceback|Error"
done
python compare.py
