#!/usr/bin/env bash
# Prepare data, check the model, then train GPT and SGHM one after another and compare.
# usage: bash run.sh [steps] [models...]
#   bash run.sh                      -> 1000 steps, gpt + sghm
#   bash run.sh 3000                 -> 3000 steps, gpt + sghm
#   bash run.sh 1000 gpt sghm sghm_nogate sghm_nomem   -> full ablation set
# env: BATCH=256 bash run.sh   (bigger batch = more tokens/sec)
set -e
cd "$(dirname "$0")"
STEPS=${1:-1000}
shift || true
MODELS=${@:-gpt sghm}
MB=200

[ -f data/prepared_${MB}MB ] || python prepare.py $MB
python test_equiv.py
mkdir -p runs
for m in $MODELS; do
  echo "=== $m ($STEPS steps) ==="
  python train.py "$m" "$STEPS" 2>&1 | tee "runs/$m.log" | grep -E "params on|compile|^step|steady|Traceback|Error"
done
python compare.py
