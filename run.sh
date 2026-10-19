#!/usr/bin/env bash
# Prepare data, run the correctness tests, then train each model one after another and compare.
# usage: bash run.sh [steps] [models...]
#   bash run.sh                          -> 1000 steps, gpt + hpt
#   bash run.sh 3000 gpt hpt hpt_s       -> 3000 steps, three models
#   MB=all bash run.sh 4000 hpt          -> full TinyStories (~550M tokens)
# env: BATCH / SEQ / LR override the model defaults; see train.py for the rest
set -e
cd "$(dirname "$0")"
STEPS=${1:-1000}
shift || true
MODELS=${@:-gpt hpt}
MB=${MB:-200}
MARK=data/prepared_${MB}$([ "$MB" = all ] || echo MB)

[ -f "$MARK" ] || python prepare.py "$MB"
python test_causal.py
python test_equiv.py
mkdir -p runs
for m in $MODELS; do
  echo "=== $m ($STEPS steps) ==="
  python train.py "$m" "$STEPS" 2>&1 | tee "runs/$m.log" | grep -E "params,|compile|^step|steady|Traceback|Error"
done
python compare.py
