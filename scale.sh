#!/usr/bin/env bash
# Scaling claims, throughput only (QUICK=1, 300 steps per point, 32k tokens per step everywhere):
#   (a) tok/s vs context length T, SAC vs FGPT (transformer)   (b) tok/s vs SAC memory slots N
cd "$(dirname "$0")"
run() { env QUICK=1 OPT=muon "$@" 2>&1 | grep -oE "throughput: [0-9.]+M" | grep -oE "[0-9.]+M" || echo FAIL; }
echo "(a) tok/s vs context length"
printf "%-6s %9s %9s %9s\n" model T=256 T=1024 T=4096
for m in fgpt sac; do
  printf "%-6s" $m
  for T in 256 1024 4096; do printf " %9s" "$(run SEQ=$T BATCH=$((32768 / T)) python train.py $m 300)"; done
  echo
done
echo "(b) tok/s vs sparse memory slots (T=256)"
printf "%-10s %9s %9s\n" model slots tok/s
for m in sac:65k sac_m262k:262k sac_m1m:1M; do
  printf "%-10s %9s %9s\n" ${m%%:*} ${m##*:} "$(run python train.py ${m%%:*} 300)"
done
