#!/usr/bin/env bash
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs
while true; do
    stamp="$(date -Is)"
    nvidia-smi --query-gpu=index,uuid,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -v stamp="$stamp" '{print stamp "," $0}' >> logs/grpo-gpu-thermal.csv
    sleep 15
done
