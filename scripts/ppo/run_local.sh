#!/usr/bin/env bash
set -euo pipefail
cd /home/jk/work/Search-R1-Qwen3
mkdir -p logs
rm -f logs/ppo_training.exit
bash scripts/retrieval/start_local.sh > logs/retriever_training.log 2>&1 &
retriever_pid=$!
echo "$retriever_pid" > logs/retriever_training.pid
echo $$ > logs/ppo_training.pid
trap 'kill "$retriever_pid" 2>/dev/null || true' EXIT
until curl -fsS --max-time 5 http://127.0.0.1:8000/health > /dev/null 2>&1; do
 if ! kill -0 "$retriever_pid" 2>/dev/null; then tail -30 logs/retriever_training.log; exit 1; fi
 sleep 10
done
if bash scripts/ppo/train.sh > logs/ppo_training_launch.log 2>&1; then
 train_exit=0
else
 train_exit=$?
fi
echo "$train_exit" > logs/ppo_training.exit
exit "$train_exit"
