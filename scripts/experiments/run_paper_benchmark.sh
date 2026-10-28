#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

variant="${1:?Usage: bash scripts/experiments/run_paper_benchmark.sh VARIANT [Hydra overrides...]}"
shift
data_dir="${DATA_DIR:-/home/jk/work/datasets/nq_hotpotqa_searchr1_paper}"
if [[ ! -s "$data_dir/benchmark_32.parquet" ]]; then
    echo "Run python scripts/prepare_paper_benchmark.py first." >&2
    exit 2
fi
experiment="paper-benchmark-${variant}"
export TRAIN_LOG="logs/${experiment}.log"

bash scripts/experiments/run_paper_reproduction.sh grpo \
    "data.train_files=$data_dir/benchmark_32.parquet" \
    'data.shuffle=false' \
    'data.train_batch_size=32' \
    'actor_rollout_ref.actor.ppo_mini_batch_size=32' \
    "data.val_files=$data_dir/validation_smoke.parquet" \
    'trainer.total_training_steps=1' \
    'trainer.val_before_train=false' \
    'trainer.save_freq=-1' \
    'trainer.test_freq=-1' \
    "trainer.experiment_name=$experiment" \
    "trainer.default_local_dir=verl_checkpoints/$experiment" \
    "trainer.logger=['console']" \
    "$@"
