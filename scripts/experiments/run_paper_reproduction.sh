#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

method="${1:-}"
if [[ "$method" != "ppo" && "$method" != "grpo" ]]; then
    echo "Usage: bash scripts/experiments/run_paper_reproduction.sh {ppo|grpo} [Hydra overrides...]" >&2
    exit 2
fi
shift

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=, read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#gpu_ids[@]} < 1 || ${#gpu_ids[@]} > 8 )); then
    echo "Select between 1 and 8 training GPUs with CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi
export N_GPUS="${#gpu_ids[@]}"
# The two four-GPU groups on this host are separated by a NUMA boundary.
# Direct peer access stalled FSDP initialization in the eight-GPU smoke run.
if (( ${#gpu_ids[@]} > 4 )); then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi

export DATA_DIR="${DATA_DIR:-/home/jk/work/datasets/nq_hotpotqa_searchr1_paper}"
for split in train test; do
    if [[ ! -s "$DATA_DIR/$split.parquet" ]]; then
        echo "Missing $DATA_DIR/$split.parquet" >&2
        exit 2
    fi
done
if [[ ! -s "$DATA_DIR/validation_500_each.parquet" ]]; then
    echo "Missing validation subset; run python scripts/prepare_paper_validation.py --data-dir $DATA_DIR" >&2
    exit 2
fi

experiment="paper-nq-hotpotqa-qwen2.5-3b-${method}"
export TRAIN_LOG="${TRAIN_LOG:-logs/${experiment}.log}"
overrides=(
    "trainer.experiment_name=$experiment"
    "trainer.default_local_dir=verl_checkpoints/$experiment"
    "data.val_files=$DATA_DIR/validation_500_each.parquet"
    'trainer.total_training_steps=500'
    'trainer.save_freq=100'
    'trainer.test_freq=100'
    'max_turns=4'
)
if [[ "$method" == "grpo" ]]; then
    overrides+=('actor_rollout_ref.rollout.n=5')
fi

bash "scripts/${method}/train.sh" "${overrides[@]}" "$@"
