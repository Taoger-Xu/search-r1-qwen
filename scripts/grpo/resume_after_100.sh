#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

experiment=paper-nq-hotpotqa-qwen2.5-3b-grpo-optimized
checkpoint_dir="verl_checkpoints/$experiment"
marker="$checkpoint_dir/latest_checkpointed_iteration.txt"
if [[ ! -f "$marker" ]]; then
    echo "No saved checkpoint at $checkpoint_dir" >&2
    exit 2
fi
saved_step="$(cat "$marker")"
step_dir="$checkpoint_dir/global_step_$saved_step"
if [[ ! "$saved_step" =~ ^[0-9]+$ ]] || (( saved_step < 100 )) || [[ ! -s "$step_dir/data.pt" ]]; then
    echo "Complete checkpoint at step 100 or later is required" >&2
    exit 2
fi
for rank in 0 1 2 3; do
    for part in model optim extra_state; do
        if [[ ! -s "$step_dir/actor/${part}_world_size_4_rank_${rank}.pt" ]]; then
            echo "Missing checkpoint file: $part rank $rank" >&2
            exit 2
        fi
    done
done

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TRAIN_LOG="logs/$experiment-resume-100.log"

bash scripts/experiments/run_paper_reproduction.sh grpo \
    "trainer.experiment_name=$experiment" \
    "trainer.default_local_dir=$checkpoint_dir" \
    'trainer.resume_mode=auto' \
    'trainer.val_before_train=false' \
    'trainer.save_freq=15' \
    'trainer.save_freq_offset=100' \
    'trainer.test_freq=10' \
    'trainer.max_complete_ckpt_to_keep=1' \
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4' \
    'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4' \
    'actor_rollout_ref.rollout.max_num_seqs=32' \
    'actor_rollout_ref.rollout.max_num_batched_tokens=8192' \
    "$@"
