#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/local_env.sh

export DATA_DIR="${DATA_DIR:-/home/jk/work/datasets/nq_search}"

# Automatically detect number of GPUs
if [ -z "${N_GPUS:-}" ]; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        N_GPUS=1
    else
        N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    fi
fi

# Avoid multi-user conflicts of Ray by using a local instance
export RAY_TMPDIR=/tmp/ray-$(whoami)
mkdir -p "$RAY_TMPDIR"
export RAY_ADDRESS=local

# wandb logging settings
export WANDB_MODE=offline              # makes Weights & Biases log locally without requiring an API key
export WANDB_PROJECT=Search-R1         # project name used when you sync
export WANDB_NAME=nq-qwen2.5-3b      # friendly run name
export WANDB_RUN_GROUP=trial-001       # group multiple processes/runs


# Evaluation configuration
export BASE_MODEL='models/nq-search-r1-ppo-qwen2.5-3b-em'  # Path to your trained PPO checkpoint (after merge)
export EXPERIMENT_NAME="nq-search-r1-ppo-qwen2.5-3b-em-eval"

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "Evaluation configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Data directory: $DATA_DIR"
echo "  Experiment name: $EXPERIMENT_NAME"
echo "  Retrieval URL: http://127.0.0.1:8000/retrieve"

# Create logs directory if it doesn't exist
mkdir -p logs inference

# Start evaluation using the trained PPO model

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_eval_search_r1 \
    'hydra.run.dir=logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4608 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.val_before_train=true \
    trainer.validation_data_dir="inference" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee "${EVAL_LOG:-logs/${EXPERIMENT_NAME}.log}"

# Capture the exit code of the evaluation process
EVAL_EXIT_CODE=$?

# Rename the validation output file to the desired name
# Find the most recent .jsonl file in the validation directory
VALIDATION_FILE=$(find inference -name "*.jsonl" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$VALIDATION_FILE" ] && [ -f "$VALIDATION_FILE" ]; then
    mv "$VALIDATION_FILE" "inference/nq_search_results.jsonl"
    echo "Copied validation output from $(basename $VALIDATION_FILE) to nq_search_results.jsonl"
else
    echo "Warning: No validation output file found to rename"
fi

echo ""
echo ":checkered_flag: Evaluation completed with exit code: $EVAL_EXIT_CODE"
