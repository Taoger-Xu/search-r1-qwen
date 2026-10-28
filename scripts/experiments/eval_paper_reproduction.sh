#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/local_env.sh

model_path="${1:-}"
if [[ -z "$model_path" || ! -f "$model_path/config.json" ]]; then
    echo "Usage: bash scripts/experiments/eval_paper_reproduction.sh /path/to/merged-hf-model [Hydra overrides...]" >&2
    exit 2
fi
shift

export DATA_DIR="${DATA_DIR:-/home/jk/work/datasets/nq_hotpotqa_searchr1_paper}"
export RAY_TMPDIR="/tmp/ray-$(whoami)"
export RAY_ADDRESS=local
export VLLM_ATTENTION_BACKEND=XFORMERS
mkdir -p "$RAY_TMPDIR" logs

IFS=, read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#gpu_ids[@]} > 8 )); then
    echo "Select at most 8 GPUs with CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_eval_search_r1 \
    'hydra.run.dir=logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    "data.train_files=$DATA_DIR/train.parquet" \
    "data.val_files=$DATA_DIR/test.parquet" \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator=grpo \
    "actor_rollout_ref.model.path=$model_path" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4608 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    "trainer.n_gpus_per_node=${#gpu_ids[@]}" \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.validation_data_dir=inference/paper \
    max_turns=4 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=3 \
    "$@" 2>&1 | tee logs/eval_paper.log
