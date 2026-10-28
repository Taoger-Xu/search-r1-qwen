#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/jk/work/envs/searchr1-retriever/bin/activate
export CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU:-7}"
export HF_HOME=/home/jk/work/datasets/.hf-cache
export HF_DATASETS_CACHE=/home/jk/work/datasets/.hf-cache/datasets
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
retriever_args=()
if [[ "${RETRIEVER_FAISS_GPU:-0}" == "1" ]]; then
    retriever_args+=(--faiss_gpu)
fi
exec python search/retrieval_server.py \
 --index_path /home/jk/work/datasets/nq_search/e5_Flat.index \
 --corpus_path /home/jk/work/datasets/nq_search/wiki-18.jsonl \
 --retriever_name e5 --retriever_model /home/jk/work/models/e5-base-v2 --topk 3 \
 "${retriever_args[@]}"
