# Shared local paths. Training uses GPUs 0-3; retrieval uses GPU 7.
source /home/jk/work/envs/searchr1-verl/bin/activate
export PYTHONPATH="/home/jk/work/Search-R1-Qwen3${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/home/jk/work/datasets/.hf-cache
export HF_DATASETS_CACHE=/home/jk/work/datasets/.hf-cache/datasets
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

# Unified experiment logs (ignored by Git).
export WANDB_DIR="/home/jk/work/Search-R1-Qwen3/logs/wandb"
export SWANLAB_LOG_DIR="/home/jk/work/Search-R1-Qwen3/logs/swanlab"
mkdir -p "$WANDB_DIR" "$SWANLAB_LOG_DIR"
