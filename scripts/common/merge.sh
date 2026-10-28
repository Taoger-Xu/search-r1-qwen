#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# FSDP Model Merger Script
# This script merges a VERL FSDP checkpoint into a HuggingFace format model
# Usage: bash merge.sh [grpo|ppo]
# Default: grpo

METHOD=${1:-grpo}

if [ "$METHOD" = "grpo" ]; then
    LOCAL_DIR="verl_checkpoints/nq-search-r1-grpo-qwen2.5-3b-em/global_step_100/actor"
    TARGET_DIR="models/nq-search-r1-grpo-qwen2.5-3b-em"
elif [ "$METHOD" = "ppo" ]; then
    LOCAL_DIR="verl_checkpoints/nq-search-r1-ppo-qwen2.5-3b-em/global_step_100/actor"
    TARGET_DIR="models/nq-search-r1-ppo-qwen2.5-3b-em"
else
    echo "Error: Invalid method '$METHOD'. Use 'grpo' or 'ppo'"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Starting FSDP model merge for $METHOD..."
echo "Source: $LOCAL_DIR"
echo "Target: $TARGET_DIR"

# Check if source directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Source directory does not exist: $LOCAL_DIR"
    echo "Make sure training has completed and checkpoint exists."
    exit 1
fi

# Run the model merger
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

# Check if the merge was successful
if [ $? -eq 0 ]; then
    echo "Model merge completed successfully!"
    echo "Merged model saved to: $TARGET_DIR"
else
    echo "Model merge failed!"
    exit 1
fi
