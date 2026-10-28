#!/usr/bin/env bash
set -euo pipefail

# Stop the current GRPO run only after veRL has committed a resumable checkpoint.
run_group="${1:?usage: stop_after_paper_checkpoint.sh PROCESS_GROUP CHECKPOINT_DIR STEP}"
checkpoint_dir="${2:?checkpoint directory required}"
target_step="${3:?target step required}"
marker="$checkpoint_dir/latest_checkpointed_iteration.txt"
step_dir="$checkpoint_dir/global_step_$target_step"

while true; do
    if [[ -f "$marker" ]] && [[ "$(cat "$marker")" == "$target_step" ]]; then
        complete=true
        [[ -s "$step_dir/data.pt" ]] || complete=false
        for rank in 0 1 2 3; do
            for part in model optim extra_state; do
                [[ -s "$step_dir/actor/${part}_world_size_4_rank_${rank}.pt" ]] || complete=false
            done
        done
        if [[ "$complete" == true ]]; then
            echo "$(date -Is) verified complete checkpoint: $step_dir"
            kill -TERM -- "-$run_group"
            echo "$(date -Is) sent SIGTERM to training process group $run_group"
            exit 0
        fi
    fi

    if ! kill -0 -- "-$run_group" 2>/dev/null; then
        echo "$(date -Is) training process group ended before checkpoint was verified" >&2
        exit 1
    fi
    sleep 20
done
