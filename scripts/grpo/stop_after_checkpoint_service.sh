#!/usr/bin/env bash
set -euo pipefail

service="${1:?usage: stop_after_paper_checkpoint_service.sh SERVICE CHECKPOINT_DIR STEP WORLD_SIZE}"
checkpoint_dir="${2:?checkpoint directory required}"
target_step="${3:?target step required}"
world_size="${4:?world size required}"
marker="$checkpoint_dir/latest_checkpointed_iteration.txt"
step_dir="$checkpoint_dir/global_step_$target_step"

while true; do
    state="$(systemctl --user show "$service" -p ActiveState --value)"
    if [[ "$state" != active && "$state" != activating ]]; then
        echo "$(date -Is) $service ended before step $target_step checkpoint was verified" >&2
        exit 1
    fi
    if [[ -f "$marker" ]] && [[ "$(cat "$marker")" == "$target_step" ]]; then
        complete=true
        [[ -s "$step_dir/data.pt" ]] || complete=false
        for ((rank=0; rank<world_size; rank++)); do
            for part in model optim extra_state; do
                [[ -s "$step_dir/actor/${part}_world_size_${world_size}_rank_${rank}.pt" ]] || complete=false
            done
        done
        if [[ "$complete" == true ]]; then
            echo "$(date -Is) verified complete checkpoint: $step_dir"
            systemctl --user stop "$service"
            echo "$(date -Is) stopped $service"
            exit 0
        fi
    fi
    sleep 20
done
