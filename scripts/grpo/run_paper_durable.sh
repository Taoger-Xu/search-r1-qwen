#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
experiment=paper-nq-hotpotqa-qwen2.5-3b-grpo-7gpu-b336
checkpoint_dir="verl_checkpoints/$experiment"
export TRAIN_LOG="logs/$experiment.log"
mkdir -p logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(/usr/bin/python3 - <<'PYGPU'
import os, subprocess
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader'], text=True, timeout=30)
devices = sorted((bus.strip(), uuid.strip()) for uuid, bus in (row.split(',') for row in rows.splitlines()))
assert len(devices) == 8, 'Expected all eight GPUs; refusing to remap a missing device'
uuids = [uuid for _, uuid in devices]
wanted = os.environ['TRAIN_GPU_UUIDS'].split(',')
assert len(wanted) == len(set(wanted)) == 7
print(','.join(str(uuids.index(uuid)) for uuid in wanted))
PYGPU
)"
[[ -n "$CUDA_VISIBLE_DEVICES" ]] || exit 1
echo "Training CUDA devices (PCI order): $CUDA_VISIBLE_DEVICES"
# Limit failure inspection to this invocation, excluding historical errors.
stat -c %s "$TRAIN_LOG" > logs/grpo-attempt-offset 2>/dev/null || echo 0 > logs/grpo-attempt-offset
for _ in {1..240}; do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
    sleep 5
done
curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null
exec /bin/bash scripts/experiments/run_paper_reproduction.sh grpo \
    "trainer.experiment_name=$experiment" \
    "trainer.default_local_dir=$checkpoint_dir" \
    trainer.resume_mode=auto \
    trainer.val_before_train=false \
    data.train_batch_size=336 \
    actor_rollout_ref.actor.ppo_mini_batch_size=336 \
    trainer.save_freq=5 \
    trainer.max_complete_ckpt_to_keep=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_seqs=96 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
