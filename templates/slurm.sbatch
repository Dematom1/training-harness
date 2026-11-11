#!/usr/bin/env bash
#SBATCH --job-name=gpt-measure
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%j.out

# Edit allocation directives for your partition/topology before submitting.
# This template is NOT submitted by installation, tests, or the sweep dry-run.
set -euo pipefail
: "${TRAIN_ENTRYPOINT:?Set TRAIN_ENTRYPOINT to your training script}"
: "${HARNESS_LOG_ROOT:?Set HARNESS_LOG_ROOT to a durable shared directory}"
: "${HARNESS_GPU_TYPE:?Set HARNESS_GPU_TYPE to the actual accelerator SKU}"
: "${HARNESS_WORKLOAD:?Set HARNESS_WORKLOAD to your comparable workload ID}"
export HARNESS_RUN_ID="${HARNESS_RUN_ID:-slurm-${SLURM_JOB_ID}}"
export HARNESS_LOG="${HARNESS_LOG_ROOT}/${HARNESS_RUN_ID}.jsonl"
export HARNESS_WORLD_SIZE="${SLURM_NTASKS}"
export HARNESS_BACKEND=ddp

# USR1 targets all srun tasks (no B: prefix), so every rank reaches the hook.
# Each rank owns one GPU. The loop initializes DDP using Slurm rank metadata.
set +e
srun --kill-on-bad-exit=1 python "$TRAIN_ENTRYPOINT"
rc=$?
set -e
if [[ "$rc" -eq 75 ]]; then
    # The loop finished a durable checkpoint and requested requeue at a safe point.
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$rc"
