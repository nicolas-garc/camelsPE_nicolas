#!/bin/bash
#SBATCH --job-name=camels-sweep
#SBATCH --time=72:00:00
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# Full observable-pair sweep: trains every pair, saves models, writes the
# per-parameter plots and feature vectors, then clusters the feature space.
#
#   mkdir -p logs
#   sbatch run_sweep.sh                      # one job, all 91 pairs, ~1-2 days
#   sbatch --array=0-7 run_sweep.sh          # 8 parallel shards (~4 h each)
#   sbatch --dependency=afterok:<id> run_sweep.sh --stages cluster
#
# Resubmitting after a timeout skips pairs that already have saved models.
# An array job splits the pairs and skips the cross-pair stages; run the
# cluster stage once afterwards (third line above).
# Adjust the module/env lines below for your cluster.

pwd; hostname; date

module add python
module add cuda
module add cudnn
# source /path/to/your/venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

SHARD=""
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
    SHARD="--shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
fi

python -u run_sweep.py \
    --data "${DATA:-../DATA/data_L50_TNG_v3.hdf5}" \
    --out  "${OUT:-sweep_output}" \
    $SHARD "$@"

date
