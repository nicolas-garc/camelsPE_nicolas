#!/bin/bash
#SBATCH --job-name=camels-sweep
#SBATCH --time=72:00:00
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# Full observable-pair sweep in one run: trains every pair, saves models,
# writes feature vectors and per-parameter plots to sweep_output/.
#
#   mkdir -p logs && sbatch run_sweep.sh
#
# If the job hits the time limit, resubmit the same command: pairs with saved
# models are loaded instead of retrained, so it continues into the same output.
# Adjust the module/env lines below for your cluster.

pwd; hostname; date

module add python
module add cuda
module add cudnn
# source /path/to/your/venv/bin/activate

cd "$SLURM_SUBMIT_DIR"
python -u run_sweep.py \
    --data "${DATA:-../DATA/data_L50_TNG_v3.hdf5}" \
    --out  "${OUT:-sweep_output}" \
    "$@"

date
