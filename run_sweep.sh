#!/bin/bash
# Full observable-pair sweep: trains every pair, saves models, feature vectors
# and plots into sweep_output/. One continuous run.
#
#   sbatch run_sweep.sh
#
# If the job hits the time limit, resubmit the exact same command: pairs whose
# models are already saved are loaded instead of retrained, so it continues
# where it stopped and writes into the same output directory.
#
# ==> EDIT THE THREE MARKED SECTIONS BELOW FOR YOUR CLUSTER. <==

#SBATCH --job-name=camels-sweep
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err

### EDIT 1 — partition / GPU.
# The models are small MLPs; a GPU helps but is not required (the script picks
# CUDA when available and falls back to CPU). Uncomment and adjust if you want
# a GPU, and set your site's partition name:
###SBATCH -p gpu
###SBATCH --gpus-per-node=1

set -euo pipefail

pwd; hostname; date

### EDIT 2 — environment. Replace with whatever makes python + the packages in
### requirements.txt available on your cluster. Examples:
# module load python/3.11 cuda cudnn
# source /path/to/venv/bin/activate
# conda activate camels

### EDIT 3 — nothing usually, but --data can point elsewhere if you did not
### use the data file shipped in data/.
python -u run_sweep.py --out "${OUT:-sweep_output}" "$@"

date
