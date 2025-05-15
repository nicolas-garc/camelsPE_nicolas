#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-type=ALL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
###SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail
#SBATCH --time=72:00:00               # Time limit hrs:min:sec #SBATCH -p gpu --gpus=1 -c 16

#SBATCH -C a100
#SBATCH -p gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 4 GPUs, 4 tasks
#SBATCH --gpus-per-node=1            # 4 GPUs
#SBATCH --cpus-per-task=12

pwd; hostname; date

module add python
module add cuda
module add cudnn


source /mnt/home/yjo10/pyenv/torch/bin/activate
export PATH="$VIRTUAL_ENV/bin:$PATH"

cd $(pwd)

python main.py > log/stdout 2> log/stderr


date

