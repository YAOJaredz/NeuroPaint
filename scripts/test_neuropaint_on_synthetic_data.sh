#!/bin/bash
#SBATCH --account=jyao7
#SBATCH --job-name="test_syn"
#SBATCH --output="test_syn.%j.out"
#SBATCH --error="test_syn.%j.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=100000
#SBATCH --partition=gpuA100x4-interactive,gpuA40x4-interactive,gpuA100x8-interactive,gpuH200x8-interactive
#SBATCH --chdir=

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

module load anaconda3_gpu
source activate neuropaint


export CMD="python -u /root_folder/src/test_perf_syn.py"


srun $CMD

conda deactivate