#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="test_ibl"
#SBATCH --output="logs/test_ibl.%j.out"
#SBATCH --error="logs/test_ibl.%j.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --mem=100000
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8,gpuH200x8

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

# module load anaconda3_gpu
# source activate neuropaint
conda activate neuropaint

#load session id for ibl
session_order_file="/work/hdd/bdye/jyao7/data/tables_and_infos/ibl_eids.txt"
eids=$(python -c "with open('$session_order_file', 'r') as file: print('\n'.join([line.strip() for line in file]))")

# Print loaded eids for debugging
echo "Loaded eids: $eids"

export CMD="python -u src/test_perf_ibl.py --eids $eids --with_reg"


srun $CMD

conda deactivate