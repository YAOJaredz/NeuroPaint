#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="test_linear_ibl"
#SBATCH --output="logs/test_linear_ibl.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --mem=100000
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8,gpuH200x8
#SBATCH --chdir=/u/jyao7/NeuroPaint

echo "Running on $(hostname)"          # Print the name of the current node
echo "Using $(nproc) CPUs"             # Print the number of CPUs on the current node
echo "SLURM_JOB_ID: $SLURM_JOB_ID"     # Print the job ID
echo "SLURM_NODELIST: $SLURM_NODELIST" # Print the list of nodes assigned to this job

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

conda activate neuropaint

#load session id for ibl
session_order_file="data/tables_and_infos/ibl_eids.txt"
eids=$(python -c "with open('$session_order_file', 'r') as file: print('\n'.join([line.strip() for line in file]))")

# Print loaded eids for debugging
echo "Loaded eids: $eids"

flags="--smooth"

python -u src/test_linear_ibl.py --eids $eids $flags