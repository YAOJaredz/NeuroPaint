#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="load_ibl"
#SBATCH --output="logs/load_ibl.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=128000
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8
#SBATCH --chdir=/u/jyao7/NeuroPaint

echo "Running on $(hostname)"          # Print the name of the current node
echo "Using $(nproc) CPUs"             # Print the number of CPUs on the current node
echo "SLURM_JOB_ID: $SLURM_JOB_ID"     # Print the job ID
echo "SLURM_NODELIST: $SLURM_NODELIST" # Print the list of nodes assigned to this job

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

conda activate neuropaint

python -u src/loader/data_loader_ibl.py