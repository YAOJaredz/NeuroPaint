#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="train_linear_ibl"
#SBATCH --output="logs/train_linear_ibl.%A_%a.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --mem=100000
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8,gpuH200x8
#SBATCH --chdir=/u/jyao7/NeuroPaint
#SBATCH --array=0-7

echo "Running on $(hostname)"          # Print the name of the current node
echo "Using $(nproc) CPUs"             # Print the number of CPUs on the current node
echo "SLURM_JOB_ID: $SLURM_JOB_ID"     # Print the job ID
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST" # Print the list of nodes assigned to this job

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

conda activate neuropaint

# Set WANDB_DIR to avoid cross-device file movement issues
export WANDB_DIR=/work/hdd/bdye/jyao7/wandb

# load session id for ibl
session_order_file="data/tables_and_infos/ibl_eids.txt"
eids=$(python -c "with open('$session_order_file', 'r') as file: print('\n'.join([line.strip() for line in file]))")

echo "Loaded eids: $eids"

declare -a COMBOS=(
    ""                              # none
    "--with_reg"                    # with_reg only
    "--consistency"                 # consistency only
    "--smooth"                      # smooth only
    "--with_reg --smooth"           # with_reg + smooth
    "--consistency --smooth"        # consistency + smooth
    "--with_reg --consistency"      # with_reg + consistency
    "--with_reg --consistency --smooth"  # with_reg + consistency + smooth
)

flags="${COMBOS[$SLURM_ARRAY_TASK_ID]}"

echo "Using flags: '$flags'"

# Run train and test with this combination
python -u src/train_linear_ibl.py --eids $eids $flags
python -u src/test_linear_ibl.py  --eids $eids $flags --override