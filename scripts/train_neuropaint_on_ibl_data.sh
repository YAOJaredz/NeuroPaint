#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="train_ibl"
#SBATCH --output="logs/train_ibl.%j.out"
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=100000
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

# Set WANDB_DIR to avoid cross-device file movement issues
export WANDB_DIR=/work/hdd/bdye/jxia4/wandb

# Calculate head node IP
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo Node IP: $head_node_ip

port=$(( 39000 + RANDOM % 1000 ))  # Random port between 39000-39999

export LAUNCHER="torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 1 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$port \
"

#load session id for ibl
session_order_file="data/tables_and_infos/ibl_eids.txt"
eids=$(python -c "with open('$session_order_file', 'r') as file: print('\n'.join([line.strip() for line in file]))")


# Print loaded eids for debugging
echo "Loaded eids: $eids"

export CMD="$LAUNCHER src/train_on_ibl.py --eids $eids --with_reg"


srun $CMD

conda deactivate

