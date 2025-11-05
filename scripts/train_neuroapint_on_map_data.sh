#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="train_map"
#SBATCH --output="logs/train_map.%j.out"
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=128000
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --chdir=/u/jyao7/NeuroPaint

echo "Running on $(hostname)"          # Print the name of the current node
echo "Using $(nproc) CPUs"             # Print the number of CPUs on the current node
echo "SLURM_JOB_ID: $SLURM_JOB_ID"     # Print the job ID
echo "SLURM_NODELIST: $SLURM_NODELIST" # Print the list of nodes assigned to this job

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

module load anaconda3_gpu/23.7.4
source activate neuropaint

# Set WANDB_DIR to avoid cross-device file movement issues
export WANDB_DIR=/root_folder2/wandb

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

# Load eids from session_order.pkl
session_order_file="/root_folder2/data/tables_and_infos/session_order.pkl"
eids=$(python -c "
import pickle
import numpy as np
session_order = pickle.load(open('$session_order_file', 'rb'))
eids = np.sort(session_order[:40])
#for 160 sessions
#eids = np.delete(eids, [ 12,  58, 123, 139, 146]) #for 160 sessions
print(' '.join(map(str, eids)))
")

# Print loaded eids for debugging
echo "Loaded eids: $eids"

export CMD="$LAUNCHER /root_folder/src/train_on_map.py --eids $eids"


srun $CMD

conda deactivate

