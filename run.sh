#!/bin/bash
#
# SBATCH submission script for NeuroPaint
#
# Save this file to /u/jyao7/NeuroPaint/run.sh and make executable:
#   chmod +x /u/jyao7/NeuroPaint/run.sh
#
# Customize the headers below to match your cluster/account/requirements.

#SBATCH --job-name=NeuroPaint
#SBATCH --account=jyao7
#SBATCH --output=/u/jyao7/NeuroPaint/logs/%x_%j.out
#SBATCH --error=/u/jyao7/NeuroPaint/logs/%x_%j.err
#SBATCH --time=12:00:00                 # HH:MM:SS
#SBATCH --partition=gpuA40x4            # change to your partition (gpu, short, etc.)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/u/jyao7/NeuroPaint



# # Example job-array (uncomment to use): run array tasks 1..10 with max 4 concurrent
# #SBATCH --array=1-10%4

conda activate neuropaint

python src/loader/data_loader_ibl.py