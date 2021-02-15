#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=train_infogan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12G
#SBATCH --mem-per-gpu=12G
#SBATCH --exclusive

srun python -m infogan train_cfg