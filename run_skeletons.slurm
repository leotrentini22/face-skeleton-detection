#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=10:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

cd tool/
srun python skeleton_process.py
deactivate
