#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=60:0:0
#SBATCH --qos=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

cd tool/
srun python skeleton_process_train.py
srun python skeleton_process_validation.py
srun python skeleton_process_test.py
deactivate
