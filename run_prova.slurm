#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug

srun python -m openpifpaf.predict /work/vita/datasets/Aff-Wild2/cropped_aligned/430/00261.jpg --image-output /home/trentini
deactivate
