#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --output=/hpc/data/home/bme/guochx/graduation_project/train.log

source /hpc/data/home/bme/guochx/.bashrc
conda activate torch18

# cat /etc/hosts
nvidia-smi
which python
srun python train.py
# jupyter lab --ip=0.0.0.0 --port=8888