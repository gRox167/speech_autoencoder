salloc -N 1 --cpus-per-task=4 -t 40:00:00 -p bme_gpu --gres=gpu:1
srun nvidia-smi