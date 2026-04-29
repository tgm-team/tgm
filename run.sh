#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=relhm_tgn.txt
#SBATCH --error=relhm_tgn_error.txt
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                  # Ask for 1 titan xp gpu:rtx8000:1
#SBATCH --mem=128G #64G                             # Ask for 32 GB of RAM
#SBATCH --time=48:00:00    #48:00:00                   # The job will run for 1 day

module load python/3.10
source /home/mila/h/huangshe/scratch/my_venv/bin/activate
export UV_CACHE_DIR="/home/mila/h/huangshe/scratch/.cache"
pwd


CUDA_VISIBLE_DEVICES=0 python -u -m examples.linkproppred.relbench.train \
    --device cuda        \
    --epochs 20          \
    --bsize 512          \
    --memory-dim 128     \
    --embed-dim 128      \
    --time-dim 128       \
    --n-nbrs 20          \
    --lr 3e-4            \
    --log-file-path run.log
