#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=tgn_nodeprop.txt 
#SBATCH --error=tgn_nodeprop_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                  # Ask for 1 titan xp gpu:rtx8000:1 
#SBATCH --mem=32G #64G                             # Ask for 32 GB of RAM
#SBATCH --time=24:00:00    #48:00:00                   # The job will run for 1 day

module load python/3.10
source my_venv/bin/activate
export UV_CACHE_DIR="/home/mila/h/huangshe/scratch/.cache"

CUDA_VISIBLE_DEVICES=0 python -u examples/nodeproppred/tgn.py 
