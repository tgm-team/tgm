#!/bin/bash
#SBATCH --job-name=opendg-edgebank
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --output=out/%x.%j.out
#SBATCH --error=out/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacob.chmura@gmail.com

source .env

echo 'Hello world'
