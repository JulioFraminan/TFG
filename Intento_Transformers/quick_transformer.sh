#!/bin/bash

#SBATCH --job-name=2_Transformer_quick
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/out-%x-%j.log
#SBATCH --error=logs/err-%x-%j.log

export CUDA_VISIBLE_DEVICES="2"

echo "Starting job"
date

cd ~/Intento_Transformers

#module load miniconda3/condabase
#eval "$(conda shell.bash hook)"
#conda activate test_env

python -u quick_test.py

date
echo "Finished"