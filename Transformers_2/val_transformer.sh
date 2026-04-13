#!/bin/bash

#SBATCH --job-name=1_Transformer_val
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/out-%x-%j.log
#SBATCH --error=logs/err-%x-%j.log

export CUDA_VISIBLE_DEVICES="1"

echo "Starting job"
date

cd /home/j.framinan/TFG_repo/Transformers_2

#module load miniconda3/condabase
#eval "$(conda shell.bash hook)"
#conda activate test_env

python -u validation.py --results-folder results/intento_mat/dit_gaussian/cfg_6f0fe854e2cd --milestone 49 --prefer-ema --sampler ddim --num-inference-steps 300 --output-subdir validation_m49 \

date
echo "Finished"