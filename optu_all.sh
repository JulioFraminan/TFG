#!/bin/bash

#SBATCH --job-name=3_Optu_All
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=200:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/out-%x-%j.log
#SBATCH --error=logs/err-%x-%j.log

export ROCR_VISIBLE_DEVICES="3"

echo "Starting job"
date

cd /home/j.framinan/TFG_repo

#module load miniconda3/condabase
#eval "$(conda shell.bash hook)"
#conda activate test_env

python -u run_optuna_all.py

date
echo "Finished"