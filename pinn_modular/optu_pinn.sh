#!/bin/bash

#SBATCH --job-name=0_Optu_PINN
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/out-%x-%j.log
#SBATCH --error=logs/err-%x-%j.log

export ROCR_VISIBLE_DEVICES="0"

echo "Starting job"
date

cd /home/j.framinan/TFG_repo/pinn_modular

#module load miniconda3/condabase
#eval "$(conda shell.bash hook)"
#conda activate test_env

python -u optuna_tune.py

date
echo "Finished"