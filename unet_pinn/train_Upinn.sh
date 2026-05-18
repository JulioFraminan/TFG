#!/bin/bash

#SBATCH --job-name=3_Upinn_train
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/out-%x-%j.log
#SBATCH --error=logs/err-%x-%j.log

set -e

export ROCR_VISIBLE_DEVICES="3"

echo "Starting job"
date

cd /home/j.framinan/TFG_repo/unet_pinn

source ~/.bashrc
conda activate test_env

python -u train.py

date
echo "Finished"