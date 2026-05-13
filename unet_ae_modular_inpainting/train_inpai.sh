#!/bin/bash

#SBATCH --job-name=2_inpai_train
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/out-%x-%j.log
#SBATCH --error=logs/err-%x-%j.log

set -e

export ROCR_VISIBLE_DEVICES="2"

echo "Starting job"
date

cd /home/j.framinan/TFG_repo/unet_ae_modular_inpainting

source ~/.bashrc
conda activate test_env

python -u train.py

date
echo "Finished"