#!/bin/bash

#SBATCH --job-name=0_3d_vts_gen
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/out-%x-%j.log
#SBATCH --error=logs/err-%x-%j.log

export CUDA_VISIBLE_DEVICES="0"

echo "Starting job"
date

cd /home/j.framinan/TFG_repo/Manipulacion_de_Planos

#module load miniconda3/condabase
#eval "$(conda shell.bash hook)"
#conda activate test_env

python -u evaluate_3d_vts.py

date
echo "Finished"
