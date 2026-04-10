#!/bin/bash

#SBATCH --job-name=3_Transformer_gen
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/j.framinan/TFG_repo/Intento_Transformers/logs/out-%x-%j.log
#SBATCH --error=/home/j.framinan/TFG_repo/Intento_Transformers/logs/err-%x-%j.log

export CUDA_VISIBLE_DEVICES="3"

echo "Starting job"
date

PROJECT_DIR=""

for CANDIDATE in \
	"${SLURM_SUBMIT_DIR:-}" \
	"${SLURM_SUBMIT_DIR:-}/Intento_Transformers" \
	"$HOME/TFG_repo/Intento_Transformers"
do
	if [[ -n "$CANDIDATE" && -f "$CANDIDATE/generate.py" && -f "$CANDIDATE/data_utils.py" && -f "$CANDIDATE/model.py" ]]; then
		PROJECT_DIR="$CANDIDATE"
		break
	fi
done

if [[ -z "$PROJECT_DIR" ]]; then
	echo "[ERROR] Estructura incompleta en: ${PROJECT_DIR:-<ninguno>}"
	echo "        Esperados: generate.py, data_utils.py, model.py"
	echo "        SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
	echo "        Candidatos probados:"
	echo "          - ${SLURM_SUBMIT_DIR:-<unset>}"
	echo "          - ${SLURM_SUBMIT_DIR:-<unset>}/Intento_Transformers"
	echo "          - $HOME/TFG_repo/Intento_Transformers"
	exit 1
fi

cd "$PROJECT_DIR"

echo "Project dir: $PROJECT_DIR"

#module load miniconda3/condabase
#eval "$(conda shell.bash hook)"
#conda activate test_env

python -u "$PROJECT_DIR/generate.py"

date
echo "Finished"
