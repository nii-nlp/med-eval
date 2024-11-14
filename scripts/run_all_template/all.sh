#!/bin/bash
# This script runs evaluation for various NLP tasks (MCQA, NER, DC, STS, MT) using a specified model and few-shot settings.
#
# Usage:
# bash scripts/run_all_template/all.sh <model_name_or_path> <model_log_dirname>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dirname  : The directory name of result log saved under $log_dir .
#
# Example:
#   bash scripts/run_all_template/all.sh my_model log_1
#   This will evaluate the model 'my_model' for all tasks (MCQA, NER, DC, STS, MT) and shot settings.
#SBATCH --job-name=med-eval
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --nodelist ip-10-3-41-123,ip-10-3-56-13
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

source /etc/profile.d/modules.sh
module use /fsx/ubuntu/yokota_lab_workspace/module
module load cuda-12.1.1 cudnn-9.5.0
source venv/bin/activate

# Arguments
model_name_or_path=$1
model_log_dirname=$2

# Semi-fixed variables
log_dir="/fsx/ubuntu/shared/evaluation/med-eval-test-log"

# Batch size settings for each task and shot setting
tasks=(
    "mcqa"
    "mt"
    "ner"
    "dc"
    "sts"
)
shot_sizes=(
    0
    3
)

# Prepare
model_log_dir="$log_dir/$model_log_dirname"
if [ ! -d "$model_log_dir" ]; then
    echo "$model_log_dir does not exist. Creating the directory: $model_log_dir"
    mkdir -p "$model_log_dir"
fi

# Main
for _task in "${tasks[@]}"; do
    for _shot in "${shot_sizes[@]}"; do
        _log_dir="$model_log_dir/${_task}-${_shot}shot"
        mkdir -p "$_log_dir"
        echo "Running $_task with ${_shot}-shot setting"
        bash "scripts/run_all_template/${_task}.sh" "$model_name_or_path" "$_log_dir" "$_shot"
    done
done
