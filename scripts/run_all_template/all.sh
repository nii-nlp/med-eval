#!/bin/bash
# This script runs evaluation for various NLP tasks (MCQA, NER, DC, STS, MT) using a specified model and few-shot settings.
#
# Usage:
# ./evaluate_all.sh <model_name_or_path> <model_log_dirname>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dirname  : The directory name of result log saved under $log_dir .
#
# Example:
#   ./evaluate_all.sh my_model log_1
#   This will evaluate the model 'my_model' using predefined batch sizes for all tasks (MCQA, NER, DC, STS, MT) and shot settings.

# Arguments
model_name_or_path=$1
model_log_dirname=$2

# Semi-fixed variables
log_dir="logs"
n_gpus=$(nvidia-smi -L | wc -l)

# Batch size settings for each task and shot setting
tasks=("mcqa" "mt" "ner" "dc" "sts")
shot_sizes=(0 3)

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

        # Execute the task-specific script with the corresponding batch size
        bash "scripts/run_all_template/${_task}.sh" "$model_name_or_path" "$_log_dir" "$_shot"
    done
done