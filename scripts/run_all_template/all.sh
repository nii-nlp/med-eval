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
task_shot_batch=(
    "mcqa 0 4"
    "mcqa 3 1"
    "mt 0 12"
    "mt 3 6"
    "ner 0 5"
    "ner 3 1"
    "dc 0 7"
    "dc 3 1"
    "sts 0 23"
    "sts 3 8"
)

# Prepare
model_log_dir="$log_dir/$model_log_dirname"
if [ ! -d "$model_log_dir" ]; then
    echo "$model_log_dir does not exist. Creating the directory: $model_log_dir"
    mkdir -p "$model_log_dir"
fi

# Main 
for entry in "${task_shot_batch[@]}"; do
    # Split the entry into task_type, shot, and batch_size
    task_type=$(echo "$entry" | awk '{print $1}')
    numshot=$(echo "$entry" | awk '{print $2}')
    batch_size=$(echo "$entry" | awk '{print $3}')

    _log_dir="$model_log_dir/$task_type-${numshot}shot"
    mkdir -p "$_log_dir"

    echo "Running $task_type with ${numshot}-shot setting using batch size: $batch_size"

    # Execute the task-specific script with the corresponding batch size
    bash "scripts/run_all_template/${task_type}.sh" "$model_name_or_path" "$_log_dir" "$numshot" "$batch_size" "$n_gpus"
done
