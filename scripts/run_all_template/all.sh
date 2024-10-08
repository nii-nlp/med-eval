#!/bin/bash
# This script runs evaluation for various NLP tasks (MCQA, NER, DC, STS, MT) using a specified model and few-shot settings.
#
# Usage:
# ./evaluate_all.sh <model_name_or_path> <model_log_dirname> [<batch_size>] [<n_gpus>]
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dirname  : The directory name of result log saved under $log_dir .
#   batch_size         : The batch size for evaluation. Defaults to 1 if not specified.
#   n_gpus             : The number of GPUs to use for evaluation. Defaults to the number of available GPUs if not specified.
#
# Example:
#   ./evaluate_all.sh my_model 1 4
#   This will evaluate the model 'my_model' with a batch size of 1 using 4 GPUs on all tasks (MCQA, NER, DC, STS, MT).
#
# Tasks:
# This script supports the following tasks:
#   - MCQA (Multiple Choice Question Answering)
#   - NER (Named Entity Recognition)
#   - DC (Domain Classification)
#   - STS (Semantic Textual Similarity)
#   - MT (Machine Translation)
#
# Few-shot settings:
# The script evaluates the model using 0-shot and 3-shot settings.

n_gpus_default=$(nvidia-smi -L | wc -l)

# Arguments
model_name_or_path=$1
model_log_dirname=$2
batch_size=${3:-1}
n_gpus=${4:-$n_gpus_default}

# Semt-fix variables
log_dir="logs"
task_types=("mcqa" "ner" "dc" "sts" "mt")
shot_sizes=(0 3)

# Prepare
model_log_dir="$log_dir/$model_log_dirname"
if [ ! -d "$model_log_dir" ] ;then
    echo "$model_log_dir does not exits. Create the directory: $model_log_dir"
    mkdir -p "$model_log_dir"
fi

# Main
for task_type in "${task_types[@]}"; do
    for numshot in "${shot_sizes[@]}"; do
        _log_dir="$model_log_dir/$task_type-${numshot}shot"
        mkdir -p "$_log_dir"
        bash "scripts/run_all_template/${task_type}.sh" "$model_name_or_path" "$_log_dir" "$numshot" "$batch_size" "$n_gpus"
    done
done
