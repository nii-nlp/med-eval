#!/bin/bash
# This script evaluates Domain Classification models on various tasks.
#
# Usage:
# ./evaluate_dc.sh <model_name_or_path> <fewshot_size>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dir      : The directory where csv output result are saved.
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Example:
#   ./evaluate_dc.sh my_model 5
#   This will evaluate the model 'my_model' using 5-shot examples.
#
# Tasks:
# This script supports the following Domain Classification tasks:
#  crade, rrtnm, smdis

# Arguments
model_name_or_path=$1
model_log_dir=$2
fewshot_size=${3:-1}

# Fixed variables
tasks=(
    "crade"
    "rrtnm"
    "smdis"
)

# Templates for evaluation
templates=(
    "context_based_mcqa_minimal"
    "dc_with_options_jp"
    "dc_with_options"
    "dc_instructed_jp"
)

# Function to run torchrun for Domain Classification tasks
function run_dc {
    joined_tasks=$1
    template=$2
    log_file=$3

    python evaluate_mcqa.py \
        --model_name_or_path "${model_name_or_path}" \
        --task "${joined_tasks}" \
        --template_name "${template}" \
        --num_fewshot "${fewshot_size}" \
        --result_csv "$log_file"
}

# Join tasks
IFS=','
joined_tasks="${tasks[*]}"
unset IFS

# Prepare
if [ ! -d "$model_log_dir" ] ;then
    echo "$model_log_dir does not exits. Create the directory: $model_log_dir"
    mkdir -p "$model_log_dir"
fi

# Main loop
for _template in "${templates[@]}"; do
    run_dc "$joined_tasks" "$_template" "$model_log_dir/all-${_template}.csv"
done