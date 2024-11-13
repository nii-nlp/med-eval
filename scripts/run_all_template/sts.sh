#!/bin/bash
# This script evaluates models on Semantic Textual Similarity (STS) tasks.
#
# Usage:
# ./evaluate_sts.sh <model_name_or_path> <fewshot_size>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dir      : The directory where csv output result are saved.
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Example:
#   ./evaluate_sts.sh my_model 5
#   This will evaluate the model 'my_model' using 5-shot examples.
#
# Tasks:
# This script supports the following STS tasks:
#  jcsts

# Arguments
model_name_or_path=$1
model_log_dir=$2
fewshot_size=${3:-0}

# Semi-fixed variables
NLI_LABELS="0,1,2,3,4,5"

# Fixed variables
tasks=(
    "jcsts"
)

# Templates for evaluation
templates=(
    "sts_minimal"
    "sts_as_nli_jp"
    "sts_as_nli"
    "sts_instructed_jp"
)

# Function to run torchrun for STS tasks
function run_sts {
    joined_tasks=$1
    template=$2
    log_file=$3

    python evaluate_sts.py \
        --model_name_or_path "${model_name_or_path}" \
        --task "${joined_tasks}" \
        --template_name "${template}" \
        --num_fewshot "${fewshot_size}" \
        --nli_labels "${NLI_LABELS}" \
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
    run_sts "$joined_tasks" "$_template" "$model_log_dir/all-${_template}.csv"
done
