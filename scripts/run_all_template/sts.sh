#!/bin/bash
# This script evaluates models on Semantic Textual Similarity (STS) tasks.
#
# Usage:
# bash scripts/run_all_template/evaluate_sts.sh <model_name_or_path> <model_log_file> <fewshot_size>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_file     : The result csv file
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Tasks:
# This script supports the following STS tasks:
#  jcsts

source scripts/run_all_template/common.sh

# Arguments
model_name_or_path=$1
model_log_file=$2
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
        --result_csv "$log_file" \
        --data_type test
}

# repeat  tasks
all_tasks=$(repeat_tasks_for_templates tasks templates)
all_templates=$(repeat_templates_for_tasks templates tasks)

run_sts "$all_tasks" "$all_templates" "$model_log_file"
