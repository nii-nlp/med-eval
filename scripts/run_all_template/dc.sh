#!/bin/bash
# This script evaluates Domain Classification models on various tasks.
#
# Usage:
# bash scripts/run_all_template/evaluate_ner.evaluate_dc <model_name_or_path> <model_log_file> <fewshot_size>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_file     : The result csv file
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Tasks:
# This script supports the following Domain Classification tasks:
#  crade, rrtnm, smdis

source scripts/run_all_template/common.sh

# Arguments
model_name_or_path=$1
model_log_file=$2
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
        --result_csv "$log_file" \
        --data_type test
}

# repeat general tasks
all_tasks=$(repeat_tasks_for_templates tasks templates)
all_templates=$(repeat_templates_for_tasks templates tasks)

run_dc "$all_tasks" "$all_templates" "$model_log_file"
