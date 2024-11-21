#!/bin/bash
# This script evaluates Named Entity Recognition (NER) models on various NER tasks.
#
# Usage:
# bash scripts/run_all_template/evaluate_ner.sh <model_name_or_path> <model_log_file> <fewshot_size>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_file     : The result csv file
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Tasks:
# This script supports the following NER tasks:
#  mrner_disease, mrner_medicine, nrner, bc2gm_jp, bc5chem_jp, bc5disease_jp, jnlpba_jp, ncbi_disease_jp

source scripts/run_all_template/common.sh

# Arguments
model_name_or_path=$1
model_log_file=$2
fewshot_size=${3:-0}

# Semi-fixed variables
max_new_tokens=128

# Fixed variables
tasks=(
    "mrner_disease"
    "mrner_medicine"
    "nrner"
    "bc2gm_jp"
    "bc5chem_jp"
    "bc5disease_jp"
    "jnlpba_jp"
    "ncbi_disease_jp")

# Templates for evaluation
templates=(
    "minimal"
    "standard"
    "english-centric"
    "instructed"
)

# Function to run torchrun for NER tasks
function run_ner {
    joined_tasks=$1
    template=$2
    log_file=$3

    python evaluate_ner.py \
        --model_name_or_path "${model_name_or_path}" \
        --task "${joined_tasks}" \
        --template_name "${template}" \
        --num_fewshot "${fewshot_size}" \
        --max_new_tokens "${max_new_tokens}" \
        --result_csv "$log_file" \
        --data_type test
}

# repeat general tasks
all_tasks=$(repeat_tasks_for_templates tasks templates)
all_templates=$(repeat_templates_for_tasks templates tasks)

run_ner "$all_tasks" "$all_templates" "$model_log_file"
