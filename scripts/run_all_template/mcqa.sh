#!/bin/bash
# This script evaluates multiple-choice QA models on various medical and scientific QA tasks.
#
# Usage:
# bash scripts/run_all_template/evaluate_mcqa.sh <model_name_or_path> <model_log_file> <few_shot_size>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_file     : The result csv file
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Tasks:
# This script supports the following tasks:
#  medmcqa, medmcqa_jp, usmleqa, medqa, igakuqa, mmlu, jmmlu, pubmedqa, pubmedqa_jp
#

source scripts/run_all_template/common.sh

# Arguments
model_name_or_path=$1
model_log_file=$2
fewshot_size=${3:-0}

# Fixed variables
general_tasks=(
    "medmcqa"
    "medmcqa_jp"
    "usmleqa"
    "usmleqa_jp"
    "medqa"
    "medqa_jp"
    "igakuqa"
    "igakuqa_en"
    "mmlu"
    "mmlu_medical"
    "mmlu_medical_jp"
    "jmmlu"
    "jmmlu_medical"
)
pubmed_tasks=(
    "pubmedqa"
    "pubmedqa_jp"
)

general_templates=(
    "mcqa_minimal"
    "mcqa_with_options_jp"
    "mcqa_with_options"
    "4o_mcqa_instructed_jp"
)
pubmed_templates=(
    "context_based_mcqa_minimal"
    "context_based_mcqa_jp"
    "context_based_mcqa"
    "context_based_mcqa_instructed_jp"
)

function run_mcqa {
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
repeated_general_tasks=$(repeat_tasks_for_templates general_tasks general_templates)
repeated_general_templates=$(repeat_templates_for_tasks general_templates general_tasks)
# repeat pubmed tasks
repeated_pubmed_tasks=$(repeat_tasks_for_templates pubmed_tasks pubmed_templates)
repeated_pubmed_templates=$(repeat_templates_for_tasks pubmed_templates pubmed_tasks)
# merge
repeated_tasks="${repeated_general_tasks},${repeated_pubmed_tasks}"
repeated_templates="${repeated_general_templates},${repeated_pubmed_templates}"
# run
run_mcqa "$repeated_tasks" "$repeated_templates" "$model_log_file"
