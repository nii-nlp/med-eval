#!/bin/bash
# This script evaluates multiple-choice QA models on various medical and scientific QA tasks.
#
# Usage:
# bash scripts/run_all_template/evaluate_mcqa.sh <model_name_or_path> <example_size>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dir      : The directory where csv output result are saved.
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Example:
#   bash scripts/run_all_template/evaluate_mcqa.sh my_model 5
#   This will evaluate the model 'my_model' using 5-shot examples.
#
# Tasks:
# This script supports the following tasks:
#  medmcqa, medmcqa_jp, usmleqa, medqa, igakuqa, mmlu, jmmlu, pubmedqa, pubmedqa_jp
#

# Arguments
model_name_or_path=$1
model_log_dir=$2
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

# Join tasks
IFS=','
joined_general_tasks="${general_tasks[*]}"
joined_pubmed_tasks="${pubmed_tasks[*]}"
unset IFS

# Prepare
if [ ! -d "$model_log_dir" ] ;then
    echo "$model_log_dir does not exits. Create the directory: $model_log_dir"
    mkdir -p "$model_log_dir"
fi

for _template in "${general_templates[@]}"; do
    run_mcqa "$joined_general_tasks" "$_template" "${model_log_dir}/all_except_pubmed-${_template}.csv"
done
for _template in "${pubmed_templates[@]}"; do
    run_mcqa "$joined_pubmed_tasks" "$_template" "$model_log_dir/pubmed-${_template}.csv"
done
