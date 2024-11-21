#!/bin/bash
# This script evaluates machine translation models on various translation tasks.
#
# Usage:
# bash scripts/run_all_template/evaluate_mt.sh <model_name_or_path> <model_log_file> <numshot>
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_file     : The result csv file
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#
# Tasks:
# This script supports the following tasks:
#  ejmmt (English-Japanese and Japanese-English translation tasks)

source scripts/run_all_template/common.sh

# Arguments
model_name_or_path=$1
model_log_file=$2
fewshot_size=${3:-0}

# Semi-fixed variables
max_new_tokens=256

# Fixed variables
# Translation templates
tasks=("ejmmt")

ejmmt_en2ja=("english-japanese")
ejmmt_ja2en=("japanese-english")

ejmmt_en2ja_template=(
    "mt_minimal"
    "english_japanese"
    "mt_english_centric_e2j"
    "mt_instructed_e2j"
)
ejmmt_ja2en_template=(
    "mt_minimal"
    "japanese_english"
    "mt_english_centric_j2e"
    "mt_instructed_j2e"
)

# Function to run torchrun for MT tasks
function run_mt {
    task=$1
    translation=$2
    template=$3
    log_file=$4

    python evaluate_mt.py \
        --model_name_or_path "${model_name_or_path}" \
        --task "${task}" \
        --template_name "${template}" \
        --num_fewshot "${fewshot_size}" \
        --translation "${translation}" \
        --max_new_tokens "${max_new_tokens}" \
        --result_csv "$log_file" \
        --data_type test
}

# repeat tasks
repeated_en2ja_tasks=$(repeat_tasks_for_templates tasks ejmmt_en2ja_template)
repeated_ja2en_tasks=$(repeat_tasks_for_templates tasks ejmmt_ja2en_template)
all_tasks="${repeated_en2ja_tasks},${repeated_ja2en_tasks}"
# repeat translations
repeated_en2ja_translations=$(repeat_tasks_for_templates ejmmt_en2ja ejmmt_en2ja_template)
repeated_ja2en_translations=$(repeat_tasks_for_templates ejmmt_ja2en ejmmt_ja2en_template)
all_translations="${repeated_en2ja_translations},${repeated_ja2en_translations}"
# join templates
repeated_en2ja_templates=$(join_with_comma "${ejmmt_en2ja_template[@]}")
repeated_ja2en_templates=$(join_with_comma "${ejmmt_en2ja_template[@]}")
all_templates="${repeated_en2ja_templates},${repeated_ja2en_templates}"

# Main loop
run_mt "$all_tasks" "${all_translations}" "$all_templates" "$model_log_file"
