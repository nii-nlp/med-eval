#!/bin/bash
# This script evaluates machine translation models on various translation tasks using torchrun for distributed evaluation.
#
# Usage:
# ./evaluate_mt.sh <model_name_or_path> <numshot> <batch_size> [<N_GPUS>]
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dir      : The directory where csv output result are saved.
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#   batch_size         : The batch size for evaluation.
#   N_GPUS (optional)  : The number of GPUs to use for distributed evaluation. Defaults to 1 if not specified.
#
# Example:
#   ./evaluate_mt.sh my_model 5 16 2
#   This will evaluate the model 'my_model' using 5-shot examples, a batch size of 16, and 2 GPUs.
#
# Tasks:
# This script supports the following tasks:
#  ejmmt (English-Japanese and Japanese-English translation tasks)

n_gpus_default=$(nvidia-smi -L | wc -l)

# Arguments
model_name_or_path=$1
model_log_dir=$2
fewshot_size=${3:-1}
batch_size=${4:-1}
n_gpus=${5:-$n_gpus_default}

# Semi-fixed variables
max_new_tokens=256

# Fixed variables
# Translation templates
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

export TOKENIZERS_PARALLELISM=false

# Function to run torchrun for MT tasks
function torchrun_mt {
    task=$1
    translation=$2
    template=$3
    log_file=$4

    torchrun --nproc_per_node="${n_gpus}" \
        evaluate_mt.py \
        --model_name_or_path "${model_name_or_path}" \
        --task "${task}" \
        --template_name "${template}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${fewshot_size}" \
        --translation "${translation}" \
        --max_new_tokens "${max_new_tokens}" \
        --result_csv "$log_file"
}

# Prepare
if [ ! -d "$model_log_dir" ] ;then
    echo "$model_log_dir does not exits. Create the directory: $model_log_dir"
    mkdir -p "$model_log_dir"
fi

# Main loop
for _template in "${ejmmt_en2ja_template[@]}"; do
    torchrun_mt "ejmmt" 'english=>japanese' "$_template" "$model_log_dir/ejmmt_e2j-${_template}.csv"
done
for _template in "${ejmmt_ja2en_template[@]}"; do
    torchrun_mt "ejmmt" 'japanese=>english' "$_template" "$model_log_dir/ejmmt_j2e-${_template}.csv"
done
