#!/bin/bash
# This script evaluates Domain Classification models on various tasks using torchrun for distributed evaluation.
#
# Usage:
# ./evaluate_dc.sh <model_name_or_path> <fewshot_size> <batch_size> [<N_GPUS>]
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dir      : The directory where csv output result are saved.
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#   batch_size         : The batch size for evaluation.
#   N_GPUS (optional)  : The number of GPUs to use for distributed evaluation. Defaults to 1 if not specified.
#
# Example:
#   ./evaluate_dc.sh my_model 5 16 2
#   This will evaluate the model 'my_model' using 5-shot examples, a batch size of 16, and 2 GPUs.
#
# Tasks:
# This script supports the following Domain Classification tasks:
#  crade, rrtnm, smdis

n_gpus_default=$(nvidia-smi -L | wc -l)

# Arguments
model_name_or_path=$1
model_log_dir=$2
fewshot_size=${3:-1}
batch_size=${4:-1}
n_gpus=${5:-$n_gpus_default}

# Semi-fixed variables
model_max_length=-1

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

export TOKENIZERS_PARALLELISM=false

# Function to run torchrun for Domain Classification tasks
function torchrun_dc {
    joined_tasks=$1
    template=$2
    log_file=$3

    torchrun --nproc_per_node="${n_gpus}" \
        evaluate_mcqa.py \
        --model_name_or_path "${model_name_or_path}" \
        --task "${joined_tasks}" \
        --template_name "${template}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${fewshot_size}" \
        --model_max_length "${model_max_length}" \
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
    torchrun_dc "$joined_tasks" "$_template" "$model_log_dir/all-${_template}.csv"
done