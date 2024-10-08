#!/bin/bash
# This script evaluates Named Entity Recognition (NER) models on various NER tasks using torchrun for distributed evaluation.
#
# Usage:
# ./evaluate_ner.sh <model_name_or_path> <numshot> <batch_size> [<N_GPUS>]
#
# Arguments:
#   model_name_or_path : The path or name (on Huggingface) of the model to evaluate.
#   model_log_dir      : The directory where csv output result are saved.
#   fewshot_size       : The number of few-shot examples to use for evaluation.
#   batch_size         : The batch size for evaluation.
#   N_GPUS (optional)  : The number of GPUs to use for distributed evaluation. Defaults to 1 if not specified.
#
# Example:
#   ./evaluate_ner.sh my_model 5 16 2
#   This will evaluate the model 'my_model' using 5-shot examples, a batch size of 16, and 2 GPUs.
#
# Tasks:
# This script supports the following NER tasks:
#  mrner_disease, mrner_medicine, nrner, bc2gm_jp, bc5chem_jp, bc5disease_jp, jnlpba_jp, ncbi_disease_jp

n_gpus_default=$(nvidia-smi -L | wc -l)

# Arguments
model_name_or_path=$1
model_log_dir=$2
fewshot_size=${3:-1}
batch_size=${4:-1}
n_gpus=${5:-$n_gpus_default}

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

export TOKENIZERS_PARALLELISM=false

# Function to run torchrun for NER tasks
function torchrun_ner {
    joined_tasks=$1
    template=$2
    log_file=$3

    torchrun --nproc_per_node="${n_gpus}" \
        evaluate_ner.py \
        --model_name_or_path "${model_name_or_path}" \
        --task "${joined_tasks}" \
        --template_name "${template}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${fewshot_size}" \
        --max_new_tokens "${max_new_tokens}" \
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
    torchrun_ner "$joined_tasks" "$_template" "$model_log_dir/all-${_template}.csv"
done
