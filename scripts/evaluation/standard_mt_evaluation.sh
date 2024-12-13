#!/bin/bash

BASE_PATH="/home/jiang/mainland/nii-nlp/med-eval"
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_GPU=${1:-1}
MASTER_PORT=${9:-2333}

model_name_or_path=${2:-"gpt2"}

task=${3:-"ejmmt"}

translation=${4:-"english=>japanese"}

template_name=${5:-"few-shot"}

batch_size=${6:-1}
seed=${7:-42}
num_fewshot=${8:-0}

torchrun --nproc_per_node=${N_GPU} \
           "${BASE_PATH}/evaluate_mt.py" \
             --model_name_or_path ${model_name_or_path} \
             --max_new_tokens 256 \
             --task ${task} \
             --translation ${translation} \
             --template_name ${template_name} \
             --batch_size ${batch_size} \
             --num_fewshot ${num_fewshot} \
             --seed ${seed}


# bash scripts/evaluation/standard_mt_evaluation.sh 8 meta-llama/Llama-2-7b-hf ejmmt "english=>japanese,japanese=>english" few-shot 8 42
