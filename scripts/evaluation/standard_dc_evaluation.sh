#!/bin/bash

BASE_PATH="/home/jiang/mainland/med-eval"
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_GPU=${1:-1}
MASTER_PORT=${10:-2333}

model_name_or_path=${2:-"gpt2"}

task=${3:-"crade,rrtnm,smdis"}
template_name=${4:-"mcqa_with_options"}

batch_size=${5:-4}
num_fewshot=${6:-0}
seed=${7:-42}
model_max_length=${8:--1}
use_knn_demo=${9:-False}

torchrun --nproc_per_node=${N_GPU} \
         --master_port $MASTER_PORT \
          "${BASE_PATH}/evaluate_mcqa.py" \
            --model_name_or_path ${model_name_or_path} \
            --task ${task} \
            --template_name ${template_name} \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed ${seed} \
            --model_max_length ${model_max_length} \
            --truncate True \
            --use_knn_demo ${use_knn_demo}
