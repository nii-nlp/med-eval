#!/bin/bash

BASE_PATH="/home/jiang/mainland/med-eval"
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_GPU=${1:-1}
MASTER_PORT=${11:-2333}

model_name_or_path=${2:-"gpt2"}

task=${3:-"mediqa_rqe"}
template_name=${4:-"standard"}
nli_labels=${5:-"No,Yes"}         # PubHealth: "No,Yes,Mixture,Unproven"    Healthver: "Contradict,Supported,Not-enough-information"
batch_size=${6:-32}
num_fewshot=${7:-0}
seed=${8:-42}
model_max_length=${9:--1}
use_knn_demo=${10:-False}

torchrun --nproc_per_node=${N_GPU} \
         --master_port $MASTER_PORT \
          "${BASE_PATH}/evaluate_nli.py" \
            --model_name_or_path ${model_name_or_path} \
            --task ${task} \
            --template_name ${template_name} \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed ${seed} \
            --model_max_length ${model_max_length} \
            --truncate False \
            --nli_labels ${nli_labels} \
            --use_knn_demo ${use_knn_demo}


## Zero-shot
# LinkLM     0.4913  (+1.74%)
# Standard   0.4739


## Few-shot
# LinkLM     0.5087   0.5435    0.5304    0.5275  (+1.3%)
# Standard   0.5      0.5217    0.5217    0.5145
