#!/bin/bash

BASE_PATH="/home/jiang/mainland/med-eval"
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_GPU=${1:-1}
MASTER_PORT=${11:-2333}

model_name_or_path=${2:-"gpt2"}

task=${3:-"jcsts"}
template_name=${4:-"sts_as_nli"}
#nli_labels=${5:-"二つの文は完全に似ていない。,二つの文は同等ではないが、同じトピックに関している。,二つの文は同等ではないが、いくつかの詳細を共有している。,二つの文は大まかに同等だが、いくつかの重要な情報が異なっていたり欠けている。,二つの文はほぼ同等だが、いくつかの重要ではない詳細が異なる。,二つの文は完全に同等で、意味が同じである。"}
nli_labels=${5:-"0,1,2,3,4,5"}
batch_size=${6:-4}
num_fewshot=${7:-0}
seed=${8:-42}
model_max_length=${9:--1}
use_knn_demo=${10:-False}

torchrun --nproc_per_node=${N_GPU} \
         --master_port $MASTER_PORT \
          "${BASE_PATH}/evaluate_sts.py" \
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
