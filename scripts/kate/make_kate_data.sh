#!/bin/bash

BASE_PATH="/home/jiang/mainland/med-eval"
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

task_name=${1:-"usmleqa"}
task_category=${2:-"mcqa"}  # nli
template_name=${3:-"mcqa_with_options"}
model_name_or_path=${4:-"facebook/contriever-msmarco"}   # facebook/mcontriever-msmarco
icl_source=${5:-"train"}   # auxiliary_train

python3 "${BASE_PATH}/kate/make_kate_data.py" \
  --batch_size 1024 \
  --model_name_or_path ${model_name_or_path} \
  --template_name ${template_name} \
  --task_name ${task_name} \
  --task_category ${task_category} \
  --icl_source ${icl_source} \
  --dump_file "${BASE_PATH}/dataset/kate/${task_name}_kate.json" \
  --topk 128
