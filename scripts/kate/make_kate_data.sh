#!/bin/bash

BASE_PATH="/home/jiang/mainland/nii-nlp/med-eval"
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

task_name=${1:-"usmleqa"}
task_category=${2:-"mcqa"}  # nli
template_name=${3:-"mcqa_with_options"}
model_name_or_path=${4:-"facebook/contriever-msmarco"}   # facebook/mcontriever-msmarco
batch_size=${5:-1024}
topk=${6:-128}

python3 "${BASE_PATH}/kate/make_kate_data.py" \
  --batch_size ${batch_size} \
  --model_name_or_path ${model_name_or_path} \
  --template_name ${template_name} \
  --task_name ${task_name} \
  --task_category ${task_category} \
  --topk ${topk} \
  --output_cache_dir "${BASE_PATH}/cache"
