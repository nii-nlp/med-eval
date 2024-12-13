#!/bin/bash

BASE_PATH="/home/jiang/mainland/nii-nlp/med-eval"
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_NODE=${1:-1}
MASTER_PORT=${9:-2333}

model_name_or_path=${2:-"gpt2"}

task=${3:-"bc5disease_jp"}  # bc2gm,bc5chem,jnlpba,bc5disease,ncbi_disease
# mrner_disease,mrner_medicine,nrner
template_name=${4:-"standard"}

batch_size=${5:-8}
num_fewshot=${6:-0}
seed=${7:-42}

use_knn_demo=${8:-False}

torchrun --nproc_per_node=${N_NODE} \
         --master_port $MASTER_PORT \
         "${BASE_PATH}/evaluate_ner.py" \
           --model_name_or_path ${model_name_or_path} \
           --max_new_tokens 128 \
           --task ${task} \
           --template_name ${template_name} \
           --batch_size ${batch_size} \
           --num_fewshot ${num_fewshot} \
           --seed ${seed} \
           --truncate False \
           --use_knn_demo ${use_knn_demo}


# bash scripts/evaluation/standard_ner_evaluation.sh 8 meta-llama/Llama-2-7b-hf mrner_medicine,mrner_disease,nrner,bc2gm_jp,bc5chem_jp,jnlpba_jp,bc5disease_jp,ncbi_disease_jp standard 4 0 42
