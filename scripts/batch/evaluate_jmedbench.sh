#!/bin/bash

BASE_PATH="/home/jiang/mainland/nii-nlp/med-eval"  # TODO: Change this to your own path
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

model_name_or_path=${1:-"meta-llama/Llama-2-7b-hf"}
batch_size=${2:-4}
num_fewshot=${6:-0}

## MCQA
torchrun --nproc_per_node=8 \
         --master_port 2333 \
          "${BASE_PATH}/evaluate_mcqa.py" \
            --model_name_or_path ${model_name_or_path} \
            --task "igakuqa,jmmlu_medical,medmcqa_jp,usmleqa_jp,medqa_jp,mmlu_medical_jp" \
            --template_name "mcqa_with_options_jp" \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed 42 \
            --model_max_length -1 \
            --truncate False

torchrun --nproc_per_node=8 \
         --master_port 2333 \
          "${BASE_PATH}/evaluate_mcqa.py" \
            --model_name_or_path ${model_name_or_path} \
            --task "pubmedqa_jp" \
            --template_name "context_based_mcqa_jp" \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed 42 \
            --model_max_length -1 \
            --truncate False


## NER
torchrun --nproc_per_node=8 \
         --master_port 2333 \
         "${BASE_PATH}/evaluate_ner.py" \
           --model_name_or_path ${model_name_or_path} \
           --max_new_tokens 128 \
           --task "mrner_disease,mrner_medicine,nrner,bc2gm_jp,bc5chem_jp,jnlpba_jp,bc5disease_jp,ncbi_disease_jp" \
           --template_name "standard" \
           --batch_size ${batch_size} \
           --num_fewshot ${num_fewshot} \
           --seed 42 \
           --truncate False

## MT
torchrun --nproc_per_node=8 \
         --master_port 2333 \
           "${BASE_PATH}/evaluate_mt.py" \
             --model_name_or_path ${model_name_or_path} \
             --max_new_tokens 256 \
             --task "ejmmt" \
             --translation "english=>japanese" \
             --template_name "english_japanese" \
             --batch_size ${batch_size} \
             --num_fewshot ${num_fewshot} \
             --seed 42

torchrun --nproc_per_node=8 \
         --master_port 2333 \
           "${BASE_PATH}/evaluate_mt.py" \
             --model_name_or_path ${model_name_or_path} \
             --max_new_tokens 256 \
             --task "ejmmt" \
             --translation "japanese=>english" \
             --template_name "japanese_english" \
             --batch_size ${batch_size} \
             --num_fewshot ${num_fewshot} \
             --seed 42

## DC
torchrun --nproc_per_node=8 \
         --master_port 2333 \
          "${BASE_PATH}/evaluate_mcqa.py" \
            --model_name_or_path ${model_name_or_path} \
            --task "crade,rrtnm,smdis" \
            --template_name "dc_with_options_jp" \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed 42 \
            --model_max_length -1 \
            --truncate True

## STS
torchrun --nproc_per_node=8 \
         --master_port 2333 \
          "${BASE_PATH}/evaluate_sts.py" \
            --model_name_or_path ${model_name_or_path} \
            --task "jcsts" \
            --template_name "sts_as_nli_jp" \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed 42 \
            --model_max_length -1 \
            --truncate False \
            --nli_labels "0,1,2,3,4,5"
