#!/bin/bash

checkpoints=(
#  "meta-llama/Llama-2-7b-hf"
#  "google/gemma-7b"
#  "epfl-llm/meditron-7b"
#  "Qwen/Qwen2-7B"
#  "mistralai/Mistral-7B-v0.3"
#  "tokyotech-llm/Swallow-7b-NVE-hf"

  # "meta-llama/Meta-Llama-3-8B"
  "/fast/ytsuta/model/llm-jp-v3-13b-exp4/iter_0494120/"
#   "Henrychur/MMed-Llama-3-8B"
)


for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  echo "===== Running ${checkpoint} with all possible templates on 0-shot ====="
  bash scripts/evaluation/standard_sts_evaluation.sh \
    8 \
    ${checkpoint} \
    "jcsts" \
    "sts_minimal,sts_as_nli,sts_as_nli_jp,sts_instructed_jp" \
    "0,1,2,3,4,5" \
    2 0 42 -1 False 2333 > "logs/sts/${checkpoint_name}_sts_0-shot.log"
done

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  echo "===== Running ${checkpoint} with all possible templates on 3-shot ====="
  bash scripts/evaluation/standard_sts_evaluation.sh \
    8 \
    ${checkpoint} \
    "jcsts" \
    "sts_minimal,sts_as_nli,sts_as_nli_jp,sts_instructed_jp" \
    "0,1,2,3,4,5" \
    2 3 42 -1 False 2333 > "logs/sts/${checkpoint_name}_sts_3-shot.log"
done