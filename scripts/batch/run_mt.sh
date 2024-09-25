#!/bin/bash

checkpoints=(
#  "meta-llama/Llama-2-7b-hf"
#  "meta-llama/Meta-Llama-3-8B"
#  "google/gemma-7b"
#  "Qwen/Qwen2-7B"
#  "mistralai/Mistral-7B-v0.3"
#  "epfl-llm/meditron-7b"
  "/fast/ytsuta/model/llm-jp-v3-13b-exp4/iter_0494120/"
#  "tokyotech-llm/Swallow-7b-NVE-hf"
#  "Henrychur/MMed-Llama-3-8B"
)


for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  echo "===== Running ${checkpoint} with all possible templates on 3-shot (e2j) ====="
  bash scripts/evaluation/standard_mt_evaluation.sh \
    8 \
    ${checkpoint} \
    "ejmmt" \
    "english=>japanese" \
    "mt_minimal,mt_english_centric_e2j,english_japanese,mt_instructed_e2j" \
    2 42 3 -1 False 2333 > "logs/mt/${checkpoint_name}_mt_e2j_3-shot.log"
done

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  echo "===== Running ${checkpoint} with all possible templates on 3-shot (e2j) ====="
  bash scripts/evaluation/standard_mt_evaluation.sh \
    8 \
    ${checkpoint} \
    "ejmmt" \
    "japanese=>english" \
    "mt_minimal,mt_english_centric_j2e,japanese_english,mt_instructed_j2e" \
    2 42 3 -1 False 2333 > "logs/mt/${checkpoint_name}_mt_j2e_3-shot.log"
done
