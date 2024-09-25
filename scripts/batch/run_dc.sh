#!/bin/bash

checkpoints=(
#  "meta-llama/Llama-2-7b-hf"
#  "meta-llama/Meta-Llama-3-8B"
#  "google/gemma-7b"
#  "Qwen/Qwen2-7B"
#  "mistralai/Mistral-7B-v0.3"
#  "epfl-llm/meditron-7b"
#  "llm-jp/llm-jp-13b-v2.0"
#  "tokyotech-llm/Swallow-7b-NVE-hf"
#  "Henrychur/MMed-Llama-3-8B"
  "/fast/ytsuta/model/llm-jp-v3-13b-exp4/iter_0494120/"
)

templates=(
    "context_based_mcqa_minimal"
    "dc_with_options"
    "dc_with_options_jp"
    "dc_instructed_jp"
)


for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  for template in "${templates[@]}"; do
    echo "===== Running ${checkpoint} with ${template} on 0-shot ====="
    bash scripts/evaluation/standard_dc_evaluation.sh \
      8 \
      ${checkpoint} \
      "crade,rrtnm,smdis" \
      ${template} 1 0 42 -1 False 2333 > "logs/dc/${checkpoint_name}_${template}_0-shot.log"
  done
done


for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  for template in "${templates[@]}"; do
    echo "===== Running ${checkpoint} with ${template} on 3-shot ====="
    bash scripts/evaluation/standard_dc_evaluation.sh \
      8 \
      ${checkpoint} \
      "crade,rrtnm,smdis" \
      ${template} 1 3 42 -1 False 2333 > "logs/dc/${checkpoint_name}_${template}_3-shot.log"
  done
done

