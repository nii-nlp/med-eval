#!/bin/bash

checkpoints=(
   "meta-llama/Llama-2-7b-hf"
   "meta-llama/Meta-Llama-3-8B"
   "Qwen/Qwen2-7B"
   "mistralai/Mistral-7B-v0.3"
   "epfl-llm/meditron-7b"
   "llm-jp/llm-jp-13b-v2.0"
   "tokyotech-llm/Swallow-7b-NVE-hf"
   "Henrychur/MMed-Llama-3-8B"
   # "google/gemma-7b"
)

templates=(
    "context_based_mcqa_minimal"
    "context_based_mcqa"
    "context_based_mcqa_jp"
    "context_based_mcqa_instructed_jp"
)


for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  for template in "${templates[@]}"; do
    echo "===== Running ${checkpoint} with ${template} on 0-shot (PubMedQA) ====="
    bash scripts/evaluation/standard_mcqa_evaluation.sh \
      8 \
      ${checkpoint} \
      "pubmedqa_jp" \
      ${template} 1 0 42 -1 False 2333 > "logs/${checkpoint_name}_${template}_pubmedqa_jp_0-shot.log"
  done
done


for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  for template in "${templates[@]}"; do
    echo "===== Running ${checkpoint} with ${template} on 3-shot (PubMedQA) ====="
    bash scripts/evaluation/standard_mcqa_evaluation.sh \
      8 \
      ${checkpoint} \
      "pubmedqa_jp" \
      ${template} 1 3 42 -1 False 2333 > "logs/${checkpoint_name}_${template}_pubmedqa_jp_3-shot.log"
  done
done

