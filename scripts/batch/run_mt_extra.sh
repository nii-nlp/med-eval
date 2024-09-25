#!/bin/bash

checkpoints=(
  "/fast/ytsuta/model/llm-jp-v3-13b-exp4/iter_0494120/"
)


for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  echo "===== Running ${checkpoint} with all possible templates on 0-shot (e2j) ====="
  bash scripts/evaluation/standard_mt_evaluation.sh \
    8 \
    ${checkpoint} \
    "ejmmt" \
    "english=>japanese" \
    "mt_minimal,mt_english_centric_e2j,english_japanese,mt_instructed_e2j" \
    4 42 0 -1 False 2333 > "logs/mt/${checkpoint_name}_mt_e2j_0-shot.log"
done

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(echo "$checkpoint" | sed 's/\//-/g')
  echo "===== Running ${checkpoint} with all possible templates on 0-shot (e2j) ====="
  bash scripts/evaluation/standard_mt_evaluation.sh \
    8 \
    ${checkpoint} \
    "ejmmt" \
    "japanese=>english" \
    "mt_minimal,mt_english_centric_j2e,japanese_english,mt_instructed_j2e" \
    4 42 0 -1 False 2333 > "logs/mt/${checkpoint_name}_mt_j2e_0-shot.log"
done
