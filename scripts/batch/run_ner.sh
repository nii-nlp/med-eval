#!/bin/bash

checkpoints=(
  "meta-llama/Llama-2-7b-hf"
)

#checkpoints=(
#  "meta-llama/Llama-2-7b-hf"
#  "tokyotech-llm/Swallow-7b-hf"
#  "epfl-llm/meditron-7b"
#  "meta-llama/Meta-Llama-3-8B"
#)


for checkpoint in "${checkpoints[@]}"; do
  bash scripts/evaluation/standard_ner_evaluation.sh \
    8 \
    ${checkpoint} \
    "mrner_disease,mrner_medicine,nrner" \
    "standard" 4 0 42 False 2333
done