# Med-Eval
English | [日本語](README.ja.md) | [中文](README.zh.md)
## Contributors
+ Junfeng JIANG: [a412133593@gmail.com](mailto:a412133593@gmail.com)

## Updates
+ 🎉 2024-08-18: All datasets have been uploaded to the Hugging Face Datasets Hub. You can find them in [Coldog2333/JMedBench](https://huggingface.co/datasets/Coldog2333/JMedBench).

## Installation
### Clone the repository
```shell
git clone https://github.com/nii-nlp/med-eval.git
cd med-eval
```

### Create conda environment
```shell
conda create -n med-eval python=3.9
```

### Preliminary
#### PyTorch
We recommend the following installation command for PyTorch since we only verify our codes with PyTorch 1.13.1 + CUDA 11.7. You can find more information on the [official website](https://pytorch.org/).
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
#### Others
```shell
pip install -r requirements.txt
pip install sacrebleu[ja]
# rollback numpy to 1.X
pip install numpy==1.26.4
```


## Introduction
This is a submodule in the JMed-LLM repository, with a similar but more flexible framework as [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

lm-evaluation-harness is a widely used library for evaluating language models, espscially on Multi-Choice Question Answering (MCQA) tasks, by computing conditional log-likelihoods for each option. However, it is not flexible enough to support the evaluation of models in many cases:
1. When we want to evaluate one task with different templates (prompts), we need to modify the source code for each task.
2. When we want to evaluate on a private task, it is hard to define.
3. The version I was using doesn't support evaluation with multiple GPUs.

Considering these issues, I developed this submodule to support the evaluation of models in a more flexible way.

## Pipeline
`EvaluationPipeline` is the core class in this submodule, which is used to evaluate models on different tasks. The pipeline consists of the following steps:
1. Setup the environment, using single or multiple GPUs with PyTorch DDP.
2. Load the model and tokenizer.
3. Load the dataset in a specific format (list of dataclass: `MCQASample`).
4. Based on the given template, prepare all the requests and compute losses.
5. Collect the losses from all GPUs and compute the final metrics.
    + Since we use DDP, some requests will be computed for multiple times and the losses may not be the same due to the precision. Therefore, we average them as the final loss.

## Prerequisites
+ Inherit from the JMed-LLM repository

## Instructions
### Supported tasks
1. MCQA tasks
    + [x] `medmcqa`: MedMCQA
    + [x] `medmcqa_jp`: MedMCQA-JP
    + [x] `usmleqa`: USMLE-QA (4 Options)
    + [x] `usmleqa_jp`: USMLE-QA-JP (4 Options)
    + [x] `medqa`: Med-QA (5 Options)
    + [x] `medqa_jp`: Med-QA-JP (5 Options)
    + [x] `pubmedqa`: PubMedQA
    + [x] `pubmedqa_jp`: PubMedQA-JP [Zero-shot only]
    + [x] `igakuqa`: IgakuQA (5-6 options) [Zero-shot only]
    + [x] `igakuqa_en`: IgakuQA-EN (5-6 options) [Zero-shot only]
    + [x] `mmlu`: MMLU
    + [x] `mmlu_medical`: MMLU-Medical
      + Some medical related subsets.
    + [x] `mmlu_medical_jp`: MMLU-Medical-JP
    + [x] `jmmlu`: JMMLU
    + [x] `jmmlu_medical`: JMMLU-Medical
2. MT tasks
    + [x] `ejmmt`: EJMMT (en->ja, ja->en)
3. NER tasks
    + [X] `mrner_disease`: MRNER-Disease from JMED-LLM
    + [X] `mrner_medicine`: MRNER-Medicine from JMED-LLM
    + [X] `nrner`: NRNER from JMED-LLM
    + [X] `bc2gm`: BC2GM from BLURB
    + [X] `bc5chem`: BC5Chem from BLURB
    + [X] `bc5disease`: BC5Disease from BLURB
    + [X] `jnlpba`: JNLPBA from BLURB
    + [X] `ncbi_disease`: NCBI-Disease from BLURB
4. NLI tasks and Fact Verification tasks
    + [X] `MediQA-RQE`
    + [X] `PubHealth`
    + [X] `HealthVer`
5. Document Classification
    + [X] `crade`
    + [X] `rrtnm`
    + [X] `smdis`
6. Semantic Text Similarity
    + [X] `jcsts`: Japanese Clinical Semantic Text Similarity

### Supported templates
1. MCQA templates
    + [x] `mcqa`: Default template for MCQA tasks.
    + [x] `mcqa_with_options`: Template for MCQA tasks providing options explicitly.
    + [x] `context_based_mcqa`: Default template for context-based MCQA tasks.
2. MT templates
3. NER templates
4. NLI templates
   + [X] `standard`: Default template for NLI tasks. Fact Verification tasks also share this template.
5. DC templates
   + [X] `mcqa_with_options`: DC task can be reformulated as MCQA task.
6. STS templates
   + [X] `sts_as_nli`: STS task can be reformulated as NLI task.


* Other templates can be found in the `templates` module.


### How to do evaluation on defined tasks?
```shell
#!/bin/bash

BASE_PATH="/home/jiang/mainland/med-eval"   # Change this to your own path
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_NODE=${1:-1}                              # Number of GPUs for evaluation
MASTER_PORT=${10:-2333}

model_name_or_path=${2:-"gpt2"}             # HF model name or checkpoint dir

task=${3:-"medmcqa"}                        # example: medmcqa / medmcqa,pubmedqa (evaluate multiple tasks at the same time)
template=${4:-"mcqa_with_options"}          # example: mcqa / mcqa_with_options,context_based_mcqa
batch_size=${5:-32}
num_fewshot=${6:-0}
seed=${7:-42}
model_max_length=${8:--1}
use_knn_demo=${9:-False}

torchrun --nproc_per_node=${N_GPU} \
         --master_port $MASTER_PORT \
          "${BASE_PATH}/evaluate_mcqa.py" \
            --model_name_or_path ${model_name_or_path} \
            --task ${task} \
            --template_name ${template_name} \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed ${seed} \
            --model_max_length ${model_max_length} \
            --truncate False \
            --use_knn_demo ${use_knn_demo}
```

### How to define a new task?
1. Go to the `tasks/base.py` module.
2. Define a function to load the dataset in a specific format.
   + output: Dict[str, List[MCQASample]]
   + MUST include "test" key. Optionally, you can include "train" keys for few-shot evaluation, or you could turn on `use_fake_demo` when running.


## Appendix
### Statistics

| MCQA              | #Train  | #Test | 
|-------------------|---------|-------|
| MedMCQA (jp)      | 182,822 | 4,183 | 
| USMLE-QA (jp)     | 10,178  | 1,273 | 
| MedQA (jp)        | 10,178  | 1,273 | 
| MMLU-medical (jp) | 45      | 1,871 | 
| JMMLU-medical (jp)| -       | 1,271 |
| IgakuQA           | -       | 1,600 | 
| PubMedQA  (jp)    | 211,269 | 1,000 | 

| NLI      | #Train  | #Test  |
|--------------|---------|--------|
| PubHealth    | 9,804   | 1,231  |
| HealthVer    | 5,292   | 903    |
| MediQA-RQE   | 8,588   | 230    |

| MT           | #Train  | #Test  |
|--------------|---------|--------|
| EJMMT        | -       | 2,480  |

| NER          | #Train  | #Test  |
|--------------|---------|--------|
| BC2GM        | 12,574  | 5,038  |
| BC5Chem      | 4,560   | 4,797  |
| BC5Disease   | 4,560   | 4,797  |
| JNLPBA       | 18,607  | 4,260  |
| NCBI-Disease | 5,424   | 940    |

| DC           | #Train | #Test |
|--------------|--------|-------|
| CRaDE        | -      | 100   |
| RRTNM        | -      | 100   |
| SMDIS        | -      | 100   |

| STS          | #Train | #Test |
|--------------|--------|-------|
| JCSTS        | -      | 3,670 |


### TODO
+ Summarization