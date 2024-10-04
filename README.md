# Med-Eval
English | [日本語](README.ja.md) | [中文](README.zh.md)
## Contributors
+ Junfeng JIANG: [a412133593@gmail.com](mailto:a412133593@gmail.com)

## Updates
+ 🎉 2024-08-18: All datasets have been uploaded to the Hugging Face Datasets Hub. You can find them in [Coldog2333/JMedBench](https://huggingface.co/datasets/Coldog2333/JMedBench).

## Installation
### Supported environment
+ Python=3.9
+ Nvidia Driver >= 450.80.02* (for pytorch compiled with cuda11.7)
+ Git LFS isntalled

### Clone the repository and install requrements
```shell
git clone https://github.com/nii-nlp/med-eval.git
cd med-eval
pip install -r requirements.txt
# git lfs install # Make sure you have git-lfs installed (https://git-lfs.com)
git clone https://huggingface.co/datasets/Coldog2333/JMedBench data/JMedBench

```

## Introduction
This is a submodule in the JMed-LLM repository, with a similar but more flexible framework as [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

lm-evaluation-harness is a widely used library for evaluating language models, espscially on Multi-Choice Question Answering (MCQA) tasks, by computing conditional log-likelihoods for each option. However, it is not flexible enough to support the evaluation of models in many cases:
1. When we want to evaluate one task with different templates (prompts), we need to modify the source codes for each task.
2. When we want to evaluate on a local dataset, it is hard to define.
3. The version we were using didn't support evaluation with multiple GPUs.

Considering these issues, we developed this submodule to support the evaluation of models in a more flexible way.


## How to do evaluation on defined tasks?
Here is an example of how to evaluate a model on the `MedMCQA` task with the `mcqa_with_options` template.
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
            --truncate False
```

### Quick evaluation on JMedBench
We also implemented several scripts to evaluate models various supported tasks. You can find them in the `scripts/evaluation` directory.

If you want to do evaluation on [JMedBench](https://huggingface.co/datasets/Coldog2333/JMedBench), you can use the following _one-line_ command:
```shell
bash scripts/evaluation/evaluate_jmedbench.sh ${model_name_or_path}
```

For example, if we want to evaluate Llama2-7B, we can use the following command:
```shell
bash scripts/evaluation/evaluate_jmedbench.sh "meta-llama/Llama-2-7b-hf"
```

After the evaluation, you could collect the results from the standard output.


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

## Tasks and prompt templates
### Supported tasks
1. MCQA tasks
    + [x] `medmcqa`: MedMCQA
    + [x] `medmcqa_jp`: MedMCQA-JP
    + [x] `usmleqa`: USMLE-QA (4 Options)
    + [x] `usmleqa_jp`: USMLE-QA-JP (4 Options)
    + [x] `medqa`: Med-QA (5 Options)
    + [x] `medqa_jp`: Med-QA-JP (5 Options)
    + [x] `pubmedqa`: PubMedQA
    + [x] `pubmedqa_jp`: PubMedQA-JP
    + [x] `igakuqa`: IgakuQA (5-6 options)
    + [x] `igakuqa_en`: IgakuQA-EN (5-6 options)
    + [x] `mmlu`: MMLU [Zero-shot only]
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
    + [X] `bc2gm_jp`: BC2GM from BLURB
    + [X] `bc5chem_jp`: BC5Chem from BLURB
    + [X] `bc5disease_jp`: BC5Disease from BLURB
    + [X] `jnlpba_jp`: JNLPBA from BLURB
    + [X] `ncbi_disease_jp`: NCBI-Disease from BLURB
4. Document Classification
    + [X] `crade`
    + [X] `rrtnm`
    + [X] `smdis`
5. Semantic Text Similarity
    + [X] `jcsts`: Japanese Clinical Semantic Text Similarity

### Supported prompt templates
For each tasks, there are four prompt templates:
+ `Minimal`: Only display a question.
+ `Standard`: Display a question with a brief explanation in japanese.
+ `English Centric`: `standard` with english explanation.
+ `Instrcuted`: Display a question with a detail instuction in japanese.


|Task|Minimal|Standard|English Centric|Instrcuted|
|---|---|---|---|---|
|MCQA (except `pubmedqa*`)|`mcqa_minimal`|`mcqa_with_options_jp`|`mcqa_with_options`|`4o_mcqa_instructed_jp`|
|MCQA (`pubmedqa*`)|`context_based_mcqa`|`context_based_mcqa_minimal`|`context_based_mcqa_jp`|`context_based_mcqa_instructed_jp`|
|MT (ja-en)|`mt_minimal`|`english_japanese`|`mt_english_centric_e2j`|`mt_instructed_e2j`|
|MT (en-ja)|`mt_minimal`|`japanese_english`|`mt_english_centric_j2e`|`mt_instructed_j2e`|
|NER|`minimal`|`standard`|`english-centric`|`instructed`|
|DC|`context_based_mcqa_minimal`|`dc_with_options_jp`|`dc_with_options`|`dc_instructed_jp`|
|STS|`sts_minimal`|`sts_as_nli_jp`|`sts_as_nli`|`sts_instructed_jp`|

See `template` directory for details.<br>
Other templates can be found in the `templates` module.

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

## Citation
If you find this code helpful for your research, please cite the following paper:
```bibtex
@misc{jiang2024jmedbenchbenchmarkevaluatingjapanese,
      title={JMedBench: A Benchmark for Evaluating Japanese Biomedical Large Language Models}, 
      author={Junfeng Jiang and Jiahao Huang and Akiko Aizawa},
      year={2024},
      eprint={2409.13317},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.13317}, 
}
```
