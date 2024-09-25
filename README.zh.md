## Med-Eval
[English](README.md) | [日本語](README.ja.md) | 中文
### 贡献者
+ Junfeng JIANG: [a412133593@gmail.com](mailto:a412133593@gmail.com)
+ 翻译助手：GPT-4


### 准备
```
pip install sacrebleu[ja]
```

### 介绍
这是JMed-LLM仓库的一个子模块，具有类似但更灵活的框架[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。

lm-evaluation-harness是一个常用的用于评估语言模型的库，尤其在选择题(MCQA)任务上，通过计算每个选项的条件概率进行评估。但是，很多情况下它无法灵活地支持模型的评估：
1. 当我们想用不同的模板(提示)来评估一个任务时，需要修改每一个任务定义的源代码。
2. 当我们想要对一个私有任务进行评估时，定义较为困难。
3. 我此前使用的版本无法支持使用多个GPU进行评估。

考虑到这些问题，我开发了这个子模块，以更灵活的方式支持模型的评估。

### Pipeline流程
`EvaluationPipeline`是此子模块中的核心类，用于对不同的任务进行模型评估。该pipeline包括以下步骤：
1. 设置环境，使用单个或多个GPU以及PyTorch DDP。
2. 加载模型和tokenizer。
3. 以特定格式加载数据集（dataclass列表：`MCQASample`）。
4. 根据提供的模板，准备所有的请求并计算损失。
5. 从所有GPU中收集损失，计算最终的度量值。
    + 由于我们使用的是DDP，一些请求可能被计算多次，由于精度的原因，损失可能无法一致。因此，我们将其平均作为最终损失。

### 必要的库
+ 继承自JMed-LLM仓库

### 使用指南
#### 支持的任务
1. 多项问答（MCQA）任务
    + [x] `medmcqa`: MedMCQA
    + [x] `medmcqa_jp`: MedMCQA-JP
    + [x] `usmleqa`: USMLE-QA (4个选项)
    + [x] `usmleqa_jp`: USMLE-QA-JP (4个选项)
    + [x] `medqa`: Med-QA (5个选项)
    + [x] `medqa_jp`: Med-QA-JP (5个选项)
    + [x] `pubmedqa`: PubMedQA
    + [x] `pubmedqa_jp`: PubMedQA-JP [仅支持0-shot]
    + [x] `igakuqa`: IgakuQA (5-6个选项) [仅支持0-shot]
    + [x] `igakuqa_en`: IgakuQA-EN (5-6个选项) [仅支持0-shot]
    + [x] `mmlu`: MMLU
    + [x] `mmlu_medical`: MMLU-Medical
     + 一些医学相关的子集。
    + [x] `mmlu_medical_jp`: MMLU-Medical-JP
2. 机器翻译（MT）任务
    + [x] `ejmmt`: EJMMT (en->ja, ja->en)
3. 命名实体识别（NER）任务
    + [X] `bc2gm`
    + [X] `bc5chem`
    + [X] `bc5disease`
    + [X] `jnlpba`
    + [X] `ncbi_disease`
4. 自然语言推理（NLI）任务和事实验证（Fact Verification）任务
    + [X] `MediQA-RQE`
    + [X] `PubHealth`
    + [X] `HealthVer`

#### 支持的模板
1. MCQA模板
    + [x] `mcqa`: MCQA任务的默认模板。
    + [x] `mcqa_with_options`: 明确提供选项的MCQA任务模板。
    + [x] `context_based_mcqa`: 基于上下文的MCQA任务的默认模板。
2. MT模板
3. NER模板
4. NLI模板
    + [X] `standard`: NLI任务的默认模板，事实验证任务也沿用此模板。

* 其他模板可以在`templates/base.py`模块中找到。

#### 如何进行已有任务的评估？
```shell
#!/bin/bash

BASE_PATH="/home/jiang/mainland/med-eval"   # 修改这个为你自己的路径
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_NODE=${1:-1}                              # 评估的GPU数量
MASTER_PORT=${10:-2333}

model_name_or_path=${2:-"gpt2"}             # HF model 名称或者 checkpoint目录

task=${3:-"medmcqa"}                        # 例子: medmcqa / medmcqa,pubmedqa (同时评估多个任务)
template=${4:-"mcqa_with_options"}          # 例子: mcqa / mcqa_with_options,context_based_mcqa
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

#### 如何定义新任务？
1. 转到`tasks/base.py`模块。
2. 定义一个函数以特定格式加载数据集。
   + 输出: Dict[str, List[MCQASample]]
   + 必须包含 "test"。你也可以选择性地包括 "train"，以便进行few-shot评估，或者在运行时打开 `use_fake_demo`开关，使用test set构建fake demo来进行few-shot评估。
