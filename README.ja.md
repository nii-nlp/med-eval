## JMed-Eval
[English](README.md) | 日本語 | [中文](README.zh.md)
### 貢献者
+ Junfeng JIANG: [a412133593@gmail.com](mailto:a412133593@gmail.com)
+ 翻訳アシスタント：GPT-4

### 準備
```shell
pip install sacrebleu[ja]
```

### 紹介
これは、JMed-LLMのサブモジュールで、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)と類似した、しかしより柔軟なフレームワークです。

lm-evaluation-harnessは、言語モデルを評価するための広く使用されるライブラリであり、特にマルチチョイス質問応答（MCQA）タスクにおいて、各オプションの条件付き対数尤度を計算することによって評価を行います。しかし、多くのケースにおいてモデルの評価をサポートするための十分な柔軟性がありません：
1. 異なるテンプレート（プロンプト）で一つのタスクを評価したい場合、タスクのソースコードを修正する必要があります。
2. プライベートなタスクで評価を行いたいとき、定義が困難です。
3. 私が使用していたバージョンは、複数のGPUを使用した評価をサポートしていませんでした。

これらの問題を考慮し、より柔軟な方法でモデルの評価をサポートするためのこのサブモジュールを開発しました。

### パイプライン
`EvaluationPipeline`はこのサブモジュールの中核となるクラスで、異なるタスクに対するモデルの評価に使用されます。パイプラインは次のステップで構成されています：
1. 環境を設定し、PyTorch DDPとともに単一または複数のGPUを使用します。
2. モデルとtokenizerを導入します。
3. 特定の形式のデータセットを導入します（dataclassのリスト：`MCQASample`）。
4. 提供されたテンプレートに基づき、すべてのリクエストを準備し、損失を計算します。
5. すべてのGPUから損失を収集し、最終的な指標を計算します。
    + DDPを使用しているため、一部のリクエストは複数回計算され、精度のために損失が同一でない場合があります。そのため、それらを平均して最終的な損失とします。

### ライブラリ
+ JMed-LLMとの同じ

### 指示
#### サポートされるタスク
1. MCQAタスク
    + [x] `medmcqa`: MedMCQA
    + [x] `medmcqa_jp`: MedMCQA-JP
    + [x] `usmleqa`: USMLE-QA（4択）
    + [x] `usmleqa_jp`: USMLE-QA-JP（4択）
    + [x] `medqa`: Med-QA（5択）
    + [x] `medqa_jp`: Med-QA-JP (5択)
    + [x] `pubmedqa`: PubMedQA
    + [x] `pubmedqa_jp`: PubMedQA-JP [Zero-shotのみ]
    + [x] `igakuqa`: IgakuQA（5～6択）[Zero-shotのみ]
    + [x] `igakuqa_en`: IgakuQA-EN（5～6択）[Zero-shotのみ]
    + [x] `mmlu`: MMLU
    + [x] `mmlu_medical`: MMLU-Medical
      + 一部の医学関連のサブセット。
    + [x] `mmlu_medical_jp`: MMLU-Medical-JP
2. MTタスク
    + [x] `ejmmt`: EJMMT（en->ja, ja->en）
3. NERタスク
    + [X] `bc2gm`
    + [X] `bc5chem`
    + [X] `bc5disease`
    + [X] `jnlpba`
    + [X] `ncbi_disease`
4. NLIタスクと事実検証タスク
    + [X] `MediQA-RQE`
    + [X] `PubHealth`
    + [X] `HealthVer`

#### サポートされるテンプレート
1. MCQAテンプレート
    + [x] `mcqa`: MCQAタスクのデフォルトテンプレート。
    + [x] `mcqa_with_options`: オプションを明示的に提供するMCQAタスクのテンプレート。
    + [x] `context_based_mcqa`: コンテキストベースのMCQAタスクのデフォルトテンプレート。
2. MTテンプレート
3. NERテンプレート
4. NLIテンプレート
    + [X] `standard`: NLIタスクのデフォルトテンプレート。事実検証タスクもこのテンプレートを使います。

* その他のテンプレートは`templates/base.py`モジュールから見つけることができます。

#### 定義されたタスクの評価はどのように行いますか？
```shell
#!/bin/bash

BASE_PATH="/home/jiang/mainland/med-eval"   # 自分のpathに変更してください。
export PYTHONPATH=$BASE_PATH
export TOKENIZERS_PARALLELISM=false

N_NODE=${1:-1}                              # 評価に使用されるGPUの数。
MASTER_PORT=${10:-2333}

model_name_or_path=${2:-"gpt2"}             # HFモデルの名前またはチェックポイントのディレクトリ。

task=${3:-"medmcqa"}                        # 例：medmcqa / medmcqa, pubmedqa（同時に複数のタスクを評価します）
template=${4:-"mcqa_with_options"}          # 例：mcqa / mcqa_with_options, context_based_mcqa
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

#### 新しいタスクを定義するには？
1. `tasks/base.py`モジュールに移動します。
2. 特定の形式でデータセットをロードする関数を定義します。
   + 出力：Dict[str, List[MCQASample]]
   + "test"が必要です。しかし、"train"はOptionalです。これがあれば、フューショット評価が可能になります。また、実行時に`use_fake_demo`をオンにするなら、"train"はないでもフューショット評価ができます。
