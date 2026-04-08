import argparse
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
import copy
import itertools
import random
from pathlib import  Path
from typing import Union, List, Optional
import ujson as json
from distutils.util import strtobool
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.distributed as dist
import warnings
warnings.filterwarnings("once")

from tool_utils import is_main_process, main_print, show_pretty_table, output_as_csv
from data_loaders.base import load_mcqa_samples
from tasks.mcqa import MCQASample, MCQARequestDataset
from data_utils import LMDataCollatorForPerplexity
from pipeline import EvaluationPipeline
from templates.jmedbench import UnifiedTemplate


def _cuda_indices_from_hf_device_map(model) -> List[int]:
    """从 device_map 加载的模型上解析实际占用的 CUDA 设备下标。"""
    hf_map = getattr(model, "hf_device_map", None) or {}
    indices = set()
    for v in hf_map.values():
        if isinstance(v, int) and v >= 0:
            indices.add(v)
        elif isinstance(v, str) and v.startswith("cuda:"):
            try:
                indices.add(int(v.split(":", 1)[1]))
            except ValueError:
                pass
    return sorted(indices)


def _local_gpu_label(pipeline: "MCQAEvaluationPipeline") -> str:
    dev = pipeline.device
    if dev.type != "cuda":
        return ""
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return f"{idx}: {torch.cuda.get_device_name(idx)}"


def get_gpus_used(pipeline: "MCQAEvaluationPipeline") -> List[str]:
    """
    本次评估实际用到的 GPU，每项为 "设备下标: 型号名称"。
    无 CUDA、或纯 CPU 时为 []。
    """
    if not torch.cuda.is_available():
        return []

    use_dm = getattr(pipeline.args, "use_device_map", False)
    if use_dm:
        idxs = _cuda_indices_from_hf_device_map(pipeline.model)
        if idxs:
            return [f"{i}: {torch.cuda.get_device_name(i)}" for i in idxs]
        label = _local_gpu_label(pipeline)
        return [label] if label else []

    if getattr(pipeline, "using_ddp", False) and dist.is_available() and dist.is_initialized():
        local = _local_gpu_label(pipeline)
        ws = dist.get_world_size()
        gathered: List[Optional[str]] = [None] * ws
        dist.all_gather_object(gathered, local)
        seen = set()
        out: List[str] = []
        for s in gathered:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        out.sort(key=lambda x: int(x.split(":", 1)[0]))
        return out

    label = _local_gpu_label(pipeline)
    return [label] if label else []


def output_as_json(
    evaluation_results: dict,
    args: argparse.Namespace,
    output_file: str,
    gpus_used: List[str],
    start_time: float = None,
    end_time: float = None,
) -> None:
    """
    将评估结果与超参数写入 JSON 文件，便于后续处理。

    结构:
    - hyperparameters: 本次运行的所有相关超参数
    - gpus_used: 本次评估占用的 GPU（"下标: 型号" 字符串列表）
    - results: { 任务名: { 模板名: { accuracy, norm_accuracy } } }
    - evaluation_start_time / evaluation_end_time / evaluation_duration_seconds: 评估时间信息
    """
    hyperparameters = {
        "seed": args.seed,
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "template_name": args.template_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_fewshot": args.num_fewshot,
        "use_fake_demo": args.use_fake_demo,
        "use_knn_demo": args.use_knn_demo,
        "knn_data_file": args.knn_data_file,
        "knn_data_dir": args.knn_data_dir,
        "knn_data_template_name": args.knn_data_template_name,
        "retriever_id": args.retriever_id,
        "corpus_filename": args.corpus_filename,
        "model_max_length": args.model_max_length,
        "truncate": args.truncate,
        "use_device_map": args.use_device_map,
    }
    # 过滤掉值为 None 的字段，避免在 JSON 中写入无意义的配置
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}
    output_data = {
        "hyperparameters": hyperparameters,
        "gpus_used": gpus_used,
        "results": dict(evaluation_results),
    }
    if start_time is not None and end_time is not None:
        output_data["evaluation_start_time"] = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()
        output_data["evaluation_end_time"] = datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat()
        output_data["evaluation_duration_seconds"] = round(end_time - start_time, 2)
    with open(output_file, "w", encoding="utf-8") as writer:
        json.dump(output_data, writer, ensure_ascii=False, indent=2)


class MCQAEvaluationPipeline(EvaluationPipeline):
    def __task_specific_preparation__(self):
        self.load_samples_f = load_mcqa_samples
        self.dataset_f = MCQARequestDataset
        self.data_collator_f = LMDataCollatorForPerplexity

    def _loglikelihood_batch(self, input_ids, labels, batch):
        n_batch = batch["input_ids"].size(0)

        lm_logits = self.model(input_ids=input_ids).logits

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses = losses.view(n_batch, -1).sum(dim=-1)

        return losses

    def evaluate(
        self,
        samples: List[MCQASample],
        demo_samples: Union[List[MCQASample], List[List[MCQASample]]] = None,
        template_name: str = None,
        dump_file: str = None
    ):
        try:
            dataset, dataloader = self.prepare_data(samples, demo_samples, template_name)
        except AssertionError:
            main_print("Skip this task due to the lack of samples for few-shot learning.")
            return
        except ValueError as e:
            main_print(f"Skip this task: {e}")
            return
        except Exception as e:
            raise e

        result_collection = []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), disable=not is_main_process()):
                batch = {k: v.to(self.device) if k in ["input_ids", "labels"] else v for k, v in batch.items()}

                losses = self._loglikelihood_batch(batch["input_ids"], batch["labels"], batch)

                for i in range(len(losses)):
                    result_collection.append((
                        batch["request_id"][i],
                        batch["option_id"][i],
                        batch["sample"][i],
                        losses[i].item(),
                        (batch["labels"][i] != -100).sum().item()
                    ))
                    if (batch["labels"][i] != -100).sum().item() == 0 and is_main_process():
                        print(batch["input_ids"][i])
                        print("-----")
                        print(batch["labels"][i])
                        print("-----")
                        print(batch["sample"][i])
                        print("-----")
                        print(losses[i].item())
                        exit(1)

        if self.using_ddp:
            all_result_collection = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(
                all_result_collection,
                result_collection
            )
            all_result_collection = list(itertools.chain(*all_result_collection))

            existed_result = {}
            deduplicated_result_collection = []
            for result in all_result_collection:
                if f"{result[0]}-{result[1]}" not in existed_result:
                    deduplicated_result_collection.append(result)
                    existed_result[f"{result[0]}-{result[1]}"] = result[3]
                else:
                    epsilon = 1e-3
                    if is_main_process():
                        saved_result = existed_result[f"{result[0]}-{result[1]}"]
                        warnings.warn(f"Detected inconsistent results [{saved_result} | {result[3]}] from different processes, but well, let's just average it.")
                        existed_result[f"{result[0]}-{result[1]}"] = (saved_result + result[3]) / 2
                    # assert abs(existed_result[f"{result[0]}-{result[1]}"] - result[2]) < epsilon, f"{existed_result[f'{result[0]}-{result[1]}']} != {result[2]}"
            all_result_collection = deduplicated_result_collection

        else:
            all_result_collection = result_collection

        # IgakuQA has different number of options for each sample
        # assert (len(all_result_collection) == dataset.num_samples * dataset.num_options), f"{len(all_result_collection)} != {dataset.num_samples * dataset.num_options}"

        losses = {k: [1e7 for _ in range(len(v.options))] for k, v in enumerate(dataset.samples)}
        n_valid_tokens = {k: [1e7 for _ in range(len(v.options))] for k, v in enumerate(dataset.samples)}
        request_id2sample = {result[0]: result[2] for result in all_result_collection}

        for request_id, option_id, sample, loss, n_valid_token in all_result_collection:
            losses[request_id][option_id] = loss
            n_valid_tokens[request_id][option_id] = n_valid_token

        predictions = []
        norm_predictions = []
        request_id2prediction = {}
        for k, v in losses.items():
            predictions.append(np.argmin(v))
            request_id2prediction[k] = np.argmin(v)
            try:
                norm_predictions.append(np.argmin([loss / n_valid_tokens[k][i] for i, loss in enumerate(v)]))
            except ZeroDivisionError:
                norm_predictions.append(np.argmin(v))
                if is_main_process():
                    warnings.warn("Error: Some options are missing...")

            if is_main_process() and dump_file:
                # record the results
                with open(dump_file, "a+", encoding="utf-8") as writer:
                    writer.write(json.dumps({
                        "request_id": k,
                        "losses": v,
                        "prediction": np.argmin(v),
                        "sample": request_id2sample[k].to_dict()
                    }) + "\n")

        ground_truths = [sample.answer_idx for sample in dataset.samples]

        accuracy = np.mean(np.array(predictions) == np.array(ground_truths))
        norm_accuracy = np.mean(np.array(norm_predictions) == np.array(ground_truths))

        if is_main_process():
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Norm Accuracy: {norm_accuracy:.4f}")

        return {
            "accuracy": accuracy,
            "norm_accuracy": norm_accuracy
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--task", type=str, default="medmcqa", help="Name of the task or the data_dir of the customized task.")
    parser.add_argument("--template_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--use_fake_demo", type=strtobool, default=False,
                        help="""According to Min et al., EMNLP 2022, we understand that we don't need to use the exact demonstrations from the training set.
Therefore, we use the question from the test set itself, and randomly select an option as the fake answer.
Experiment should that it doesn't affect the performance and even perform similar when we need to find true demos from other similar dataset like MedQA w.r.t. IgakuQA.
In default, we don't use this option, but use the exact demonstrations from the training set""")

    parser.add_argument("--use_knn_demo", type=strtobool, default=False,
                        help="Use pre-retrieved KNN-based few-shot learning for the demonstration.")
    parser.add_argument("--knn_data_file", type=str, default=None)
    parser.add_argument("--knn_data_dir", type=str, default=None)
    parser.add_argument("--knn_data_template_name", type=str, default=None)
    parser.add_argument("--retriever_id", type=str, default=None)
    parser.add_argument("--corpus_filename", type=str, default=None)

    parser.add_argument("--model_max_length", type=int, default=None, help="Maximum length of the model input.")

    parser.add_argument("--truncate", type=strtobool, default=False)
    parser.add_argument("--dump_file", type=str, default=None)
    parser.add_argument("--result_csv", type=str, default=None, help="Deprecated: 建议使用 --result_json 输出 JSON 便于后续处理")
    parser.add_argument("--result_json", type=str, default=None, help="将超参数与各数据集结果保存为 JSON 文件")
    parser.add_argument("--use_device_map", type=strtobool, default=False,
                        help="Use device_map='auto' to distribute model across multiple GPUs (model parallelism)")

    args = parser.parse_args()

    if args.model_max_length == -1:
        args.model_max_length = None
    if args.result_csv:
        parent_path = Path(args.result_csv).parent
        assert parent_path.exists(), f"{parent_path} does not exist. Cannot write output."
    if args.result_json:
        parent_path = Path(args.result_json).parent
        assert parent_path.exists(), f"{parent_path} does not exist. Cannot write output."

    pipeline = MCQAEvaluationPipeline(args)

    # load task
    tasks = args.task.split(",")
    template_names = args.template_name.split(",")
    if len(template_names) == 1:
        template_names = template_names * len(tasks)

    assert len(tasks) == len(template_names), f"Number of tasks and templates should be the same, but got {len(tasks)} != {len(template_names)}"

    evaluation_results = defaultdict(lambda: defaultdict(dict))
    eval_start_time = time.time()
    for task, template_name in zip(tasks, template_names):
        samples = pipeline.load_downstream_task(dataset_name=task)

        # evaluation starts
        # Skip if no test samples available
        if len(samples["test"]) == 0:
            main_print(f"Skipping task {task} due to empty test set.")
            continue

        if args.use_fake_demo:
            ## Reference: Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (Min et al., 2022)
            shuffle_test_samples = copy.deepcopy(samples["test"])
            for j in range(len(shuffle_test_samples)):
                random.shuffle(shuffle_test_samples[j].options)

            evaluation_result = pipeline.evaluate(
                samples["test"],
                demo_samples=shuffle_test_samples if args.num_fewshot > 0 else None,
                template_name=template_name,
                dump_file=args.dump_file
            )

        elif args.use_knn_demo:
            demo_samples = []
            template = UnifiedTemplate()

            train_samples = json.load(open(args.corpus_filename, "r", encoding="utf-8"))["train"] if args.corpus_filename is not None else samples["train"]

            with open(args.knn_data_file) as f:
                for line in f:
                    indices = [int(index) for index in line.strip().split(",")][1:]
                    demo_sample_list = []
                    for i in range(args.num_fewshot):
                        instantiated_sample = template.instantiate_template_full(
                            sample=train_samples[indices[i]],
                            template_name="Standard"
                        )
                        demo_sample_list.append(instantiated_sample)

                    demo_samples.append(demo_sample_list)

            evaluation_result = pipeline.evaluate(
                samples["test"],
                demo_samples=demo_samples,
                template_name=template_name,
                dump_file=args.dump_file
            )

        else:
            # Skip if no test samples available
            if len(samples["test"]) == 0:
                main_print(f"Skipping task {task} due to empty test set.")
                continue

            evaluation_result = pipeline.evaluate(
                samples["test"],
                demo_samples=samples["train"] if args.num_fewshot > 0 else None,
                template_name=template_name,
                dump_file=args.dump_file
            )
        if evaluation_result is None:
            continue
        evaluation_results[task][template_name] = evaluation_result

    eval_end_time = time.time()
    show_pretty_table(evaluation_results)
    if args.result_json:
        gpus_used = get_gpus_used(pipeline)
        if is_main_process():
            output_as_json(
                evaluation_results,
                args,
                args.result_json,
                gpus_used=gpus_used,
                start_time=eval_start_time,
                end_time=eval_end_time,
            )
    if args.result_csv:
        output_as_csv(evaluation_results, args.result_csv)
