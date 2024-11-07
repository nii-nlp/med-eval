import argparse
from collections import defaultdict
import copy
import itertools
import random
from pathlib import  Path
from typing import Union, List
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

    parser.add_argument("--model_max_length", type=int, default=None, help="Maximum length of the model input.")

    parser.add_argument("--truncate", type=strtobool, default=False)
    parser.add_argument("--dump_file", type=str, default=None)
    parser.add_argument("--result_csv", type=str, default=None)

    args = parser.parse_args()

    if args.model_max_length == -1:
        args.model_max_length = None
    if args.result_csv is not None:
        parent_path = Path(args.result_csv).parent.exists()
        assert parent_path, f"{parent_path} does not exists. Cannot write output."

    pipeline = MCQAEvaluationPipeline(args)

    # load task
    tasks = args.task.split(",")
    template_names = args.template_name.split(",")
    if len(template_names) == 1:
        template_names = template_names * len(tasks)

    assert len(tasks) == len(template_names), f"Number of tasks and templates should be the same, but got {len(tasks)} != {len(template_names)}"

    evaluation_results = defaultdict(lambda: defaultdict(dict))
    for task, template_name in zip(tasks, template_names):
        samples = pipeline.load_downstream_task(dataset_name=task)

        # evaluation starts
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

            if args.knn_data_file is not None:
                with open(args.knn_data_file) as f:
                    for line in f:
                        indices = [int(index) for index in line.strip().split(",")][1:]
                        demo_sample_list = []
                        for i in range(args.num_fewshot):
                            demo_sample_list.append(samples["train"][indices[i]])

                        demo_samples.append(demo_sample_list)

            else:
                raise ValueError("Please provide the KNN data file.")

            evaluation_result = pipeline.evaluate(
                samples["test"],
                demo_samples=demo_samples,
                template_name=template_name,
                dump_file=args.dump_file
            )

        else:
            evaluation_result = pipeline.evaluate(
                samples["test"],
                demo_samples=samples["train"] if args.num_fewshot > 0 else None,
                template_name=template_name,
                dump_file=args.dump_file
            )
        if evaluation_result is None:
            continue
        evaluation_results[task][template_name] = evaluation_result

    show_pretty_table(evaluation_results)
    if args.result_csv:
        output_as_csv(evaluation_results,args.result_csv)
