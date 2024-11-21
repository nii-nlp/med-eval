import argparse
import copy
import json
from collections import defaultdict
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import SamplingParams

from data_loaders.base import load_nli_samples
from data_utils import LMDataCollatorForPerplexity
from pipeline import EvaluationPipeline
from tasks.nli import NLIRequestDataset, NLISample
from tool_utils import output_as_csv, show_pretty_table


class STSEvaluationPipeline(EvaluationPipeline):
    def __task_specific_preparation__(self):
        self.load_samples_f = load_nli_samples
        self.dataset_f = NLIRequestDataset
        self.data_collator_f = LMDataCollatorForPerplexity

    def init_verbalizer(self, nli_labels: list[str]):
        self.label_set = nli_labels

    def _loglikelihood_batch(self, input_ids, labels, batch):
        n_batch = batch["input_ids"].size(0)

        lm_logits = self.model(input_ids=input_ids).logits

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        losses = losses.view(n_batch, -1).sum(dim=-1)

        return losses

    def prepare_data(
        self,
        samples: list,
        demo_samples: list = None,
        template_name: str = None,
    ):
        dataset = self.dataset_f(
            samples=samples,
            demo_samples=demo_samples,
            tokenizer=self.tokenizer,
            template_name=template_name,
            num_fewshot=self.args.num_fewshot,
            truncate=self.args.truncate,
            label_set=self.label_set,
        )
        data_collator = self.data_collator_f(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            collate_fn=data_collator,
            shuffle=False,
            drop_last=False,
        )

        return dataset, dataloader

    def evaluate(
        self,
        samples: list[NLISample],
        demo_samples: list[NLISample] | list[list[NLISample]] = None,
        template_name: str = None,
        dump_file: str = None,
    ):
        dataset, dataloader = self.prepare_data(samples, demo_samples, template_name)

        result_collection = []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch = {
                    k: v.to(self.model.device) if k in ["input_ids", "labels"] else v
                    for k, v in batch.items()
                }

                losses = self._loglikelihood_batch(
                    batch["input_ids"], batch["labels"], batch
                )

                for i in range(len(losses)):
                    result_collection.append(
                        (
                            batch["request_id"][i],
                            batch["option_id"][i],
                            batch["sample"][i],
                            losses[i].item(),
                            (batch["labels"][i] != -100).sum().item(),
                        )
                    )
                    if (batch["labels"][i] != -100).sum().item() == 0:
                        print(batch["input_ids"][i])
                        print("-----")
                        print(batch["labels"][i])
                        print("-----")
                        print(batch["sample"][i])
                        print("-----")
                        print(losses[i].item())
                        exit(1)
        losses = {
            k: [1e7 for _ in range(dataset.samples[0].n_label)]
            for k, v in enumerate(dataset.samples)
        }
        n_valid_tokens = {
            k: [1e7 for _ in range(dataset.samples[0].n_label)]
            for k, v in enumerate(dataset.samples)
        }
        request_id2sample = {result[0]: result[2] for result in result_collection}

        for request_id, option_id, sample, loss, n_valid_token in result_collection:
            losses[request_id][option_id] = loss
            n_valid_tokens[request_id][option_id] = n_valid_token

        predictions = []
        request_id2prediction = {}
        for k, v in losses.items():
            predictions.append(np.argmin(v))
            request_id2prediction[k] = np.argmin(v)

            if dump_file:
                # record the results
                with open(dump_file, "a+", encoding="utf-8") as writer:
                    writer.write(
                        json.dumps(
                            {
                                "request_id": k,
                                "losses": v,
                                "prediction": np.argmin(v),
                                "sample": request_id2sample[k].to_dict(),
                            }
                        )
                        + "\n"
                    )

        ground_truths = [sample.label for sample in dataset.samples]

        ## Pearson Correlation
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        pearson = scipy.stats.pearsonr(predictions, ground_truths).statistic

        print(f"Pearson Correlation: {pearson:.4f}")

        return {
            "pearson": pearson,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument(
        "--data_type", type=str, default="validation", choices=["validation", "test"]
    )
    parser.add_argument(
        "--task",
        type=str,
        default="jcsts",
        help="Name of the task or the data_dir of the customized task.",
    )
    parser.add_argument("--template_name", type=str, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument(
        "--use_fake_demo",
        type=strtobool,
        default=False,
        help="""According to Min et al., EMNLP 2022, we understand that we don't need to use the exact demonstrations from the training set.
Therefore, we use the question from the test set itself, and randomly select an option as the fake answer.
Experiment should that it doesn't affect the performance and even perform similar when we need to find true demos from other similar dataset like MedQA w.r.t. IgakuQA.
In default, we don't use this option, but use the exact demonstrations from the training set""",
    )

    parser.add_argument(
        "--knn_data_file",
        type=str,
        default=None,
        help="Use pre-retrieved KNN-based few-shot learning for the demonstration.",
    )

    parser.add_argument("--truncate", type=strtobool, default=False)
    parser.add_argument(
        "--nli_labels", type=str, default="No,Yes"
    )  # "No,Yes|No,Yes,Mixture,Unproven"
    parser.add_argument("--dump_file", type=str, default=None)
    parser.add_argument("--result_csv", type=str, default=None)
    parser.add_argument("--task_category", type=str, default="sts")

    args = parser.parse_args()

    if args.result_csv is not None:
        parent_path = Path(args.result_csv).parent.exists()
        assert parent_path, f"{parent_path} does not exists. Cannot write output."

    pipeline = STSEvaluationPipeline(args)

    # load task
    tasks = args.task.split(",")
    template_names = args.template_name.split(",")
    if len(template_names) == 1:
        template_names = template_names * len(tasks)

    nli_labels = args.nli_labels.split("|")
    if len(nli_labels) == 1:
        nli_labels = nli_labels * len(tasks)

    evaluation_results = defaultdict(lambda: defaultdict(dict))
    for task, label_set in zip(tasks, nli_labels):
        for template_name in template_names:

            samples = pipeline.load_downstream_task(dataset_name=task)
            pipeline.init_verbalizer(label_set.split(","))

            assert samples["test"][0].n_label == len(
                label_set.split(",")
            ), f"Number of labels in the task ({samples['test'][0].n_label}) is not equal to the number of labels in the arguments ({len(label_set.split(','))})"

            # evaluation starts
            if args.num_fewshot == 0:
                demo_samples = None
            elif args.use_fake_demo:
                ## Reference: Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (Min et al., 2022)
                demo_samples = copy.deepcopy(samples[args.data_type])

            elif args.knn_data_file:
                demo_samples = []
                with open(args.knn_data_file) as f:
                    json_data = json.load(f)
                for i in range(len(json_data["test"])):
                    indices = copy.deepcopy(json_data["test"][i]["few_shot_indices"])
                    source = json_data["test"][i]["metadata"]["source"]
                    demo_sample_list = []
                    for index in indices:
                        demo_sample_list.append(samples[source][index])

                    demo_samples.append(demo_sample_list)
            else:
                demo_samples = samples["train"]
            evaluation_result = pipeline.evaluate(
                samples[args.data_type],
                demo_samples=demo_samples,
                template_name=template_name,
                dump_file=args.dump_file,
            )

            evaluation_results[task][template_name] = evaluation_result

    show_pretty_table(evaluation_results)
    if args.result_csv:
        output_as_csv(evaluation_results, args.result_csv)
