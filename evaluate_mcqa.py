import argparse
import copy
import json
import logging
import random
import warnings
from collections import defaultdict
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import torch
from vllm import SamplingParams

from data_loaders.base import load_mcqa_samples
from pipeline import EvaluationPipeline
from tasks.mcqa import MCQARequestDataset, MCQASample
from tool_utils import output_as_csv, show_pretty_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class MCQAEvaluationPipeline(EvaluationPipeline):
    def __task_specific_preparation__(self):
        self.load_samples_f = load_mcqa_samples
        self.dataset_f = MCQARequestDataset
        self.sampling_params = SamplingParams(
            temperature=0.0,
            prompt_logprobs=0,
            max_tokens=1,
            seed=self.args.seed,
        )

    def _loglikelihood_batch(self, input_ids: list[list[int]]):
        outputs = self.model.generate(
            sampling_params=self.sampling_params, prompt_token_ids=input_ids
        )
        logprobs = [
            [
                list(input_token.values())[0].logprob
                for input_token in _out.prompt_logprobs
                if input_token is not None
            ]
            for _out in outputs
        ]
        return logprobs

    def evaluate(
        self,
        samples: list[MCQASample],
        demo_samples: list[MCQASample] | list[list[MCQASample]] = None,
        template_name: str = None,
        dump_file: str = None,
    ):
        try:
            dataset = self.dataset_f(
                samples=samples,
                demo_samples=demo_samples,
                tokenizer=self.tokenizer,
                template_name=template_name,
                num_fewshot=self.args.num_fewshot,
                truncate=self.args.truncate,
            )

        except AssertionError as e:
            logger.warning(e)
            logger.warning("Skip this task due to the lack of samples for few-shot learning.")
            return
        except Exception as e:
            raise e

        result_collection = []

        prompt_token_ids = [dataset[j]["input_ids"] for j in range(len(dataset))]
        with torch.inference_mode():
            logprobs = self._loglikelihood_batch(prompt_token_ids)

        for i in range(len(logprobs)):
            is_target_outputs = np.array(dataset[i]["labels"][1:]) != -100
            target_loss = -np.where(is_target_outputs, logprobs[i], 0).sum()
            result_collection.append(
                (
                    dataset[i]["request_id"],
                    dataset[i]["option_id"],
                    dataset[i]["sample"],
                    target_loss,
                    is_target_outputs.sum(),
                )
            )

        # IgakuQA has different number of options for each sample
        # assert (len(result_collection) == dataset.num_samples * dataset.num_options), f"{len(result_collection)} != {dataset.num_samples * dataset.num_options}"

        losses = {
            k: [1e7 for _ in range(len(v.options))]
            for k, v in enumerate(dataset.samples)
        }
        n_valid_tokens = {
            k: [1e7 for _ in range(len(v.options))]
            for k, v in enumerate(dataset.samples)
        }

        for request_id, option_id, sample, loss, n_valid_token in result_collection:
            losses[request_id][option_id] = loss
            n_valid_tokens[request_id][option_id] = n_valid_token

        predictions = []
        norm_predictions = []
        request_id2prediction = {}
        for k, v in losses.items():
            predictions.append(np.argmin(v))
            request_id2prediction[k] = np.argmin(v)
            try:
                norm_predictions.append(
                    np.argmin([loss / n_valid_tokens[k][i] for i, loss in enumerate(v)])
                )
            except ZeroDivisionError:
                norm_predictions.append(np.argmin(v))
                warnings.warn("Error: Some options are missing...")

            if dump_file:
                request_id2sample = {
                    result[0]: result[2] for result in result_collection
                }
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

        ground_truths = [sample.answer_idx for sample in dataset.samples]

        accuracy = np.mean(np.array(predictions) == np.array(ground_truths))
        norm_accuracy = np.mean(np.array(norm_predictions) == np.array(ground_truths))

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Norm Accuracy: {norm_accuracy:.4f}")

        return {"accuracy": accuracy, "norm_accuracy": norm_accuracy}


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
        default="medmcqa",
        help="Name of the task or the data_dir of the customized task.",
    )
    parser.add_argument("--template_name", type=str, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    use_fake_demo_help = "According to Min et al., EMNLP 2022, we understand that we don't need to use the exact demonstrations from the training set. Therefore, we use the question from the test set itself, and randomly select an option as the fake answer. Experiment should that it doesn't affect the performance and even perform similar when we need to find true demos from other similar dataset like MedQA w.r.t. IgakuQA. In default, we don't use this option, but use the exact demonstrations from the training set"
    parser.add_argument(
        "--use_fake_demo",
        type=strtobool,
        default=False,
        help=use_fake_demo_help,
    )

    parser.add_argument(
        "--knn_data_file",
        type=str,
        default=None,
        help="Use pre-retrieved KNN-based few-shot learning for the demonstration.",
    )

    parser.add_argument("--truncate", type=strtobool, default=False)
    parser.add_argument("--dump_file", type=str, default=None)
    parser.add_argument("--result_csv", type=str, default=None)

    args = parser.parse_args()

    if args.result_csv is not None:
        parent_path = Path(args.result_csv).parent.exists()
        assert parent_path, f"{parent_path} does not exists. Cannot write output."

    pipeline = MCQAEvaluationPipeline(args)

    # load task
    tasks = args.task.split(",")
    template_names = args.template_name.split(",")
    if len(template_names) == 1:
        template_names = template_names * len(tasks)

    assert len(tasks) == len(
        template_names
    ), f"Number of tasks and templates should be the same, but got {len(tasks)} != {len(template_names)}"

    evaluation_results = defaultdict(lambda: defaultdict(dict))
    for task, template_name in zip(tasks, template_names):
        samples = pipeline.load_downstream_task(dataset_name=task)

        # evaluation starts
        if args.num_fewshot == 0:
            demo_samples = None
        elif args.use_fake_demo:
            ## Reference: Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (Min et al., 2022)
            shuffle_test_samples = copy.deepcopy(samples[args.data_type])
            for j in range(len(shuffle_test_samples)):
                random.shuffle(shuffle_test_samples[j].options)
            demo_samples = shuffle_test_samples
        elif args.knn_data_file:
            demo_samples = []
            with open(args.knn_data_file) as f:
                for line in f:
                    indices = [int(index) for index in line.strip().split(",")][1:]
                    demo_sample_list = [
                        samples["train"][indices[i]] for i in range(args.num_fewshot)
                    ]
                    demo_samples.append(demo_sample_list)
        else:
            demo_samples = samples["train"]

        if len(samples[args.data_type]) == 0:
            logger.warning("Skip this task due to the lack of samples.")
            continue

        evaluation_result = pipeline.evaluate(
            samples[args.data_type],
            demo_samples=demo_samples,
            template_name=template_name,
            dump_file=args.dump_file,
        )
        if evaluation_result is None:
            continue
        evaluation_results[task][template_name] = evaluation_result

    show_pretty_table(evaluation_results)
    if args.result_csv:
        output_as_csv(evaluation_results, args.result_csv)
