import argparse
from collections import defaultdict
from distutils.util import strtobool
from pathlib import Path

import torch
from vllm import SamplingParams

from data_loaders.base import load_ner
from pipeline import EvaluationPipeline
from tasks.ner import NERRequestDataset, NERSample
from tool_utils import output_as_csv, show_pretty_table


class GenerationForNERPipeline(EvaluationPipeline):
    def __task_specific_preparation__(self):
        self.load_samples_f = load_ner
        self.dataset_f = NERRequestDataset
        self.stop_strings = [self.tokenizer.eos_token, "\n"]
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.args.max_new_tokens,
            seed=self.args.seed,
            stop=self.stop_strings,
        )

    def evaluate(
        self,
        samples: list[NERSample],
        demo_samples: list[NERSample] = None,
        template_name: str = "standard",
    ):
        dataset = self.dataset_f(
            samples=samples,
            demo_samples=demo_samples,
            tokenizer=self.tokenizer,
            template_name=template_name,
            num_fewshot=self.args.num_fewshot,
        )

        result_collection = []
        prompt_token_ids = [dataset[j]["input_ids"] for j in range(len(dataset))]
        with torch.inference_mode():
            outputs = self.model.generate(
                sampling_params=self.sampling_params,
                prompt_token_ids=prompt_token_ids,
            )
        outputs_ids = [_out.outputs[0].token_ids for _out in outputs]
        predictions = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)

        for i, prediction in enumerate(predictions):
            result_collection.append(
                (
                    dataset[i]["request_id"],
                    dataset[i]["sample"],
                    prediction,
                    dataset[i]["sample"].labels,
                )
            )

        ## Compute metrics
        entity_f1_scores = []

        def _post_process(prediction_raw_text):
            if prediction_raw_text == "":
                return ["none"]
            else:
                predictions = prediction_raw_text.split(", ")
                predictions = [p.strip().lstrip().lower() for p in predictions]
                return predictions

        first_case_flag = True
        for result in result_collection:
            prediction = result[2].split("\n")[0].strip().lstrip()
            predictions = _post_process(prediction)

            ground_truths = [r.strip().lstrip().lower() for r in result[3]]

            if first_case_flag:
                print(
                    f"=====\n{result[1].text}\n-----\nPrediction: {predictions}\n-----\nReference: {ground_truths}\n====="
                )
                first_case_flag = False

            # Compute F1 score
            prediction_set = set(predictions)
            reference_set = set(ground_truths)
            intersection = prediction_set.intersection(reference_set)
            precision = (
                len(intersection) / len(prediction_set)
                if len(prediction_set) > 0
                else 0
            )
            recall = (
                len(intersection) / len(reference_set) if len(reference_set) > 0 else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
            entity_f1_scores.append(f1)

        entity_f1_score = sum(entity_f1_scores) / len(entity_f1_scores)
        print(f"Entity F1 score: {entity_f1_score:.4f}")

        return {"F1 Entity-level": entity_f1_score}


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
        default="bc5disease_jp",
        help="Name of the task or the data_dir of the customized task.",
    )
    parser.add_argument("--template_name", type=str, default="standard")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument(
        "--use_knn_demo",
        type=strtobool,
        default=False,
        help="Use pre-retrieved KNN-based few-shot learning for the demonstration.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--truncate", type=strtobool, default=False)
    parser.add_argument("--result_csv", type=str, default=None)
    parser.add_argument("--task_category", type=str, default="ner")

    args = parser.parse_args()
    if args.result_csv is not None:
        parent_path = Path(args.result_csv).parent.exists()
        assert parent_path, f"{parent_path} does not exists. Cannot write output."

    pipeline = GenerationForNERPipeline(args)

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
        evaluation_result = pipeline.evaluate(
            samples[args.data_type],
            demo_samples=samples["train"],
            template_name=template_name,
        )

        evaluation_results[task][template_name] = evaluation_result

    show_pretty_table(evaluation_results)
    if args.result_csv:
        output_as_csv(evaluation_results, args.result_csv)
