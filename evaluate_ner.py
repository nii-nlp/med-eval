import argparse
import itertools
from typing import List
import torch
from prettytable import PrettyTable
from tqdm import tqdm
from distutils.util import strtobool
from transformers import GenerationConfig
import torch.distributed as dist

from tool_utils import main_print, is_main_process
from tasks.ner import NERSample, NERRequestDataset
from data_loaders.base import load_blurb, load_ner
from data_utils import LMDataCollatorForGeneration
from pipeline import EvaluationPipeline


SUPPORTING_TASKS = ["bc2gm", "bc5chem", "bc5disease", "jnlpba", "ncbi_disease"]


class GenerationForNERPipeline(EvaluationPipeline):
    def __prepare_tokenizer_and_model__(self, model_name_or_path):
        super().__prepare_tokenizer_and_model__(model_name_or_path)
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        self.generation_config.stop_strings = [self.tokenizer.eos_token, "\n"]

    def __task_specific_preparation__(self):
        self.load_samples_f = load_ner
        self.dataset_f = NERRequestDataset
        self.data_collator_f = LMDataCollatorForGeneration

    def evaluate(
            self,
            samples: List[NERSample],
            demo_samples: List[NERSample] = None,
            template_name: str = "standard"
    ):
        dataset, dataloader = self.prepare_data(samples, demo_samples, template_name)

        result_collection = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), disable=not is_main_process()):
                batch = {k: v.to(self.device) if k in ["input_ids"] else v for k, v in batch.items()}

                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    max_new_tokens=self.args.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    generation_config=self.generation_config,
                    num_beams=1,
                    do_sample=False,
                )

                predictions = self.tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[-1]:], skip_special_tokens=True)
                for i, prediction in enumerate(predictions):
                    result_collection.append((
                        batch["request_id"][i],
                        batch["sample"][i],
                        prediction,
                        batch["sample"][i].labels
                    ))

        if self.using_ddp:
            all_result_collection = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(
                all_result_collection,
                result_collection
            )
            all_result_collection = list(itertools.chain(*all_result_collection))

            existed_request_ids = set()
            deduplicated_result_collection = []
            for result in all_result_collection:
                if f"{result[0]}" not in existed_request_ids:
                    deduplicated_result_collection.append(result)
                    existed_request_ids.add(f"{result[0]}")
                else:
                    pass
            all_result_collection = deduplicated_result_collection

        else:
            all_result_collection = result_collection

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
        for result in all_result_collection:
            prediction = result[2].split("\n")[0].strip().lstrip()
            predictions = _post_process(prediction)

            ground_truths = [r.strip().lstrip().lower() for r in result[3]]

            if first_case_flag:
                main_print(f"=====\n{result[1].text}\n-----\nPrediction: {predictions}\n-----\nReference: {ground_truths}\n=====")
                first_case_flag = False

            # Compute F1 score
            prediction_set = set(predictions)
            reference_set = set(ground_truths)
            intersection = prediction_set.intersection(reference_set)
            precision = len(intersection) / len(prediction_set) if len(prediction_set) > 0 else 0
            recall = len(intersection) / len(reference_set) if len(reference_set) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            entity_f1_scores.append(f1)

        entity_f1_score = sum(entity_f1_scores) / len(entity_f1_scores)
        main_print(f"Entity F1 score: {entity_f1_score:.4f}")

        return {
            "F1 Entity-level": entity_f1_score
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--task", type=str, default="bc5disease",
        help="Name of the task or the data_dir of the customized task.")
    parser.add_argument("--template_name", type=str, default="standard")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--use_knn_demo", type=strtobool, default=False,
                        help="Use pre-retrieved KNN-based few-shot learning for the demonstration.")

    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--truncate", type=strtobool, default=False)

    args = parser.parse_args()

    pipeline = GenerationForNERPipeline(args)

    # load task
    tasks = args.task.split(",")
    evaluation_results = {}
    for task in tasks:
        samples = pipeline.load_downstream_task(dataset_name=task)

        evaluation_result = pipeline.evaluate(
            samples["test"],
            demo_samples=samples["train"],
            template_name=args.template_name
        )

        evaluation_results[task] = evaluation_result

    tb = PrettyTable()
    tb.field_names = ["Task", "F1 Entity-level"]
    for task, result in evaluation_results.items():
        tb.add_row([task, result["F1 Entity-level"]])

    main_print(tb)
