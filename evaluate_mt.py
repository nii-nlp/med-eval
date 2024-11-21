import argparse
import unicodedata
from collections import defaultdict
from pathlib import Path

import sacrebleu
import torch
from vllm import SamplingParams

from data_loaders.base import load_ejmmt
from pipeline import EvaluationPipeline
from tasks.mt import MTRequestDataset, MTSample
from tool_utils import output_as_csv, show_pretty_table


class GenerationForMTPipeline(EvaluationPipeline):
    def __task_specific_preparation__(self):
        self.load_samples_f = load_ejmmt
        self.dataset_f = MTRequestDataset
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.args.max_new_tokens,
            seed=self.args.seed,
        )

    def load_downstream_task(self, task_name: str, source_lang: str, target_lang: str):
        if task_name == "ejmmt":
            dataset = load_ejmmt(task_name, source_lang, target_lang)
        else:
            raise ValueError(f"Unknown task: {task_name}")
        return dataset

    def evaluate(
        self,
        samples: list[MTSample],
        demo_samples: list[MTSample] = None,
        template_name: str = None,
        source_lang: str = "english",
        target_lang: str = "japanese",
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
                    dataset[i]["sample"].target_text,
                )
            )

        ## Compute metrics
        bleu_scores = []

        def _post_process(prediction_raw_text, target_lang):
            prediction = prediction_raw_text.split("\n")[0].strip().lstrip()
            if target_lang == "japanese":
                prediction = unicodedata.normalize("NFKC", prediction)
            return prediction

        first_case_flag = True
        for result in result_collection:
            prediction = _post_process(result[2], target_lang)

            if first_case_flag:
                print(
                    f"=====\n{result[1].source_text}\n-----\nPrediction: {prediction}\n-----\nReference: {result[3]}\n====="
                )
                first_case_flag = False

            if target_lang == "japanese":
                reference = unicodedata.normalize("NFKC", result[3])
                bleu_score = sacrebleu.corpus_bleu(
                    [prediction], [[reference]], tokenize="ja-mecab"
                ).score
            else:
                bleu_score = sacrebleu.corpus_bleu([prediction], [[result[3]]]).score
            bleu_scores.append(bleu_score)

        bleu_score = sum(bleu_scores) / len(bleu_scores)

        print(f"BLEU: {bleu_score}")

        return {"bleu": bleu_score}


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
        default="ejmmt",
        help="Name of the task or the data_dir of the customized task.",
    )
    parser.add_argument("--template_name", type=str, default="mt_minimal")
    parser.add_argument("--num_fewshot", type=int, default=0)

    parser.add_argument("--translation", type=str, default="english-japanese")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--result_csv", type=str, default=None)
    parser.add_argument("--task_category", type=str, default="mt")

    args = parser.parse_args()

    tasks = args.task.split(",")
    translations = args.translation.split(",")
    templates = args.template_name.split(",")
    if args.result_csv is not None:
        parent_path = Path(args.result_csv).parent.exists()
        assert parent_path, f"{parent_path} does not exists. Cannot write output."

    assert (len(tasks) == len(translations)) and (
        len(tasks) == 1 or len(tasks) == len(templates)
    )

    if len(tasks) != len(templates):
        tasks = [tasks[0] for _ in range(len(templates))]
        translations = [translations[0] for _ in range(len(templates))]

    pipeline = GenerationForMTPipeline(args)

    all_bleus, all_tasks = [], []
    evaluation_results = defaultdict(lambda: defaultdict(dict))
    for _task, _translation, _template in zip(tasks, translations, templates):
        source_lang, target_lang = _translation.split("-")
        samples = pipeline.load_downstream_task(_task, source_lang, target_lang)
        evaluation_result = pipeline.evaluate(
            samples[args.data_type],
            demo_samples=samples["train"] if args.num_fewshot > 0 else None,
            template_name=_template,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        translation_short = source_lang[:2] + "2" + target_lang[:2]
        full_task_name = _task + "-" + translation_short
        evaluation_results[full_task_name][_template] = evaluation_result

    show_pretty_table(evaluation_results)
    if args.result_csv:
        output_as_csv(evaluation_results, args.result_csv)
