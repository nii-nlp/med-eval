import argparse
from collections import defaultdict
import itertools
from pathlib import  Path
from typing import List
import torch
import torch.distributed as dist
from tqdm import tqdm
import unicodedata
import sacrebleu


from tool_utils import main_print, is_main_process, show_pretty_table, output_as_csv
from data_loaders.base import load_ejmmt
from tasks.mt import MTSample, MTRequestDataset
from data_utils import LMDataCollatorForGeneration
from pipeline import EvaluationPipeline


class GenerationForMTPipeline(EvaluationPipeline):
    def __task_specific_preparation__(self):
        self.load_samples_f = load_ejmmt
        self.dataset_f = MTRequestDataset
        self.data_collator_f = LMDataCollatorForGeneration

    def load_downstream_task(self, task_name: str, source_lang: str, target_lang: str):
        if task_name == "ejmmt":
            return load_ejmmt(task_name, source_lang, target_lang)
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def evaluate(
        self,
        samples: List[MTSample],
        demo_samples: List[MTSample] = None,
        template_name: str = None,
        source_lang: str = "english",
        target_lang: str = "japanese"
    ):
        # template_name = "_".join((template_name, source_lang, target_lang))
        dataset, dataloader = self.prepare_data(samples, demo_samples, template_name)

        result_collection = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), disable=not is_main_process()):
                batch = {k: v.to(self.device) if k in ["input_ids"] else v for k, v in batch.items()}

                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    max_new_tokens=self.args.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    # generation_config=self.generation_config,
                    num_beams=1,
                    do_sample=False,
                    # top_p=None,
                    # temperature=None,
                )

                predictions = self.tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[-1]:], skip_special_tokens=True)
                # predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for i, prediction in enumerate(predictions):
                    result_collection.append((
                            batch["request_id"][i],
                            batch["sample"][i],
                            prediction,
                            batch["sample"][i].target_text
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
        bleu_scores = []

        def _post_process(prediction_raw_text, target_lang):
            prediction = prediction_raw_text.split("\n")[0].strip().lstrip()
            # prediction = prediction_raw_text
            if target_lang == "japanese":
                prediction = unicodedata.normalize('NFKC', prediction)
            return prediction

        first_case_flag = True
        for result in all_result_collection:
            prediction = _post_process(result[2], target_lang)

            if first_case_flag:
                main_print(f"=====\n{result[1].source_text}\n-----\nPrediction: {prediction}\n-----\nReference: {result[3]}\n=====")
                first_case_flag = False

            if target_lang == "japanese":
                reference = unicodedata.normalize('NFKC', result[3])
                bleu_score = sacrebleu.corpus_bleu([prediction], [[reference]], tokenize="ja-mecab").score
            else:
                bleu_score = sacrebleu.corpus_bleu([prediction], [[result[3]]]).score
            bleu_scores.append(bleu_score)

        bleu_score = sum(bleu_scores) / len(bleu_scores)

        if is_main_process():
            print(f"BLEU: {bleu_score}")

        return {
            "bleu": bleu_score
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument(
        "--task", type=str, default="medmcqa",
        help="Name of the task or the data_dir of the customized task.")
    parser.add_argument("--template_name", type=str, default="zero-shot")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=0)

    parser.add_argument("--translation", type=str, default="english=>japanese")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--result_csv", type=str, default=None)

    args = parser.parse_args()

    tasks = args.task.split(",")
    translations = args.translation.split(",")
    templates = args.template_name.split(",")
    if args.result_csv is not None:
        parent_path = Path(args.result_csv).parent.exists()
        assert parent_path, f"{parent_path} does not exists. Cannot write output."

    assert (len(tasks) == len(translations)) or (len(tasks) == 1 and len(translations) > 1)

    if len(tasks) == 1:
        tasks = [tasks[0] for _ in range(len(translations))]

    pipeline = GenerationForMTPipeline(args)

    all_bleus, all_tasks = [], []
    evaluation_results = defaultdict(lambda: defaultdict(dict))
    for task, translation in zip(tasks, translations):
        source_lang, target_lang = translation.split("=>")

        samples = pipeline.load_downstream_task(args.task, source_lang, target_lang)

        for template in templates:
            evaluation_result = pipeline.evaluate(
                samples["test"],
                demo_samples=samples["train"] if args.num_fewshot > 0 else None,
                template_name=template,
                source_lang=source_lang,
                target_lang=target_lang
            )

            evaluation_results[task][template] = evaluation_result

            # all_bleus.append(evaluation_results["bleu"])
            # all_tasks.append(f"{task} ({source_lang}=>{target_lang})")

    show_pretty_table(evaluation_results)
    if args.result_csv:
        output_as_csv(evaluation_results,args.result_csv)