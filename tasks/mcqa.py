import random
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
)

import warnings
# warnings.filterwarnings("once")

from templates import MCQATemplate
from tasks.base import RequestDataset
from tool_utils import main_print, is_main_process

try:
    from rank_bm25 import BM25Okapi
except:
    main_print("BM25 is not installed. Please install rank_bm25 to use BM25.")


@dataclass
class MCQASample:
    sample_id: Union[str, int]
    question: str
    options: List[str]
    answer_idx: int
    n_options: int
    metadata: Dict[str, Any]

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "options": self.options,
            "answer_idx": self.answer_idx,
            "n_options": self.n_options,
            "metadata": self.metadata
        }

    def __post_init__(self):
        self.n_options = len(self.options)
        for option in self.options:
            assert option != "", f"Empty option in {self}"


class MCQARequestDataset(RequestDataset):
    def __init__(
            self,
            samples: List[MCQASample],
            demo_samples: List[MCQASample] = None,
            tokenizer: PreTrainedTokenizer = None,
            template_name: str = "mcqa",
            num_fewshot: int = 0,
            truncate: bool = False,
    ):
        super().__init__(
            samples=samples,
            demo_samples=demo_samples,
            tokenizer=tokenizer,
            template_name=template_name,
            num_fewshot=num_fewshot,
            truncate=truncate,
        )

    def __task_sepcific_preparation__(self):
        assert isinstance(self.samples[0], MCQASample)
        self.template_f = MCQATemplate

    def construct_requests(self):
        main_print("Constructing requests...")
        first_sample_flag = True

        requests = []
        for i, sample in enumerate(tqdm(self.samples, total=len(self.samples), desc="Constructing requests", disable=not is_main_process())):

            # Few-shot evaluation: In-context learning
            if self.num_fewshot > 0:
                # random
                if isinstance(self.demo_samples[0], MCQASample):
                    candidate_few_shot_demos = random.choices(self.demo_samples, k=min(self.num_fewshot * 2, len(self.demo_samples)))

                # knn
                elif isinstance(self.demo_samples[0], List):
                    candidate_few_shot_demos = self.demo_samples[i]

                else:
                    raise NotImplementedError

                assert len(candidate_few_shot_demos) >= self.num_fewshot

                deduplicating_text_set = set()
                valid_few_shot_demos = []
                for demo in candidate_few_shot_demos:
                    if isinstance(demo, str):
                        text = demo
                    else:
                        text = self.instantiate_template(demo) + f"{demo.options[demo.answer_idx]}"

                    if text not in deduplicating_text_set:
                        deduplicating_text_set.add(text)
                        valid_few_shot_demos.append(text)

                valid_few_shot_demos = valid_few_shot_demos[:self.num_fewshot][::-1]

                few_shot_input_text = "\n\n".join(valid_few_shot_demos)
                input_text = few_shot_input_text + "\n\n"

            else:
                input_text = ""

            input_text += self.instantiate_template(sample)
            input_token_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"].squeeze().tolist()

            all_output_token_ids = []
            for j, option in enumerate(sample.options):
                output_text = input_text + "{}".format(option)

                if first_sample_flag and sample.answer_idx == j:
                    main_print(f'=====\n{input_text}\n-----\n{"{}".format(option)}\n=====')
                    first_sample_flag = False

                all_token_ids = self.tokenizer(output_text, return_tensors="pt")["input_ids"].squeeze().tolist()

                output_token_ids = all_token_ids[len(input_token_ids):]

                all_output_token_ids.append(output_token_ids)

            try:
                max_n_length_of_output = max([len(output_token_ids) for output_token_ids in all_output_token_ids])
            except ValueError:
                if is_main_process():
                    print(all_output_token_ids)
                    print(sample)
                    warnings.warn(f"max() arg is an empty sequence")
                    exit(1)

            max_n_length_of_input = self.tokenizer.model_max_length - max_n_length_of_output

            for j, output_token_ids in enumerate(all_output_token_ids):
                if len(input_token_ids) > max_n_length_of_input and is_main_process():
                    if self.truncate:
                        warnings.warn(f"Input text is too long: {len(input_token_ids)}. It is truncated now.")
                    else:
                        warnings.warn(f"Just a reminder, input text is too long: {len(input_token_ids)}.")

                if self.truncate:
                    input_token_ids = input_token_ids[-max_n_length_of_input:]

                if self.tokenizer.bos_token_id is not None:
                    all_token_ids = [self.tokenizer.bos_token_id] + input_token_ids[1:] + output_token_ids
                    labels = [-100] * len(input_token_ids) + output_token_ids
                else:
                    all_token_ids = input_token_ids + output_token_ids
                    labels = [-100] * len(input_token_ids) + output_token_ids

                requests.append(
                    dict(
                        request_id=i,
                        option_id=j,
                        sample=sample,
                        input_ids=all_token_ids,
                        labels=labels,
                    )
                )

        return requests

    def construct_requests_chat(self):
        main_print("Constructing requests in chatting format...")
        first_sample_flag = True

        requests = []
        for i, sample in enumerate(self.samples):
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Please answer the following medical question, by selecting the most appropriate answer from the options below."},
            ]

            # Few-shot evaluation: In-context learning
            if self.num_fewshot > 0:
                few_shot_demos = random.choices(self.demo_samples, k=self.num_fewshot)

                # multi-turn dialogue
                for demo in few_shot_demos:
                    messages.extend([
                        {"role": "user", "content": self.instantiate_template(demo)},
                        {"role": "assistant", "content": f" {demo.options[demo.answer_idx]}"}
                    ])
                # single-turn dialogue
                # few_shot_input_text = "\n\n".join([
                #     self.instantiate_template(demo) + f" {demo.options[demo.answer_idx]}"
                #     for demo in few_shot_demos
                # ])
                # input_text = few_shot_input_text + "\n\n"

            else:
                # input_text = ""
                pass

            # multi-turn dialogue
            messages.append({"role": "user", "content": self.instantiate_template(sample)})

            # single-turn dialogue
            # input_text += self.instantiate_template(sample)
            # messages.append({"role": "user", "content": input_text})

            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            input_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_text))

            all_output_token_ids = []
            for j, option in enumerate(sample.options):
                output_text = input_text + " {}".format(option)
                if first_sample_flag:
                    main_print(f'=====\n{input_text}\n-----\n{" {}".format(option)}\n=====')
                    first_sample_flag = False

                all_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(output_text))

                output_token_ids = all_token_ids[len(input_token_ids):]

                all_output_token_ids.append(output_token_ids)

            max_n_length_of_output = max([len(output_token_ids) for output_token_ids in all_output_token_ids])
            max_n_length_of_input = self.tokenizer.model_max_length - max_n_length_of_output

            for j, output_token_ids in enumerate(all_output_token_ids):
                if len(input_token_ids) > max_n_length_of_input and is_main_process():
                    warnings.warn(f"Input text is too long: {len(input_token_ids)}. It is truncated now.")
                input_token_ids = input_token_ids[-max_n_length_of_input:]

                all_token_ids = input_token_ids + output_token_ids
                labels = [-100] * len(input_token_ids) + output_token_ids

                requests.append(
                    dict(
                        request_id=i,
                        option_id=j,
                        sample=sample,
                        input_ids=all_token_ids,
                        labels=labels,
                    )
                )
        return requests

