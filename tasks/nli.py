import random
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
)

import warnings
# warnings.filterwarnings("once")

from templates import NLITemplate
from tasks.base import RequestDataset
from tool_utils import main_print, is_main_process


NLI_LABELS = ["Yes", "No"]


@dataclass
class NLISample:
    sample_id: Union[str, int]
    premise: str
    hypothesis: List[str]
    label: int = None                       # 1: entailment, 0: not entailment
    n_label: int = 2
    metadata: Dict[str, Any] = None

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "label": self.label,
            "n_label": self.n_label,
            "metadata": self.metadata
        }


class NLIRequestDataset(RequestDataset):
    def __init__(
            self,
            samples: List[NLISample],
            demo_samples: List[NLISample] = None,
            tokenizer: PreTrainedTokenizer = None,
            template_name: str = "rqe",
            num_fewshot: int = 0,
            truncate: bool = False,
            label_set: List[str] = None
    ):
        self.samples = samples
        self.demo_samples = demo_samples if demo_samples is not None else []
        assert isinstance(samples[0], NLISample)

        self.template_name = template_name

        self.num_fewshot = num_fewshot
        assert self.num_fewshot == 0 or (self.num_fewshot > 0 and self.num_fewshot <= len(self.demo_samples))

        self.tokenizer = tokenizer
        self.truncate = truncate

        self.__task_sepcific_preparation__()
        self.template = self.template_f(template_name)

        self.label_set = label_set

        if self.tokenizer.name_or_path in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
            self.requests = self.construct_requests_chat()
        else:
            self.requests = self.construct_requests()

    def __task_sepcific_preparation__(self):
        assert isinstance(self.samples[0], NLISample)
        self.template_f = NLITemplate

    def construct_requests(self):
        main_print("Constructing requests...")
        first_sample_flag = True

        requests = []
        for i, sample in enumerate(tqdm(self.samples, total=len(self.samples), desc="Constructing requests")):

            # Few-shot evaluation: In-context learning
            if self.num_fewshot > 0:
                # random
                if isinstance(self.demo_samples[0], NLISample):
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
                    text = self.instantiate_template(demo) + f" {self.label_set[demo.label]}"
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
            for j, option in enumerate(self.label_set):
                output_text = input_text + " {}".format(option)
                if first_sample_flag and self.label_set[sample.label] == option:
                    main_print(f'=====\n{input_text}\n-----\n{" {}".format(option)}\n=====')
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

