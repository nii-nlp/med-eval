from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import List
import random

from tasks.base import RequestDataset
from templates import MTTemplate
from tool_utils import main_print


@dataclass
class MTSample:
    source_text: str
    target_text: str
    source_language: str = None
    target_language: str = None


class MTRequestDataset(RequestDataset):
    def __init__(
            self,
            samples: List[MTSample],
            demo_samples: List[MTSample] = None,
            tokenizer: PreTrainedTokenizer = None,
            template_name: str = "",
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
        assert isinstance(self.samples[0], MTSample)
        self.template_f = MTTemplate

    def construct_requests(self):
        main_print("Constructing requests...")
        first_sample_flag = True

        requests = []
        for i, sample in enumerate(self.samples):

            # Few-shot evaluation: In-context learning
            if self.num_fewshot > 0:

                # random
                if isinstance(self.demo_samples[0], MTSample):
                    candidate_few_shot_demos = random.choices(self.demo_samples, k=min(self.num_fewshot * 2, len(self.demo_samples)))

                else:
                    raise NotImplementedError

                assert len(candidate_few_shot_demos) >= self.num_fewshot

                deduplicating_text_set = set()
                valid_few_shot_demos = []
                for demo in candidate_few_shot_demos:
                    text = self.instantiate_template(demo) + f" {demo.target_text}"
                    if text not in deduplicating_text_set:
                        deduplicating_text_set.add(text)
                        valid_few_shot_demos.append(text)

                valid_few_shot_demos = valid_few_shot_demos[:self.num_fewshot][::-1]

                few_shot_input_text = "\n".join(valid_few_shot_demos)
                input_text = few_shot_input_text + "\n"

            else:
                input_text = ""

            input_text += self.instantiate_template(sample)

            if first_sample_flag:
                main_print(f'=====\n{input_text}\n-----\n{" {}".format(sample.target_text)}\n=====')
                first_sample_flag = False

            encodings = self.tokenizer(input_text)

            requests.append(
                dict(
                    request_id=i,
                    sample=sample,
                    input_ids=encodings["input_ids"],
                )
            )

        return requests
