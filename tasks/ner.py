from dataclasses import dataclass
from typing import List
from transformers import PreTrainedTokenizer
import random

from templates import NERTemplate
from tasks.base import RequestDataset
from tool_utils import main_print


@dataclass
class NERSample:
    text: str
    labels: List[str]
    entity_type: str = None

    def to_dict(self):
        return {
            "text": self.text,
            "labels": self.labels,
            "entity_type": self.entity_type
        }


class NERRequestDataset(RequestDataset):
    def __init__(
            self,
            samples: List[NERSample],
            demo_samples: List[NERSample] = None,
            tokenizer: PreTrainedTokenizer = None,
            template_name: str = "standard",
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
        assert isinstance(self.samples[0], NERSample)
        self.template_f = NERTemplate

    def construct_requests(self):
        main_print("Constructing requests...")
        first_sample_flag = True

        requests = []
        for i, sample in enumerate(self.samples):
            ## Instruction for the NER task
            input_text = f"Please extract all {sample.entity_type.lower()}s mentioned in the paragraph.\n"

            # Few-shot evaluation: In-context learning
            if self.num_fewshot > 0:
                few_shot_demos = random.choices(self.demo_samples, k=self.num_fewshot + 10)
                valid_few_shot_demos = []
                for demo in few_shot_demos:
                    try:
                        valid_few_shot_demos.append(self.instantiate_template(demo) + f" {', '.join(demo.labels)}")
                    except:
                        continue

                    if len(valid_few_shot_demos) == self.num_fewshot:
                        break

                few_shot_input_text = "\n".join(valid_few_shot_demos)
                input_text += few_shot_input_text + "\n"
            else:
                pass

            input_text += self.instantiate_template(sample)

            if first_sample_flag:
                main_print(f'=====\n{input_text}\n-----\n{" {}".format(", ".join(sample.labels))}\n=====')
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

