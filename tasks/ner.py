import random
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from tasks.base import RequestDataset
from templates import NERTemplate


@dataclass
class NERSample:
    text: str
    labels: list[str]
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
            samples: list[NERSample],
            demo_samples: list[NERSample] = None,
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
        print("Constructing requests...")
        first_sample_flag = True

        requests = []
        for i, sample in enumerate(self.samples):
            ## Instruction for the NER task
            instantiated_sample =  self.instantiate_template(sample)
            if self.template_name == "standard":
                if "以下の" in instantiated_sample and "において" in instantiated_sample:
                    input_text = ""
                else:
                    input_text = f"以下の段落において、{sample.entity_type}は？\n"

            elif self.template_name == "minimal":
                input_text = ""
                if "以下の" in instantiated_sample and "において" in instantiated_sample:
                    s_list = instantiated_sample.split("\n")
                    instantiated_sample = "\n".join(s_list[1:])

            elif self.template_name == "english-centric":
                entity_jp2en = {
                    "患者に実際に認められた病変や症状の存在を示す異常などの所見を表す表現": "Disease",
                    "薬品にかかわる値": "Medicine Value",
                    "薬品名": "Medicine Key",
                    "遺伝子": "gene",
                    "化学物質": "chemical",
                    "疾患": "disease",
                    "タンパク質、DNA、RNA、細胞株、または細胞タイプ": "protein, DNA, RNA, cell line, or cell type",
                }
                input_text = f"Please extract all {entity_jp2en[sample.entity_type].lower()}s mentioned in the paragraph.\n"
                if "以下の" in instantiated_sample and "において" in instantiated_sample:
                    s_list = instantiated_sample.split("\n")
                    instantiated_sample = "\n".join(s_list[1:])
            elif self.template_name == "instructed":
                input_text = f"""あなたは医療分野の専門家です。\n
                あなたは{sample.entity_type}のフレーズを含む段落を与えられます。\n
                あなたのタスクは段落からこれらすべてのフレーズを抽出することです。\n
                抽出されたフレーズのみを返し、それらを英語のカンマ（,）で区切る必要があります。\n"""
                if "以下の" in instantiated_sample and "において" in instantiated_sample:
                    s_list = instantiated_sample.split("\n")
                    instantiated_sample = "\n".join(s_list[1:])
            # input_text = f"Please extract all {sample.entity_type.lower()}s mentioned in the paragraph.\n"

            # Few-shot evaluation: In-context learning
            if self.num_fewshot > 0:
                few_shot_demos = random.choices(self.demo_samples, k=self.num_fewshot + 10)
                valid_few_shot_demos = []
                for demo in few_shot_demos:
                    try:
                        assert demo.labels
                        valid_few_shot_demos.append(self.instantiate_template(demo) + f" {', '.join(demo.labels)}")
                    except:
                        continue

                    if len(valid_few_shot_demos) == self.num_fewshot:
                        break

                few_shot_input_text = "\n".join(valid_few_shot_demos)
                input_text += few_shot_input_text + "\n"
            else:
                pass

            # input_text += self.instantiate_template(sample)
            input_text += instantiated_sample

            if first_sample_flag:
                print(f'=====\n{input_text}\n-----\n{" {}".format(", ".join(sample.labels))}\n=====')
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

