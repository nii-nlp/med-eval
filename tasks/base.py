from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RequestDataset(Dataset):
    def __init__(
        self,
        samples: list,
        demo_samples: list = None,
        tokenizer: PreTrainedTokenizer = None,
        template_name: str = "",
        num_fewshot: int = 0,
        truncate: bool = False,
        *args,
        **kwargs,
    ):
        self.samples = samples
        self.demo_samples = demo_samples if demo_samples is not None else []

        self.template_name = template_name

        self.num_fewshot = num_fewshot
        assert self.num_fewshot == 0 or (
            self.num_fewshot > 0 and self.num_fewshot <= len(self.demo_samples)
        ), f"{self.num_fewshot} {len(self.demo_samples)}"

        self.tokenizer = tokenizer
        self.truncate = truncate

        self.__task_sepcific_preparation__()
        self.template = self.template_f(template_name)

        if self.tokenizer.name_or_path in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
            self.requests = self.construct_requests_chat()
        else:
            self.requests = self.construct_requests()

    def __task_sepcific_preparation__(self):
        self.template_f = None
        raise NotImplementedError

    def instantiate_template(self, sample):
        return self.template.instantiate_template(sample)

    def construct_requests(self):
        raise NotImplementedError

    def construct_requests_chat(self):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.requests[index]

    def __len__(self):
        return len(self.requests)

    @property
    def num_samples(self):
        return len(self.samples)
