import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.trainer_utils import set_seed
from vllm import LLM


class EvaluationPipeline:
    def __init__(self, args):
        self.args = args
        self.__setup_environment__()
        self.__prepare_tokenizer_and_model__(self.args.model_name_or_path)
        self.__task_specific_preparation__()

    def __setup_environment__(self):
        set_seed(self.args.seed)

    def __prepare_tokenizer_and_model__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left", trust_remote_code=True
        )

        # pad token
        pad_token_not_exist = (
            self.tokenizer.pad_token_id is None or self.tokenizer.pad_token is None
        )
        if pad_token_not_exist:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.model_config = AutoConfig.from_pretrained(model_name_or_path)

        if model_name_or_path in [
            "meta-llama/Meta-Llama-3-8B",
            "hfl/llama-3-chinese-8b",
            "tokyotech-llm/Swallow-7b-hf",
            "epfl-llm/meditron-7b",
        ]:
            self.model_config.torch_dtype = torch.float16
        self.model = LLM(
            model_name_or_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            tensor_parallel_size=1,
        )
        self.model.set_tokenizer(self.tokenizer)

    def __task_specific_preparation__(self):
        self.load_samples_f = None
        self.dataset_f = None
        self.data_collator_f = None
        raise NotImplementedError

    def prepare_data(
        self, samples: list, demo_samples: list = None, template_name: str = None
    ):
        dataset = self.dataset_f(
            samples=samples,
            demo_samples=demo_samples,
            tokenizer=self.tokenizer,
            template_name=template_name,
            num_fewshot=(
                self.args.num_fewshot if hasattr(self.args, "num_fewshot") else 0
            ),
            truncate=self.args.truncate if hasattr(self.args, "truncate") else False,
        )
        data_collator = self.data_collator_f(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            collate_fn=data_collator,
            num_workers=self.args.num_workers,
            shuffle=False,
            drop_last=False,
        )

        return dataset, dataloader

    def reconstruct_dataset(self,dataset: Dataset):
        if len(dataset["validation"]) != 0:
            # Validation data alerady prerared
            return dataset
        train_size = len(dataset["train"])
        test_size = len(dataset["test"])
        if train_size > 9 * test_size:
            dev_size = test_size
        elif train_size < (test_size // 2):
            dev_size = 5
        else:
            dev_size = train_size // 10
        dataset["validation"] = dataset["train"][-dev_size:]
        dataset["train"] = dataset["train"][:-dev_size]
        return dataset

    def load_downstream_task(self, *args, **kwargs):
        dataset_name = kwargs.get("dataset_name", None)

        try:
            dataset=self.load_samples_f(dataset_name)
            dataset=self.reconstruct_dataset(dataset)
            return dataset
        except:
            raise ValueError(f"Unknown task: {dataset_name}")
