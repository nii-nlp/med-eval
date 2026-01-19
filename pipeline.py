import os
import random
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import set_seed
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import warnings
warnings.filterwarnings("once")


from tool_utils import is_main_process


class EvaluationPipeline:
    def __init__(self, args):
        self.args = args
        self.__setup_environment__()
        self.__task_specific_preparation__()
        self.__prepare_tokenizer_and_model__(self.args.model_name_or_path)

    def __setup_environment__(self):
        set_seed(self.args.seed)

        self.using_ddp = False
        try:
            dist.init_process_group(backend='nccl')
            self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(self.device)
            self.using_ddp = True
        except:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __prepare_tokenizer_and_model__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True
        )

        # 设置 model_max_length（用于截断）
        if hasattr(self.args, "model_max_length") and self.args.model_max_length is not None:
            self.tokenizer.model_max_length = self.args.model_max_length

        # pad token
        pad_token_not_exist = self.tokenizer.pad_token_id is None or self.tokenizer.pad_token is None
        if pad_token_not_exist:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.model_config = AutoConfig.from_pretrained(model_name_or_path)

        if model_name_or_path in ["meta-llama/Meta-Llama-3-8B", "hfl/llama-3-chinese-8b", "tokyotech-llm/Swallow-7b-hf", "epfl-llm/meditron-7b"]:
            self.model_config.torch_dtype = torch.float16

        if model_name_or_path in ["bigscience/mt0-small", "bigscience/mt0-xl", "facebook/nllb-200-distilled-600M"]:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                torch_dtype=getattr(self.model_config, "torch_dtype", None),
                use_cache=True
            )
        else:
            # 对于大模型，使用 device_map="auto" 自动分布到多个 GPU
            use_device_map = getattr(self.args, "use_device_map", False)
            if use_device_map:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=getattr(self.model_config, "torch_dtype", torch.bfloat16),
                    device_map="auto",
                    use_cache=True,
                    trust_remote_code=True
                )
                self.device = self.model.device  # 使用模型的设备
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=getattr(self.model_config, "torch_dtype", None),
                    use_cache=True
                )
        if pad_token_not_exist:
            self.model.resize_token_embeddings(len(self.tokenizer))

        if not getattr(self.args, "use_device_map", False):
            self.model = self.model.to(self.device)
        self.model.eval()

    def __task_specific_preparation__(self):
        self.load_samples_f = None
        self.dataset_f = None
        self.data_collator_f = None
        raise NotImplementedError

    def prepare_data(
        self,
        samples: List,
        demo_samples: List = None,
        template_name: str = None
    ):
        dataset = self.dataset_f(
            samples=samples,
            demo_samples=demo_samples,
            tokenizer=self.tokenizer,
            template_name=template_name,
            num_fewshot=self.args.num_fewshot if hasattr(self.args, "num_fewshot") else 0,
            truncate=self.args.truncate if hasattr(self.args, "truncate") else False
        )
        data_collator = self.data_collator_f(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            collate_fn=data_collator,
            sampler=DistributedSampler(dataset) if self.using_ddp else None,
            num_workers=self.args.num_workers,
            shuffle=False,
            drop_last=False
        )

        return dataset, dataloader

    def load_downstream_task(self, *args, **kwargs):
        dataset_name = kwargs.get("dataset_name", None)

        try:
            return self.load_samples_f(dataset_name)
        except:
            raise ValueError(f"Unknown task: {dataset_name}")

